import os
import json
import pandas as pd
from datetime import datetime, timezone

from src.data.sources import LEAGUES, season_code_for, prev_season, fetch_league, split_played_future, fetch_fixtures_fallback, pick_1x2_odds
from src.models.poisson_elo import run_elo, fit_poisson, FitModels, predict as predict_pe
from src.models.ml_ensemble import train_models, compute_latest_team_form, predict_proba
from src.engine.value import implied_prob, remove_overround, calc, score, label
from src.backtest.backtest import backtest

FUTURE_WINDOW_DAYS = 21
EV_THRESHOLD_BT = 0.03

# 融合权重：Poisson/Elo vs ML
W_PE = 0.60
W_ML = 0.40

def fuse_probs(pe: tuple[float,float,float], ml: tuple[float,float,float] | None) -> tuple[float,float,float]:
    if ml is None:
        return pe
    ph = W_PE*pe[0] + W_ML*ml[0]
    pd = W_PE*pe[1] + W_ML*ml[1]
    pa = W_PE*pe[2] + W_ML*ml[2]
    s = ph+pd+pa
    return (ph/s, pd/s, pa/s) if s>0 else pe

def main():
    os.makedirs("site/data", exist_ok=True)
    with open("site/.nojekyll","w",encoding="utf-8") as f:
        f.write("")

    now = datetime.now(timezone.utc)
    sc = season_code_for(now.date())
    sc_prev = prev_season(sc, 1)

    played_parts = []
    future_parts = []

    # 1) 拉数据（当季+上季训练）
    for div, lname in LEAGUES.items():
        for season in [sc, sc_prev]:
            try:
                df = fetch_league(div, season)
                played, future = split_played_future(df)
                played["Div"] = div
                played["League"] = lname
                future["Div"] = div
                future["League"] = lname
                played_parts.append(played)
                if season == sc:
                    future_parts.append(future)
            except Exception as e:
                print("WARN fetch failed:", div, season, e)

    if not played_parts:
        raise RuntimeError("No played data fetched")

    played_df = pd.concat(played_parts, ignore_index=True)

    # 2) 训练 Poisson/Elo
    elo = run_elo(played_df[["Date","HomeTeam","AwayTeam","FTHG","FTAG"]].copy())
    m_h, m_a = fit_poisson(played_df[["HomeTeam","AwayTeam","FTHG","FTAG"]].copy())
    pe_models = FitModels(home=m_h, away=m_a, elo=elo)

    # 3) 训练 ML（RF+MLP）
    ml_models = train_models(played_df[["Date","HomeTeam","AwayTeam","FTHG","FTAG"]].copy())
    team_form = compute_latest_team_form(played_df[["Date","HomeTeam","AwayTeam","FTHG","FTAG"]].copy()) if ml_models else {}

    # 4) 未来赛程：当季 CSV 的 future，取不到则 fixtures.csv 兜底
    fx = pd.DataFrame()
    if future_parts:
        fx = pd.concat(future_parts, ignore_index=True)
        fx = fx.dropna(subset=["Date","HomeTeam","AwayTeam"]).copy()

    if fx.empty:
        try:
            fx = fetch_fixtures_fallback()
            print("INFO using fixtures.csv fallback")
        except Exception as e:
            print("WARN fixtures fallback failed:", e)
            fx = pd.DataFrame()

    if not fx.empty:
        fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce")
        fx = fx.dropna(subset=["Date"]).copy()
        fx = fx[fx["Date"] >= pd.Timestamp(now.date())]
        fx = fx[fx["Date"] <= pd.Timestamp(now.date()) + pd.Timedelta(days=FUTURE_WINDOW_DAYS)]
        fx = fx.sort_values(["Date","League","HomeTeam"])

    rows = []
    for _, r in fx.iterrows():
        home, away = str(r["HomeTeam"]), str(r["AwayTeam"])
        pe = predict_pe(pe_models, home, away)
        pe_probs = (pe["p_home"], pe["p_draw"], pe["p_away"])

        ml_probs = None
        if ml_models:
            ml_probs = predict_proba(ml_models, team_form, home, away)

        ph, pd_, pa = fuse_probs(pe_probs, ml_probs)

        oh, od, oa, book = pick_1x2_odds(r)

        best = None
        s = None
        lab = "-"
        pick = None
        evv = None
        kellyv = None

        why = f"PE xG {pe['xg_home']:.2f}-{pe['xg_away']:.2f} | EloΔ {pe['elo_diff']:.0f}"
        if ml_probs is not None:
            why += f" | ML {ml_probs[0]:.2f}/{ml_probs[1]:.2f}/{ml_probs[2]:.2f}"

        if oh and od and oa:
            q1, qx, q2 = implied_prob(oh), implied_prob(od), implied_prob(oa)
            f1, fx_, f2 = remove_overround(q1, qx, q2)

            c1 = calc(ph, oh, f1, "主胜")
            cx = calc(pd_, od, fx_, "平")
            c2 = calc(pa, oa, f2, "客胜")
            best = max([c1, cx, c2], key=lambda x: x.ev)

            s = score(best)
            lab = label(s)
            pick = best.pick
            evv = round(best.ev, 4)
            kellyv = round(min(best.kelly, 0.08), 4)
            why += f" | Odds({book}) {oh}/{od}/{oa} | Gap {best.gap:+.3f}"
        else:
            # 没赔率也展示模型结果
            lab = "模型"

        rows.append({
            "date": str(pd.to_datetime(r["Date"]).date()),
            "league": r.get("League",""),
            "div": r.get("Div",""),
            "time": str(r.get("Time","")) if "Time" in r else "",
            "home": home,
            "away": away,
            "xg_home": round(pe["xg_home"], 2),
            "xg_away": round(pe["xg_away"], 2),
            "p_home": round(ph, 4),
            "p_draw": round(pd_, 4),
            "p_away": round(pa, 4),
            "p_over25": round(pe["p_over25"], 4),
            "p_btts": round(pe["p_btts"], 4),
            "most_likely_score": pe["most_likely_score"],
            "odds_win": oh,
            "odds_draw": od,
            "odds_lose": oa,
            "book": book,
            "pick": pick,
            "ev": evv,
            "kelly": kellyv,
            "score": s,
            "label": lab,
            "why": why,
        })

    # 5) 回测（只用 Poisson/Elo，保持稳定；你想也可以换成融合概率）
    bt = backtest(
        played_df,
        lambda h,a: predict_pe(pe_models, h, a),
        ev_threshold=EV_THRESHOLD_BT
    )

    top = sorted([x for x in rows if x.get("score") is not None], key=lambda z: z["score"], reverse=True)[:50]

    payload = {
        "meta": {
            "generated_at_utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "python": "3.12",
            "seasons_used": [sc, sc_prev],
            "window_days": FUTURE_WINDOW_DAYS,
            "fusion": {"W_PE": W_PE, "W_ML": W_ML, "ml_enabled": bool(ml_models)},
            "note": "Full: football-data + fixtures fallback + Poisson/Elo + RF+MLP + fusion + EV/Kelly + backtest",
        },
        "stats": {
            "fixtures": len(rows),
            "top": len(top),
            "backtest": bt,
        },
        "top_picks": top,
        "all": rows,
    }

    with open("site/data/picks.json","w",encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("OK: site/data/picks.json written. fixtures=", len(rows), "top=", len(top))

if __name__ == "__main__":
    main()
