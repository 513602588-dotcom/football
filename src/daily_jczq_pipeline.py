#!/usr/bin/env python3
"""Daily Sporttery (JCZQ) prediction pipeline.

Pipeline stages:
1) Crawl latest JCZQ fixtures (500.com) + Okooo history
2) Build Poisson + Elo + ML + bookmaker fusion probabilities
3) Generate dual-LLM reasoning (OpenAI relay + Gemini relay)
4) Export site JSON files used by GitHub Pages frontend
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from src.backtest.backtest import backtest
from src.collect import export_500, export_okooo, export_okooo_jczq, now_cn_date
from src.engine.value import calc, implied_prob, label, remove_overround, score
from src.models.bookmaker import predict_from_odds
from src.models.ml_ensemble import compute_latest_team_form, predict_proba, train_models
from src.models.poisson_elo import FitModels, fit_poisson, predict as predict_pe, run_elo
from src.models.upset import avoid_upset

OUT_DIR = Path("site/data")
PICKS_PATH = OUT_DIR / "picks.json"
TOP_PATH = OUT_DIR / "top_picks.json"
PREDICTIONS_PATH = OUT_DIR / "predictions.json"

FUTURE_DAYS = 2
TOP_N = 4
W_PE = 0.50
W_ML = 0.30
W_BM = 0.20
ODDS_SPORTS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]


@dataclass
class LLMConfig:
    openai_base: str
    openai_key: str
    openai_model: str
    gemini_base: str
    gemini_key: str
    gemini_model: str


def env_value(*keys: str, default: str = "") -> str:
    for k in keys:
        v = os.getenv(k, "").strip()
        if v:
            return v
    return default


def valid_key(v: str) -> bool:
    if not v:
        return False
    lv = v.lower()
    bad = ["your_", "_here", "changeme", "example", "placeholder"]
    return not any(x in lv for x in bad)


def parse_model_candidates(model_value: str) -> List[str]:
    raw_items = [x.strip() for x in str(model_value or "").split(",") if x.strip()]
    if not raw_items:
        raw_items = ["gpt-5"]

    aliases = {
        # Common typo from env/user input.
        "gtp-5.4": "gpt-5.4",
        "gtp-5": "gpt-5",
    }

    out: List[str] = []
    seen: Set[str] = set()

    def push(name: str) -> None:
        n = name.strip()
        if not n:
            return
        ln = n.lower()
        if ln in seen:
            return
        seen.add(ln)
        out.append(n)

    for item in raw_items:
        normalized = aliases.get(item.lower(), item)
        push(normalized)

        low = normalized.lower()
        # Expand short version aliases users often provide.
        if low in {"5.4", "gpt5.4", "gpt-5.4"}:
            push("gpt-5.4")
        if low in {"5.3", "gpt5.3", "gpt-5.3"}:
            push("gpt-5.3")
        if low in {"3.1", "gemini3.1", "gemini-3.1"}:
            push("gemini-3.1")
            push("gemini-3.1-pro")
        if low in {"3.0", "gemini3.0", "gemini-3.0"}:
            push("gemini-3.0")
            push("gemini-3.0-pro")

        # Add pragmatic fallbacks so a single unavailable model does not kill LLM output.
        if low.startswith("gpt"):
            push("gpt-5")
            push("gpt-4o-mini")
        if low.startswith("gemini"):
            push("gemini-2.5-pro")
            push("gemini-1.5-pro")

    return out


def _team_name_quality(name: str) -> bool:
    t = str(name or "").strip()
    if len(t) < 2:
        return False
    # Drop noise like pure numeric ids from fallback parsers.
    if re.fullmatch(r"\d+", t):
        return False
    alnum = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", t)
    if not alnum:
        return False
    digit_ratio = sum(ch.isdigit() for ch in alnum) / max(1, len(alnum))
    if digit_ratio >= 0.6:
        return False
    return True


def probe_external_connections() -> Dict[str, object]:
    out: Dict[str, object] = {}

    api_key = env_value("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY")
    if valid_key(api_key):
        try:
            r = requests.get(
                env_value("API_FOOTBALL_BASE", default="https://v3.football.api-sports.io").rstrip("/") + "/status",
                headers={"x-apisports-key": api_key},
                timeout=12,
            )
            out["api_football"] = {"ok": r.ok, "status": r.status_code}
        except Exception as exc:
            out["api_football"] = {"ok": False, "error": str(exc)}
    else:
        out["api_football"] = {"ok": False, "error": "missing_or_placeholder_key"}

    fdb_key = env_value("FOOTBALL_DATA_KEY", "FOOTBALL_DATA_API_KEY")
    if valid_key(fdb_key):
        try:
            r = requests.get(
                "https://api.football-data.org/v4/matches",
                headers={"X-Auth-Token": fdb_key},
                params={"dateFrom": now_cn_date(), "dateTo": now_cn_date()},
                timeout=12,
            )
            out["football_data"] = {"ok": r.ok, "status": r.status_code}
        except Exception as exc:
            out["football_data"] = {"ok": False, "error": str(exc)}
    else:
        out["football_data"] = {"ok": False, "error": "missing_or_placeholder_key"}

    odds_key = env_value("ODDS_API_KEY", "THE_ODDS_API_KEY")
    if valid_key(odds_key):
        try:
            r = requests.get(
                "https://api.the-odds-api.com/v4/sports",
                params={"apiKey": odds_key},
                timeout=12,
            )
            out["odds_api"] = {"ok": r.ok, "status": r.status_code}
        except Exception as exc:
            out["odds_api"] = {"ok": False, "error": str(exc)}
    else:
        out["odds_api"] = {"ok": False, "error": "missing_or_placeholder_key"}

    # Probe relay style chat API with tiny call.
    def _probe_llm(name: str, base: str, key: str, model: str) -> Dict[str, object]:
        if not valid_key(key):
            return {"ok": False, "error": "missing_or_placeholder_key"}
        try:
            r = requests.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                    "temperature": 0,
                },
                timeout=14,
            )
            return {"ok": r.ok, "status": r.status_code}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    openai_models = env_value("OPENAI_MODEL", default="gpt-5.4,gpt-5.3")
    gemini_models = env_value("GEMINI_MODEL", default="gemini-3.1-pro,gemini-3.0-pro")

    openai_probe = {"ok": False, "error": "all_model_candidates_failed"}
    for cand in parse_model_candidates(openai_models):
        probe_one = _probe_llm(
            "openai_relay",
            env_value("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_RELAY_URL", default="https://nan.meta-api.vip/v1"),
            env_value("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_RELAY_KEY"),
            cand,
        )
        probe_one["model"] = cand
        if probe_one.get("error") == "missing_or_placeholder_key":
            openai_probe = probe_one
            break
        if probe_one.get("ok"):
            openai_probe = probe_one
            break
        openai_probe = probe_one
    out["openai_relay"] = openai_probe

    gemini_probe = {"ok": False, "error": "all_model_candidates_failed"}
    for cand in parse_model_candidates(gemini_models):
        probe_one = _probe_llm(
            "gemini_relay",
            env_value("GEMINI_BASE_URL", "GEMINI_API_BASE", "GEMINI_RELAY_URL", default="https://once.novai.su/v1"),
            env_value("GEMINI_API_KEY", "GEMINI_KEY", "GEMINI_RELAY_KEY"),
            cand,
        )
        probe_one["model"] = cand
        if probe_one.get("error") == "missing_or_placeholder_key":
            gemini_probe = probe_one
            break
        if probe_one.get("ok"):
            gemini_probe = probe_one
            break
        gemini_probe = probe_one
    out["gemini_relay"] = gemini_probe
    return out


def load_runtime_env() -> None:
    # Keep committed .env/.env.example safe placeholders, and load private overrides from .env.local.
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=True)


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_history_df() -> pd.DataFrame:
    """Load Okooo history and map to model schema.

    Expected input columns from crawler:
    - date, home, away, score, odds_win, odds_draw, odds_lose
    """
    src_csv = OUT_DIR / "history_okooo.csv"
    if not src_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(src_csv)
    if df.empty:
        return df

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df.get("date"), errors="coerce")
    out["HomeTeam"] = df.get("home", "")
    out["AwayTeam"] = df.get("away", "")

    score_split = df.get("score", "").astype(str).str.extract(r"(\d+)\s*[-:]\s*(\d+)")
    out["FTHG"] = pd.to_numeric(score_split[0], errors="coerce")
    out["FTAG"] = pd.to_numeric(score_split[1], errors="coerce")

    # Backtest module expects B365 columns; we use Okooo SP as substitute.
    out["B365H"] = pd.to_numeric(df.get("odds_win"), errors="coerce")
    out["B365D"] = pd.to_numeric(df.get("odds_draw"), errors="coerce")
    out["B365A"] = pd.to_numeric(df.get("odds_lose"), errors="coerce")

    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    return out


def load_jczq_fixtures() -> pd.DataFrame:
    source_rows: List[Dict[str, object]] = []
    for fn in ["jczq_okooo.json", "jczq.json"]:
        p = OUT_DIR / fn
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rs = data.get("matches") or []
            # 优先保留澳客来源（通常 SP 更完整），500 作为补充。
            if fn == "jczq_okooo.json":
                source_rows = rs + source_rows
            else:
                source_rows += rs
        except Exception:
            continue

    rows = source_rows

    # 默认只分析竞彩（JCZQ）。需要全局赛事回退时可显式开启。
    allow_global_fallback = os.getenv("ALLOW_GLOBAL_FIXTURE_FALLBACK", "false").lower() == "true"
    if not rows and allow_global_fallback:
        rows = fetch_fallback_fixtures()
    if not rows:
        return pd.DataFrame()

    fx = pd.DataFrame(rows)
    fx["date"] = fx.get("date", "").astype(str)
    kick = fx.get("time", "").astype(str).str.extract(r"(\d{1,2}:\d{2})")[0].fillna("00:00")
    fx["Date"] = pd.to_datetime(fx["date"] + " " + kick, errors="coerce")
    fx = fx.rename(columns={"home": "HomeTeam", "away": "AwayTeam", "league": "League"})
    fx["source"] = fx.get("source", "")
    fx["odds_win"] = pd.to_numeric(fx.get("odds_win"), errors="coerce")
    fx["odds_draw"] = pd.to_numeric(fx.get("odds_draw"), errors="coerce")
    fx["odds_lose"] = pd.to_numeric(fx.get("odds_lose"), errors="coerce")

    today = datetime.strptime(now_cn_date(), "%Y-%m-%d")
    upper = today + timedelta(days=FUTURE_DAYS)

    fx = fx.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    fx = fx[
        fx["HomeTeam"].astype(str).map(_team_name_quality)
        & fx["AwayTeam"].astype(str).map(_team_name_quality)
    ].copy()
    # 先按比赛键去重，澳客（okooo）优先。
    fx["_priority"] = fx["source"].astype(str).str.contains("okooo", case=False, na=False).astype(int)
    fx = fx.sort_values(["_priority", "Date"], ascending=[False, True])
    fx = fx.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"], keep="first").copy()
    in_window = fx[(fx["Date"] >= today) & (fx["Date"] <= upper)].copy()
    if not in_window.empty:
        return in_window.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)

    # 回退策略：若当天窗口无比赛，使用最近可用赛程，避免页面完全空白。
    fx = fx.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)
    return fx.tail(30).reset_index(drop=True)


def _norm_team(name: str) -> str:
    n = (name or "").strip().lower()
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", n)
    return n


def fetch_api_sports_fixtures(start: datetime, end: datetime) -> List[Dict[str, object]]:
    key = env_value("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY")
    if not valid_key(key):
        return []

    base = env_value("API_FOOTBALL_BASE", default="https://v3.football.api-sports.io")
    out: List[Dict[str, object]] = []
    day = start
    while day <= end:
        ds = day.strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                base.rstrip("/") + "/fixtures",
                headers={"x-apisports-key": key},
                params={"date": ds, "timezone": "Asia/Shanghai"},
                timeout=20,
            )
            resp.raise_for_status()
            items = (resp.json() or {}).get("response") or []
            for m in items:
                league = ((m.get("league") or {}).get("name")) or "竞彩"
                teams = m.get("teams") or {}
                home = ((teams.get("home") or {}).get("name")) or ""
                away = ((teams.get("away") or {}).get("name")) or ""
                fixture = m.get("fixture") or {}
                dttm = (fixture.get("date") or "")[:16].replace("T", " ")
                out.append(
                    {
                        "date": ds,
                        "time": dttm[-5:] if len(dttm) >= 5 else "00:00",
                        "league": league,
                        "home": home,
                        "away": away,
                        "odds_win": None,
                        "odds_draw": None,
                        "odds_lose": None,
                        "source": "api-football",
                    }
                )
        except Exception:
            pass
        day += timedelta(days=1)
    return out


def fetch_football_data_fixtures(start: datetime, end: datetime) -> List[Dict[str, object]]:
    key = env_value("FOOTBALL_DATA_KEY", "FOOTBALL_DATA_API_KEY")
    if not valid_key(key):
        return []

    url = "https://api.football-data.org/v4/matches"
    out: List[Dict[str, object]] = []
    try:
        resp = requests.get(
            url,
            headers={"X-Auth-Token": key},
            params={
                "dateFrom": start.strftime("%Y-%m-%d"),
                "dateTo": end.strftime("%Y-%m-%d"),
            },
            timeout=20,
        )
        resp.raise_for_status()
        items = (resp.json() or {}).get("matches") or []
        for m in items:
            utc = m.get("utcDate", "")
            date = utc[:10] if len(utc) >= 10 else start.strftime("%Y-%m-%d")
            tm = utc[11:16] if len(utc) >= 16 else "00:00"
            out.append(
                {
                    "date": date,
                    "time": tm,
                    "league": ((m.get("competition") or {}).get("name")) or "竞彩",
                    "home": ((m.get("homeTeam") or {}).get("name")) or "",
                    "away": ((m.get("awayTeam") or {}).get("name")) or "",
                    "odds_win": None,
                    "odds_draw": None,
                    "odds_lose": None,
                    "source": "football-data",
                }
            )
    except Exception:
        return []
    return out


def fetch_fallback_fixtures() -> List[Dict[str, object]]:
    today = datetime.strptime(now_cn_date(), "%Y-%m-%d")
    start = today - timedelta(days=1)
    upper = today + timedelta(days=max(FUTURE_DAYS, 4))
    rows = fetch_api_sports_fixtures(start, upper)
    rows += fetch_football_data_fixtures(start, upper)

    seen: Set[Tuple[str, str, str]] = set()
    dedup: List[Dict[str, object]] = []
    for r in rows:
        key = (str(r.get("date", "")), _norm_team(str(r.get("home", ""))), _norm_team(str(r.get("away", ""))) )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def build_odds_lookup() -> Dict[Tuple[str, str], Tuple[Optional[float], Optional[float], Optional[float]]]:
    key = env_value("ODDS_API_KEY", "THE_ODDS_API_KEY")
    if not valid_key(key):
        return {}

    lookup: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    base = "https://api.the-odds-api.com/v4/sports"
    for sport in ODDS_SPORTS:
        try:
            resp = requests.get(
                f"{base}/{sport}/odds",
                params={
                    "apiKey": key,
                    "regions": "uk,eu",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=20,
            )
            resp.raise_for_status()
            events = resp.json() or []
            for e in events:
                home = str(e.get("home_team") or "")
                away = ""
                for t in e.get("teams") or []:
                    if t != home:
                        away = t
                        break
                if not home or not away:
                    continue

                oh = od = oa = None
                for bk in e.get("bookmakers") or []:
                    for mk in bk.get("markets") or []:
                        if mk.get("key") != "h2h":
                            continue
                        for o in mk.get("outcomes") or []:
                            nm = str(o.get("name") or "")
                            pr = o.get("price")
                            if nm == home:
                                oh = float(pr)
                            elif nm == away:
                                oa = float(pr)
                            else:
                                od = float(pr)
                    if oh and oa:
                        break

                lookup[(_norm_team(home), _norm_team(away))] = (oh, od, oa)
        except Exception:
            continue
    return lookup


def fuse_probs(
    pe: Tuple[float, float, float],
    ml: Optional[Tuple[float, float, float]],
    bm: Optional[Tuple[float, float, float]],
) -> Tuple[float, float, float, Dict[str, float]]:
    weights = {"pe": W_PE, "ml": W_ML if ml else 0.0, "bm": W_BM if bm else 0.0}
    ws = weights["pe"] + weights["ml"] + weights["bm"]
    if ws <= 0:
        return pe[0], pe[1], pe[2], weights

    ph = (weights["pe"] * pe[0] + weights["ml"] * (ml[0] if ml else 0.0) + weights["bm"] * (bm[0] if bm else 0.0)) / ws
    pd_ = (weights["pe"] * pe[1] + weights["ml"] * (ml[1] if ml else 0.0) + weights["bm"] * (bm[1] if bm else 0.0)) / ws
    pa = (weights["pe"] * pe[2] + weights["ml"] * (ml[2] if ml else 0.0) + weights["bm"] * (bm[2] if bm else 0.0)) / ws

    ph, pd_, pa = avoid_upset(ph, pd_, pa)
    s = ph + pd_ + pa
    if s <= 0:
        return pe[0], pe[1], pe[2], weights
    return ph / s, pd_ / s, pa / s, weights


def safe_predict_pe(models: Optional[FitModels], home: str, away: str) -> Dict[str, float]:
    if not models:
        return {
            "p_home": 0.45,
            "p_draw": 0.28,
            "p_away": 0.27,
            "xg_home": 1.40,
            "xg_away": 1.12,
            "most_likely_score": "2-1",
        }
    return predict_pe(models, home, away)


def estimate_xg_from_probs(ph: float, pd_: float, pa: float) -> Tuple[float, float]:
    total = 2.45
    xh = total * (ph + 0.5 * pd_)
    xa = total * (pa + 0.5 * pd_)
    xh = max(0.55, min(2.9, xh))
    xa = max(0.45, min(2.7, xa))
    return round(xh, 2), round(xa, 2)


def estimate_scoreline(ph: float, pd_: float, pa: float) -> str:
    if pd_ >= max(ph, pa):
        return "1-1" if pd_ > 0.30 else "0-0"

    if ph >= pa:
        gap = ph - pa
        if gap > 0.22:
            return "3-1"
        if gap > 0.10:
            return "2-1"
        return "1-0"

    gap = pa - ph
    if gap > 0.22:
        return "1-3"
    if gap > 0.10:
        return "1-2"
    return "0-1"


def llm_chat_completion(base: str, key: str, model: str, prompt: str) -> Optional[str]:
    if not base or not key:
        return None

    url = base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def clean_llm_text(txt: str) -> Optional[str]:
        t = re.sub(r"\s+", " ", str(txt or "")).strip()
        if not t:
            return None
        if len(t) < 10:
            return None
        if t in {"战术", "赔率", "分析", "建议", "可博", "主胜", "客胜", "平局"}:
            return None
        # Prefer outputs with Chinese content; avoid accidental empty/english-only fragments.
        if len(re.findall(r"[\u4e00-\u9fff]", t)) < 4:
            return None
        return t[:180]

    for m in parse_model_candidates(model):
        payload = {
            "model": m,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a concise football analyst. "
                        "Reply in Chinese with exactly 2 short lines: "
                        "line1 tactical reason, line2 value/odds reason."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 120,
        }

        for attempt in range(1, 4):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=25)
                if resp.ok:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if not choices:
                        break
                    txt = (choices[0].get("message") or {}).get("content", "")
                    cleaned = clean_llm_text(txt)
                    if cleaned:
                        return cleaned
                    break

                # Retry transient gateway/rate-limit errors; otherwise move to next model.
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    time.sleep(0.7 * attempt)
                    continue
                break
            except Exception:
                if attempt < 3:
                    time.sleep(0.7 * attempt)
                    continue
                break
    return None


def build_llm_reason(cfg: LLMConfig, pick: Dict[str, object]) -> Tuple[str, str, Optional[str], Optional[str]]:
    prompt = (
        f"比赛: {pick['home']} vs {pick['away']}\n"
        f"概率: 主{pick['p_home']:.2f} 平{pick['p_draw']:.2f} 客{pick['p_away']:.2f}\n"
        f"xG: {pick['xg_home']:.2f}-{pick['xg_away']:.2f}\n"
        f"推荐: {pick['pick']} EV={pick.get('ev', 0)}\n"
        "请输出两行: 第1行写战术面理由; 第2行写赔率价值面理由。"
    )

    gpt_text = llm_chat_completion(cfg.openai_base, cfg.openai_key, cfg.openai_model, prompt)
    gem_text = llm_chat_completion(cfg.gemini_base, cfg.gemini_key, cfg.gemini_model, prompt)

    if gpt_text and gem_text:
        return f"GPT: {gpt_text} | Gemini: {gem_text}", "both", gpt_text, gem_text
    if gpt_text:
        return f"GPT: {gpt_text}", "openai", gpt_text, None
    if gem_text:
        return f"Gemini: {gem_text}", "gemini", None, gem_text

    fallback = "模型共识: 主队进攻效率更优, 概率与赔率存在正EV区间。"
    return fallback, "fallback", None, None


def build_prediction_rows(fx: pd.DataFrame, history: pd.DataFrame) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    # Model training is optional; when data is insufficient we still produce predictions.
    pe_models: Optional[FitModels] = None
    ml_models = None
    team_form: Dict[str, Dict[str, float]] = {}

    played = history.dropna(subset=["FTHG", "FTAG"]).copy() if not history.empty else pd.DataFrame()
    if len(played) >= 50:
        elo = run_elo(played[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy())
        mh, ma = fit_poisson(played[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy())
        pe_models = FitModels(home=mh, away=ma, elo=elo)

    if len(played) >= 800:
        ml_models = train_models(played)
        if ml_models is not None:
            team_form = compute_latest_team_form(played)

    rows: List[Dict[str, object]] = []
    odds_lookup = build_odds_lookup()

    for _, r in fx.iterrows():
        home = str(r.get("HomeTeam", "")).strip()
        away = str(r.get("AwayTeam", "")).strip()
        if not home or not away:
            continue

        pe = safe_predict_pe(pe_models, home, away)
        pe_p = (float(pe["p_home"]), float(pe["p_draw"]), float(pe["p_away"]))
        ml_p = predict_proba(ml_models, team_form, home, away) if ml_models else None

        odds = (
            float(r["odds_win"]) if pd.notna(r.get("odds_win")) else None,
            float(r["odds_draw"]) if pd.notna(r.get("odds_draw")) else None,
            float(r["odds_lose"]) if pd.notna(r.get("odds_lose")) else None,
        )
        if not all(odds):
            odds = odds_lookup.get((_norm_team(home), _norm_team(away)), odds)
        bm_p = predict_from_odds(odds) if all(odds) else None

        ph, pd_, pa, weights = fuse_probs(pe_p, ml_p, bm_p)
        dyn_xg_home, dyn_xg_away = estimate_xg_from_probs(ph, pd_, pa)
        dyn_score = estimate_scoreline(ph, pd_, pa)

        evv = None
        kellyv = None
        pick = "模型"
        pick_score = None
        status = "-"

        if all(odds):
            q1, qx, q2 = implied_prob(odds[0]), implied_prob(odds[1]), implied_prob(odds[2])
            f1, fx_, f2 = remove_overround(q1, qx, q2)
            best = max(
                [
                    calc(ph, odds[0], f1, "主胜"),
                    calc(pd_, odds[1], fx_, "平"),
                    calc(pa, odds[2], f2, "客胜"),
                ],
                key=lambda x: x.ev,
            )
            evv = round(best.ev, 4)
            kellyv = round(min(best.kelly, 0.08), 4)
            pick = best.pick
            pick_score = score(best)
            status = label(pick_score)

        dt = pd.to_datetime(r.get("Date"), errors="coerce")
        kick_time = str(r.get("time", "")).strip()
        most_likely_score = pe.get("most_likely_score", "") if pe_models is not None else ""
        if not most_likely_score or most_likely_score == "2-1":
            most_likely_score = dyn_score

        xg_home = round(float(pe.get("xg_home", dyn_xg_home)), 2) if pe_models is not None else dyn_xg_home
        xg_away = round(float(pe.get("xg_away", dyn_xg_away)), 2) if pe_models is not None else dyn_xg_away

        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d") if not pd.isna(dt) else str(r.get("date", "")),
                "time": kick_time,
                "league": r.get("League", "竞彩"),
                "home": home,
                "away": away,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "p_home": round(ph, 4),
                "p_draw": round(pd_, 4),
                "p_away": round(pa, 4),
                "pe_p": [round(pe_p[0], 4), round(pe_p[1], 4), round(pe_p[2], 4)],
                "ml_p": [round(ml_p[0], 4), round(ml_p[1], 4), round(ml_p[2], 4)] if ml_p else None,
                "bm_p": [round(bm_p[0], 4), round(bm_p[1], 4), round(bm_p[2], 4)] if bm_p else None,
                "most_likely_score": most_likely_score,
                "odds_win": odds[0],
                "odds_draw": odds[1],
                "odds_lose": odds[2],
                "ev": evv,
                "kelly": kellyv,
                "pick": pick,
                "score": pick_score,
                "label": status,
                "why": (
                    f"融合权重 PE:{weights['pe']:.2f} ML:{weights['ml']:.2f} BM:{weights['bm']:.2f}; "
                    f"主胜率{ph*100:.1f}%, xG差{float(pe.get('xg_home', 1.4)) - float(pe.get('xg_away', 1.1)):.2f}"
                ),
            }
        )

    # Minimal backtest to keep dashboard metrics stable.
    bt = {"matches_used": 0, "bets": 0, "roi": 0.0, "hit_rate": 0.0, "avg_ev": 0.0, "logloss": 0.0}
    if len(played) > 0 and pe_models is not None:
        bt = backtest(
            played,
            lambda h, a: safe_predict_pe(pe_models, h, a),
            ev_threshold=0.03,
        )

    return rows, bt


def build_payload(rows: List[Dict[str, object]], bt: Dict[str, object], llm_cfg: LLMConfig) -> Dict[str, object]:
    ranked = [x for x in rows if x.get("ev") is not None]
    ranked.sort(key=lambda x: (x.get("score") or 0, x.get("ev") or -9), reverse=True)

    top = ranked[:TOP_N] if ranked else rows[:TOP_N]

    llm_used = {"both": 0, "openai": 0, "gemini": 0, "fallback": 0}
    for p in top:
        base_reason = str(p.get("why", ""))
        llm_reason, llm_status, gpt_reason, gemini_reason = build_llm_reason(llm_cfg, p)
        llm_used[llm_status] = llm_used.get(llm_status, 0) + 1
        p["why"] = f"{base_reason} | {llm_reason}"
        p["llm_status"] = llm_status
        p["reasons"] = {
            "base": base_reason,
            "gpt": gpt_reason,
            "gemini": gemini_reason,
            "fallback": llm_reason if llm_status == "fallback" else None,
        }

    return {
        "meta": {
            "generated_at_utc": utc_now_str(),
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "seasons_used": ["JCZQ+OKOOO"],
            "fusion": {
                "W_PE": W_PE,
                "W_ML": W_ML,
                "W_BM": W_BM,
                "ml_enabled": any(x.get("ml_p") for x in rows),
            },
            "llm": {
                "openai_model": llm_cfg.openai_model,
                "gemini_model": llm_cfg.gemini_model,
                "usage": llm_used,
            },
            "scope": "Only China Sporttery JCZQ",
            "schedule_bjt": ["09:30", "21:30"],
        },
        "stats": {
            "fixtures": len(rows),
            "top": len(top),
            "backtest": bt,
        },
        "top_picks": top,
        "all": rows,
    }


def write_outputs(payload: Dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    TOP_PATH.write_text(json.dumps(payload.get("top_picks", []), ensure_ascii=False, indent=2), encoding="utf-8")
    PREDICTIONS_PATH.write_text(json.dumps(payload.get("all", []), ensure_ascii=False, indent=2), encoding="utf-8")

    # Backward compatibility with old files.
    (OUT_DIR / "picks_updated.json").write_text(
        json.dumps(payload.get("top_picks", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "complete_predictions.json").write_text(
        json.dumps(payload.get("all", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_llm_config() -> LLMConfig:
    openai_key = env_value("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_RELAY_KEY")
    gemini_key = env_value("GEMINI_API_KEY", "GEMINI_KEY", "GEMINI_RELAY_KEY")
    return LLMConfig(
        openai_base=env_value("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_RELAY_URL", default="https://nan.meta-api.vip/v1"),
        openai_key=openai_key if valid_key(openai_key) else "",
        # Keep env-overridable; first candidate is preferred, others are auto fallback.
        openai_model=env_value("OPENAI_MODEL", default="gpt-5.4,gpt-5.3"),
        gemini_base=env_value("GEMINI_BASE_URL", "GEMINI_API_BASE", "GEMINI_RELAY_URL", default="https://once.novai.su/v1"),
        gemini_key=gemini_key if valid_key(gemini_key) else "",
        gemini_model=env_value("GEMINI_MODEL", default="gemini-3.1-pro,gemini-3.0-pro"),
    )


def run() -> int:
    load_runtime_env()
    Path("site").mkdir(parents=True, exist_ok=True)
    Path("site/.nojekyll").write_text("", encoding="utf-8")

    api_football_on = valid_key(env_value("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY"))
    football_data_on = valid_key(env_value("FOOTBALL_DATA_KEY", "FOOTBALL_DATA_API_KEY"))
    odds_api_on = valid_key(env_value("ODDS_API_KEY", "THE_ODDS_API_KEY"))
    llm_openai_on = valid_key(env_value("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_RELAY_KEY"))
    llm_gemini_on = valid_key(env_value("GEMINI_API_KEY", "GEMINI_KEY", "GEMINI_RELAY_KEY"))

    print(
        "[env] api_football=%s football_data=%s odds_api=%s openai=%s gemini=%s"
        % (
            api_football_on,
            football_data_on,
            odds_api_on,
            llm_openai_on,
            llm_gemini_on,
        )
    )

    if not (api_football_on or football_data_on or odds_api_on or llm_openai_on or llm_gemini_on):
        print("WARN all keys are missing/placeholder. Put real keys in .env.local (local) or GitHub Actions Secrets (CI).")

    probe = probe_external_connections()
    print("[probe]", json.dumps(probe, ensure_ascii=False))

    print(f"[1/4] crawl start {utc_now_str()}")
    try:
        export_500(days=4, direction="future")
    except Exception as exc:
        print(f"WARN export_500 failed: {exc}")

    try:
        payload_okooo = export_okooo_jczq()
        (OUT_DIR / "jczq_okooo.json").write_text(json.dumps(payload_okooo, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"INFO okooo_jczq count={payload_okooo.get('meta', {}).get('count', 0)}")
    except Exception as exc:
        print(f"WARN export_okooo_jczq failed: {exc}")

    try:
        export_okooo(start_date=now_cn_date(), days=10, version="full")
    except Exception as exc:
        print(f"WARN export_okooo failed: {exc}")

    print("[2/4] load datasets")
    fx = load_jczq_fixtures()
    history = load_history_df()

    if fx.empty:
        print("ERROR: no JCZQ fixtures found")

        # 若抓取暂时不可用，则复用上一次有效结果，保证 GitHub Pages 正常发布。
        if PICKS_PATH.exists():
            try:
                old_payload = json.loads(PICKS_PATH.read_text(encoding="utf-8"))
                old_all = old_payload.get("all") or []
                if old_all:
                    old_payload.setdefault("meta", {})["generated_at_utc"] = utc_now_str()
                    old_payload.setdefault("meta", {})["warning"] = "No fresh fixtures, reused last successful snapshot"
                    write_outputs(old_payload)
                    print(f"DONE reused previous snapshot all={len(old_all)}")
                    return 0
            except Exception as exc:
                print(f"WARN reuse previous snapshot failed: {exc}")

        payload = {
            "meta": {
                "generated_at_utc": utc_now_str(),
                "scope": "Only China Sporttery JCZQ",
                "error": "No fixtures from crawler",
                "warning": "Published empty snapshot to keep pipeline green",
            },
            "stats": {"fixtures": 0, "top": 0, "backtest": {"matches_used": 0, "bets": 0, "roi": 0.0, "hit_rate": 0.0, "avg_ev": 0.0, "logloss": 0.0}},
            "top_picks": [],
            "all": [],
        }
        write_outputs(payload)
        return 0

    print(f"[3/4] model inference fixtures={len(fx)} history={len(history)}")
    rows, bt = build_prediction_rows(fx, history)

    print("[4/4] llm fusion + export")
    llm_cfg = load_llm_config()
    payload = build_payload(rows, bt, llm_cfg)
    payload.setdefault("meta", {})["api_usage"] = {
        "api_football_enabled": api_football_on,
        "football_data_enabled": football_data_on,
        "odds_api_enabled": odds_api_on,
        "allow_global_fixture_fallback": os.getenv("ALLOW_GLOBAL_FIXTURE_FALLBACK", "false").lower() == "true",
    }
    payload.setdefault("meta", {})["connection_probe"] = probe
    write_outputs(payload)

    print(f"DONE top={len(payload['top_picks'])} all={len(payload['all'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
