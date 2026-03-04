import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .utils import now_cn_date, to_float, safe_read_html, HEADERS as BASE_HEADERS, decode_response

# 特定来源额外Referer
HEADERS = {**BASE_HEADERS, "Referer": "https://www.okooo.cn/"}

# _now_cn_date 已迁移到 utils.now_cn_date


# _safe_read_html 已迁移到 utils.safe_read_html


# _to_float 已迁移到 utils.to_float

def _guess_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    由于澳客列名可能变化，这里做“弱匹配猜列”
    返回映射：home/away/time/league/score/sp3/sp1/sp0
    """
    cols = [str(c) for c in df.columns]
    low = [c.lower() for c in cols]

    def pick(*keys):
        for k in keys:
            for i,c in enumerate(cols):
                if k in low[i]:
                    return c
        return ""

    return {
        "league": pick("联赛","赛事","league","match"),
        "time":   pick("时间","开赛","time","kick"),
        "home":   pick("主队","home","主"),
        "away":   pick("客队","away","客"),
        "score":  pick("比分","score"),
        # SP 胜平负通常叫：SP(胜) / 胜SP / 主胜 / 3
        "sp3":    pick("sp胜","胜sp","主胜","sp(胜)","3"),
        "sp1":    pick("sp平","平sp","平局","sp(平)","1"),
        "sp0":    pick("sp负","负sp","客胜","sp(负)","0"),
    }

def _normalize(df: pd.DataFrame, date_str: str, source: str) -> pd.DataFrame:
    m = _guess_cols(df)

    out = pd.DataFrame()
    out["date"] = date_str
    out["source"] = source

    out["league"] = df[m["league"]] if m["league"] in df.columns else ""
    out["time"]   = df[m["time"]] if m["time"] in df.columns else ""
    out["home"]   = df[m["home"]] if m["home"] in df.columns else ""
    out["away"]   = df[m["away"]] if m["away"] in df.columns else ""
    out["score"]  = df[m["score"]] if m["score"] in df.columns else ""

    out["odds_win"]  = df[m["sp3"]].map(to_float) if m["sp3"] in df.columns else None
    out["odds_draw"] = df[m["sp1"]].map(to_float) if m["sp1"] in df.columns else None
    out["odds_lose"] = df[m["sp0"]].map(to_float) if m["sp0"] in df.columns else None

    # 简单清洗：丢掉没主客队的行
    out = out[(out["home"].astype(str).str.len() > 0) & (out["away"].astype(str).str.len() > 0)]
    return out.reset_index(drop=True)

def fetch_day(date_str: str, version: str = "full") -> Optional[pd.DataFrame]:
    """抓取指定日期的澳客历史数据，返回已规范化的DataFrame。"""
    if version == "full":
        url = f"https://www.okooo.cn/jingcai/{date_str}/"
        src = "okooo_full"
    else:
        url = f"https://m.okooo.com/kaijiang/sport.php?LotteryType=SportteryNWDL&LotteryNo={date_str}"
        src = "okooo_simple"

    r = requests.get(url, headers=HEADERS, timeout=20)
    html = decode_response(r)
    tables = safe_read_html(html)
    if not tables:
        return None

    df0 = tables[0]
    best = max(tables, key=lambda d: len(d), default=df0)
    return _normalize(best, date_str=date_str, source=src)

def export_history(start_date: str, days: int = 7, version: str = "full") -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    frames = []
    for i in range(days):
        d = (start - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            df = fetch_day(d, version=version)
            if df is not None and len(df) > 0:
                frames.append(df)
                print(f"✅ {d} ok, rows={len(df)}")
            else:
                print(f"⚠️ {d} empty")
        except Exception as e:
            print(f"❌ {d} failed: {e}")
        time.sleep(random.uniform(1.0, 2.2))

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["date","league","time","home","away","score","odds_win","odds_draw","odds_lose","source"])

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=now_cn_date(), help="YYYY-MM-DD (China date)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--version", choices=["full","simple"], default="full")
    args = ap.parse_args()

    out_json = Path("site/data/history_okooo.json")
    out_csv  = Path("site/data/history_okooo.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # 旧数据兜底
    old = None
    if out_json.exists():
        try:
            old = json.loads(out_json.read_text(encoding="utf-8"))
        except:
            old = None

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": "okooo.cn",
        "start": args.start,
        "days": args.days,
        "version": args.version,
    }

    try:
        df = export_history(args.start, args.days, args.version)
        records = df.to_dict("records")
        meta["count"] = len(records)

        payload = {"meta": meta, "matches": records}
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("OK:", out_json, "count=", meta["count"])
    except Exception as e:
        # 不让 Actions 炸：保留旧文件
        if old:
            old.setdefault("meta", {})["error"] = str(e)
            out_json.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
            print("WARN kept old history_okooo.json error=", e)
        else:
            payload = {"meta": {**meta, "count": 0, "error": str(e)}, "matches": []}
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print("ERROR wrote empty history_okooo.json error=", e)

if __name__ == "__main__":
    main()
