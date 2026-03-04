import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from .utils import decode_response

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

HEADERS = {
    "User-Agent": UA,
    "Referer": "https://trade.500.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _to_float(x: str) -> Optional[float]:
    try:
        x = (x or "").strip()
        if not x:
            return None
        return float(x)
    except:
        return None

def _fetch_one_day(date_str: str) -> List[Dict[str, Any]]:
    url = f"https://trade.500.com/jczq/?date={date_str}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    html = decode_response(r)
    soup = BeautifulSoup(html, "html.parser")

    matches: List[Dict[str, Any]] = []
    for row in soup.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 6:
            continue

        match_no = tds[0].get_text(strip=True)
        # 过滤：只要“周X001”这类
        if not re.search(r"周[一二三四五六日天]\d{2,3}", match_no):
            continue

        league_a = tds[1].find("a")
        league = league_a.get_text(strip=True) if league_a else tds[1].get_text(strip=True)
        kick_time = tds[2].get_text(strip=True)

        team_as = tds[3].find_all("a")
        home = team_as[0].get_text(strip=True) if len(team_as) >= 1 else ""
        away = team_as[-1].get_text(strip=True) if len(team_as) >= 2 else ""

        handicap = tds[4].get_text(strip=True)

        sp_td = tds[5]
        sp_spans = sp_td.find_all("span")
        sp_win  = sp_spans[0].get_text(strip=True) if len(sp_spans) > 0 else ""
        sp_draw = sp_spans[1].get_text(strip=True) if len(sp_spans) > 1 else ""
        sp_lose = sp_spans[2].get_text(strip=True) if len(sp_spans) > 2 else ""

        raw = sp_td.get_text(" ", strip=True)
        status = raw.replace(sp_win, "").replace(sp_draw, "").replace(sp_lose, "").strip()

        extra = tds[7].get_text(strip=True) if len(tds) > 7 else ""

        matches.append({
            "date": date_str,
            "match_no": match_no,
            "league": league,
            "time": kick_time,
            "home": home,
            "away": away,
            "handicap": handicap,

            # 多给几套字段名，兼容你前端/旧代码
            "odds_win": _to_float(sp_win),
            "odds_draw": _to_float(sp_draw),
            "odds_lose": _to_float(sp_lose),
            "sp_win": _to_float(sp_win),
            "sp_draw": _to_float(sp_draw),
            "sp_lose": _to_float(sp_lose),
            "win": _to_float(sp_win),
            "draw": _to_float(sp_draw),
            "lose": _to_float(sp_lose),

            "status": status,
            "extra": extra,
            "source": "trade.500.com/jczq"
        })
    return matches

def export(days_forward: int = 2) -> Dict[str, Any]:
    # 中国日期
    now_cn = datetime.now(timezone.utc) + timedelta(hours=8)
    base = now_cn.date()

    allm: List[Dict[str, Any]] = []
    for i in range(days_forward + 1):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        allm.extend(_fetch_one_day(d))

    return {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "source": "trade.500.com/jczq",
            "days_forward": days_forward,
            "count": len(allm),
        },
        "matches": allm
    }

def main():
    out_path = Path("site/data/jczq.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    old = None
    if out_path.exists():
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
        except:
            old = None

    try:
        payload = export(days_forward=2)
        # 若抓到 0，且旧数据存在，就保留旧数据（防止站点变空）
        if payload["meta"]["count"] == 0 and old:
            payload = old
            payload.setdefault("meta", {})["note"] = "Fetch returned 0, kept previous data"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("OK jczq.json count=", payload.get("meta", {}).get("count"))
    except Exception as e:
        if old:
            old.setdefault("meta", {})["error"] = str(e)
            out_path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
            print("WARN kept old jczq.json error=", e)
        else:
            payload = {"meta":{"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                               "source":"trade.500.com/jczq","count":0,"error":str(e)},
                       "matches":[]}
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print("ERROR wrote empty jczq.json error=", e)

if __name__ == "__main__":
    main()
