import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from .utils import to_float, HEADERS as BASE_HEADERS, decode_response

# 针对500网的额外请求头
HEADERS = {**BASE_HEADERS, "Referer": "https://trade.500.com/"}

# 为了保持旧接口兼容，保留名称
_to_float = to_float

def fetch_one_day(date_str: str) -> List[Dict]:
    """抓取指定日期的500网竞彩数据，返回字典列表。"""
    url = f"https://trade.500.com/jczq/?date={date_str}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    # the site is served in gbk/gb2312; ``decode_response`` takes care of
    # forcing that encoding and falling back gracefully.
    html = decode_response(r)
    soup = BeautifulSoup(html, "html.parser")

    matches: List[Dict] = []
    for row in soup.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 6:
            continue

        match_num = tds[0].get_text(strip=True)
        if not match_num or len(match_num) > 10:
            continue

        league_a = tds[1].find("a")
        league = league_a.get_text(strip=True) if league_a else tds[1].get_text(strip=True)
        kick_time = tds[2].get_text(strip=True)

        team_as = tds[3].find_all("a")
        home = team_as[0].get_text(strip=True) if team_as else tds[3].get_text(strip=True)
        away = team_as[-1].get_text(strip=True) if len(team_as) > 1 else ""

        handicap = tds[4].get_text(strip=True)
        sp_spans = tds[5].find_all("span")
        sp_win = sp_spans[0].get_text(strip=True) if len(sp_spans) > 0 else ""
        sp_draw = sp_spans[1].get_text(strip=True) if len(sp_spans) > 1 else ""
        sp_lose = sp_spans[2].get_text(strip=True) if len(sp_spans) > 2 else ""

        matches.append({
            "date": date_str,
            "match": match_num,
            "league": league,
            "time": kick_time,
            "home": home,
            "away": away,
            "handicap": handicap,
            "odds_win": _to_float(sp_win),
            "odds_draw": _to_float(sp_draw),
            "odds_lose": _to_float(sp_lose),
            "source": "trade.500.com/jczq",
        })
    return matches

def export(days: int = 1) -> Dict:
    # 调用工具函数获取中国时区的今日日期
    # 按中国日期，利用工具函数获取当前日期字符串并解析
    from .utils import now_cn_date
    today = datetime.strptime(now_cn_date(), "%Y-%m-%d")
    all_matches: List[Dict] = []
    for i in range(days):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            ms = fetch_one_day(d)
            all_matches.extend(ms)
        except Exception as e:
            # 单天失败不影响整体
            continue

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "source": "trade.500.com/jczq",
            "days": days,
            "count": len(all_matches),
        },
        "matches": all_matches
    }
    return payload

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
        payload = export(days=1)
        # 如果抓到 0 场，保留旧数据，避免网站变空
        if payload["meta"]["count"] == 0 and old:
            payload = old
            payload.setdefault("meta", {})["note"] = "Fetch returned 0, kept previous data"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("OK jczq.json count=", payload.get("meta", {}).get("count"))
    except Exception as e:
        if old:
            old.setdefault("meta", {})["error"] = str(e)
            out_path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
            print("WARN kept old jczq.json, error=", e)
        else:
            payload = {"meta":{"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                               "source":"trade.500.com/jczq","count":0,"error":str(e)},
                       "matches":[]}
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print("ERROR wrote empty jczq.json, error=", e)

if __name__ == "__main__":
    main()
