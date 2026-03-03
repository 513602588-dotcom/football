import json, re, time, hashlib, os
from typing import Any, Dict, List, Optional
import requests

JSONP_RE = re.compile(r"^[^(]*\((.*)\)\s*;?\s*$", re.S)

def _parse_json_or_jsonp(text: str) -> Any:
    t = text.strip()
    m = JSONP_RE.match(t)
    if m:
        return json.loads(m.group(1))
    return json.loads(t)

def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    os.makedirs("data/cache", exist_ok=True)
    return f"data/cache/wencai51_{h}.json"

def _load_cache(key: str, ttl: int = 600) -> Optional[Any]:
    p = _cache_path(key)
    if not os.path.exists(p): 
        return None
    if time.time() - os.path.getmtime(p) > ttl:
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def _save_cache(key: str, obj: Any) -> None:
    p = _cache_path(key)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def _walk_candidates(obj: Any) -> List[Dict]:
    """
    递归找“像比赛”的 dict：同时包含 主/客 或 home/away 字段
    """
    out = []
    if isinstance(obj, dict):
        keys = set(obj.keys())
        has_home = any(k in keys for k in ["home","Home","h","h_cn","主队","主","homeTeam","HomeTeam","hn"])
        has_away = any(k in keys for k in ["away","Away","a","a_cn","客队","客","awayTeam","AwayTeam","an"])
        if has_home and has_away:
            out.append(obj)
        for v in obj.values():
            out.extend(_walk_candidates(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_walk_candidates(v))
    return out

def _pick_name(d: Dict, options: List[str]) -> str:
    for k in options:
        if k in d and d[k] not in [None, ""]:
            return str(d[k])
    return ""

def _pick_odds_1x2(d: Dict) -> Dict[str, Optional[float]]:
    """
    尝试提取胜平负赔率/SP
    """
    def f(x):
        try:
            if x is None: return None
            v = float(str(x).strip())
            return v if v > 1 else None
        except:
            return None

    # 常见：sp_3/sp_1/sp_0 或 win/draw/lose
    for ks in [
        ("sp_3","sp_1","sp_0"),
        ("win","draw","lose"),
        ("h","d","a"),
        ("odds_home","odds_draw","odds_away"),
        ("homeWin","draw","awayWin"),
    ]:
        ow, od, oa = f(d.get(ks[0])), f(d.get(ks[1])), f(d.get(ks[2]))
        if ow and od and oa:
            return {"win": ow, "draw": od, "lose": oa}

    # 有些放在数组/对象里
    for k in ["odds","sp","had","1x2"]:
        v = d.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            ow, od, oa = f(v[0]), f(v[1]), f(v[2])
            if ow and od and oa:
                return {"win": ow, "draw": od, "lose": oa}
        if isinstance(v, dict):
            ow = f(v.get("win") or v.get("3"))
            od = f(v.get("draw") or v.get("1"))
            oa = f(v.get("lose") or v.get("0"))
            if ow and od and oa:
                return {"win": ow, "draw": od, "lose": oa}

    return {"win": None, "draw": None, "lose": None}

def fetch_wencai51(api_url: str, mt: str, ttl: int = 600) -> Dict[str, Any]:
    """
    api_url：你用抓包探测出来的“比赛列表/赔率接口”
    mt：页面链接里的 mt
    """
    key = f"{api_url}|{mt}"
    cached = _load_cache(key, ttl)
    if cached is not None:
        return {"raw": cached, "matches": _extract_matches(cached)}

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://m.wencai51.cn/",
    }

    # 兼容：如果 api_url 里写了 {mt}，就替换；否则把 mt 当 query 追加
    url = api_url.replace("{mt}", mt)
    if "{mt}" not in api_url and "mt=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}mt={mt}"

    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()

    txt = r.text
    try:
        data = _parse_json_or_jsonp(txt)
    except Exception:
        # 有些接口返回 text/plain 包 JSON
        data = json.loads(txt)

    _save_cache(key, data)
    return {"raw": data, "matches": _extract_matches(data)}

def _extract_matches(data: Any) -> List[Dict[str, Any]]:
    cands = _walk_candidates(data)
    out = []
    for d in cands:
        home = _pick_name(d, ["home","HomeTeam","homeTeam","h_cn","主队","主","h","hn"])
        away = _pick_name(d, ["away","AwayTeam","awayTeam","a_cn","客队","客","a","an"])
        if not home or not away:
            continue

        league = _pick_name(d, ["league","League","l_cn","联赛","赛事","competition"])
        time_ = _pick_name(d, ["time","Time","start_time","bt","开赛","比赛时间","dateTime","kickoff"])
        odds = _pick_odds_1x2(d)
        handicap = _pick_name(d, ["handicap","rq","goalline","让球"])

        out.append({
            "league": league,
            "time": time_,
            "home": home,
            "away": away,
            "odds_win": odds["win"],
            "odds_draw": odds["draw"],
            "odds_lose": odds["lose"],
            "handicap": handicap,
            "raw": d,
        })

    # 去重
    uniq = {}
    for m in out:
        key = (m["home"], m["away"], m["time"])
        uniq[key] = m
    return list(uniq.values())
