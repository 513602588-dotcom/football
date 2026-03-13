#!/usr/bin/env python3
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
from src.engine.value import calc, implied_prob, label, remove_overround, score
from src.models.bookmaker import predict_from_odds
from src.models.upset import avoid_upset

FOOTBALL_DATA_LEAGUES = {
    "PL": "英超", "PD": "西甲", "BL1": "德甲", "SA": "意甲", "FL1": "法甲",
    "CL": "欧冠", "EL": "欧联", "WC": "世界杯", "EC": "欧洲杯", "FRI": "国际友谊赛",
}
LEAGUE_ODDS_MAPPING = {
    "英超": "soccer_epl", "西甲": "soccer_spain_la_liga", "德甲": "soccer_germany_bundesliga",
    "意甲": "soccer_italy_serie_a", "法甲": "soccer_france_ligue_one", "欧冠": "soccer_uefa_champs_league",
    "欧联": "soccer_uefa_europa_league", "世界杯": "soccer_fifa_world_cup", "欧洲杯": "soccer_uefa_european_championship",
    "国际友谊赛": "soccer_international_friendlies",
}
TEAM_NAME_CN_MAP = {
    "West Ham United FC": "西汉姆联", "Manchester City FC": "曼城", "Burnley FC": "伯恩利",
    "AFC Bournemouth": "伯恩茅斯", "Arsenal FC": "阿森纳", "Chelsea FC": "切尔西",
    "Liverpool FC": "利物浦", "Tottenham Hotspur FC": "托特纳姆热刺", "Manchester United FC": "曼联",
    "Newcastle United FC": "纽卡斯尔联", "Brighton & Hove Albion FC": "布莱顿", "Aston Villa FC": "阿斯顿维拉",
    "Fulham FC": "富勒姆", "Crystal Palace FC": "水晶宫", "Brentford FC": "布伦特福德",
    "Wolverhampton Wanderers FC": "狼队", "Nottingham Forest FC": "诺丁汉森林", "Leeds United FC": "利兹联",
    "Sunderland AFC": "桑德兰", "Everton FC": "埃弗顿", "Leicester City FC": "莱斯特城",
    "Southampton FC": "南安普顿", "Ipswich Town FC": "伊普斯维奇", "Oxford United FC": "牛津联",
    "Deportivo Alavés": "阿拉维斯", "Villarreal CF": "比利亚雷亚尔", "Girona FC": "赫罗纳",
    "Athletic Club": "毕尔巴鄂竞技", "Club Atlético de Madrid": "马德里竞技", "Getafe CF": "赫塔费",
    "Real Oviedo": "奥维耶多", "Valencia CF": "瓦伦西亚", "Real Madrid CF": "皇家马德里",
    "Elche CF": "埃尔切", "RCD Mallorca": "马略卡", "RCD Espanyol de Barcelona": "西班牙人",
    "FC Barcelona": "巴塞罗那", "Sevilla FC": "塞维利亚", "Real Betis Balompié": "皇家贝蒂斯",
    "RC Celta de Vigo": "塞尔塔", "Real Sociedad de Fútbol": "皇家社会", "CA Osasuna": "奥萨苏纳",
    "Rayo Vallecano de Madrid": "巴列卡诺", "Levante UD": "莱万特", "Real Valladolid CF": "巴拉多利德",
    "CD Leganés": "莱加内斯", "Borussia Mönchengladbach": "门兴格拉德巴赫", "FC St. Pauli 1910": "圣保利",
    "Bayer 04 Leverkusen": "勒沃库森", "FC Bayern München": "拜仁慕尼黑", "Borussia Dortmund": "多特蒙德",
    "FC Augsburg": "奥格斯堡", "Eintracht Frankfurt": "法兰克福", "1. FC Heidenheim 1846": "海登海姆",
    "TSG 1899 Hoffenheim": "霍芬海姆", "VfL Wolfsburg": "沃尔夫斯堡", "Hamburger SV": "汉堡",
    "1. FC Köln": "科隆", "SV Werder Bremen": "云达不莱梅", "1. FSV Mainz 05": "美因茨",
    "SC Freiburg": "弗赖堡", "1. FC Union Berlin": "柏林联合", "VfB Stuttgart": "斯图加特",
    "RB Leipzig": "RB莱比锡", "Holstein Kiel": "荷尔斯泰因基尔", "FC Schalke 04": "沙尔克04",
    "Torino FC": "都灵", "Parma Calcio 1913": "帕尔马", "FC Internazionale Milano": "国际米兰",
    "Atalanta BC": "亚特兰大", "SSC Napoli": "那不勒斯", "US Lecce": "莱切",
    "Udinese Calcio": "乌迪内斯", "Juventus FC": "尤文图斯", "Hellas Verona FC": "维罗纳",
    "Genoa CFC": "热那亚", "AC Pisa 1909": "比萨", "Cagliari Calcio": "卡利亚里",
    "US Sassuolo Calcio": "萨索洛", "Bologna FC 1909": "博洛尼亚", "Como 1907": "科莫",
    "AS Roma": "罗马", "SS Lazio": "拉齐奥", "AC Milan": "AC米兰",
    "US Cremonese": "克雷莫内塞", "ACF Fiorentina": "佛罗伦萨", "Venezia FC": "威尼斯",
    "Modena FC": "摩德纳", "Olympique de Marseille": "马赛", "AJ Auxerre": "欧塞尔",
    "FC Lorient": "洛里昂", "Racing Club de Lens": "朗斯", "Angers SCO": "昂热",
    "OGC Nice": "尼斯", "AS Monaco FC": "摩纳哥", "Stade Brestois 29": "布雷斯特",
    "RC Strasbourg Alsace": "斯特拉斯堡", "Paris FC": "巴黎FC", "FC Metz": "梅斯",
    "Toulouse FC": "图卢兹", "Le Havre AC": "勒阿弗尔", "Olympique Lyonnais": "里昂",
    "Stade Rennais FC 1901": "雷恩", "Lille OSC": "里尔", "Paris Saint-Germain FC": "巴黎圣日耳曼",
    "Stade de Reims": "兰斯", "Girondins de Bordeaux": "波尔多", "Sporting Clube de Portugal": "葡萄牙体育",
    "FK Bodø/Glimt": "博德闪耀", "Galatasaray SK": "加拉塔萨雷", "Benfica SL": "本菲卡",
    "FC Porto": "波尔图", "PSV Eindhoven": "埃因霍温", "Feyenoord Rotterdam": "费耶诺德",
    "Celtic FC": "凯尔特人", "Rangers FC": "流浪者",
}
WEIGHT_ODDS = 0.45
WEIGHT_FORM = 0.30
WEIGHT_H2H = 0.15
WEIGHT_XG = 0.10
LEAGUE_BASELINE = {
    "premier league": {"home_bias": 0.06, "draw_rate": 0.26, "avg_goals": 2.8},
    "la liga": {"home_bias": 0.05, "draw_rate": 0.27, "avg_goals": 2.7},
    "bundesliga": {"home_bias": 0.06, "draw_rate": 0.24, "avg_goals": 3.2},
    "serie a": {"home_bias": 0.04, "draw_rate": 0.29, "avg_goals": 2.6},
    "ligue 1": {"home_bias": 0.03, "draw_rate": 0.32, "avg_goals": 2.4},
    "champions league": {"home_bias": 0.04, "draw_rate": 0.25, "avg_goals": 2.9},
    "europa league": {"home_bias": 0.04, "draw_rate": 0.26, "avg_goals": 2.7},
    "default": {"home_bias": 0.05, "draw_rate": 0.28, "avg_goals": 2.7},
}
OUT_DIR = Path("site/data")
PICKS_PATH = OUT_DIR / "picks.json"
TOP_PATH = OUT_DIR / "top_picks.json"
PREDICTIONS_PATH = OUT_DIR / "predictions.json"
FUTURE_DAYS = 3
PAST_DAYS = 0
TOP_N = 4
ODDS_CACHE_FILE = OUT_DIR / "odds_cache.json"
CACHE_EXPIRE_HOURS = 1

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str

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
        raw_items = ["doubao-1.5-pro-32k"]
    aliases = {
        "豆包pro": "doubao-1.5-pro-32k", "豆包flash": "doubao-1.5-flash-8b",
        "doubao-pro": "doubao-1.5-pro-32k", "doubao-flash": "doubao-1.5-flash-8b",
    }
    out: List[str] = []
    seen: Set[str] = set()
    def push(name: str) -> None:
        n = name.strip()
        if not n or n.lower() in seen:
            return
        seen.add(n.lower())
        out.append(n)
    for item in raw_items:
        normalized = aliases.get(item.lower(), item)
        push(normalized)
        low = normalized.lower()
        if "pro" in low:
            push("doubao-1.5-pro-32k")
            push("doubao-1.5-flash-8b")
        if "flash" in low:
            push("doubao-1.5-flash-8b")
            push("doubao-1.5-pro-32k")
    return out

def _team_name_quality(name: str) -> bool:
    t = str(name or "").strip()
    if len(t) < 2:
        return False
    alnum = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", t)
    return bool(alnum)

def get_league_baseline(league_name: str) -> Dict[str, float]:
    if not league_name:
        return LEAGUE_BASELINE["default"]
    league_lower = league_name.strip().lower()
    for key, baseline in LEAGUE_BASELINE.items():
        if key in league_lower:
            return baseline
    return LEAGUE_BASELINE["default"]

def _norm_team(name: str) -> str:
    n = (name or "").strip().lower()
    n = re.sub(r"\(.*?\)", "", n)
    remove_words = [
        "fc", "cf", "ac", "sc", "as", "rc", "ssc", "us", "afc", "if", "fk", "sk", "bv", "bsc",
        "aj", "1907", "1908", "1909", "1910", "1913", "1919", "1920", "1925", "1930", "1945",
        "de", "calcio", "1846", "1860", "1899", "1900", "1901", "1902", "1903", "1904", "1905", "1906"
    ]
    for word in remove_words:
        n = re.sub(rf"\b{word}\b", "", n)
    n = n.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    n = n.replace("'", "").replace(".", "").replace("-", "")
    n = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", n)
    return n.strip()

def team_name_to_cn(team_name: str) -> str:
    if not team_name:
        return ""
    if team_name in TEAM_NAME_CN_MAP:
        return TEAM_NAME_CN_MAP[team_name]
    clean_name = re.sub(r" FC| CF| AC| SC| 19\d{2}$", "", team_name.strip())
    for en_name, cn_name in TEAM_NAME_CN_MAP.items():
        clean_en = re.sub(r" FC| CF| AC| SC| 19\d{2}$", "", en_name.strip())
        if clean_name == clean_en:
            return cn_name
    return team_name

def fetch_team_recent_form(team_name: str, api_key: str) -> Tuple[float, float, float]:
    if not valid_key(api_key):
        return 0.5, 1.4, 1.2
    try:
        resp = requests.get(
            "https://v3.football.api-sports.io/teams/search",
            headers={"x-apisports-key": api_key},
            params={"name": team_name},
            timeout=15,
        )
        resp.raise_for_status()
        teams = resp.json().get("response", [])
        if not teams:
            return 0.5, 1.4, 1.2
        team_id = teams[0]["team"]["id"]
        resp = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": api_key},
            params={"team": team_id, "last": 10},
            timeout=15,
        )
        resp.raise_for_status()
        fixtures = resp.json().get("response", [])
        if not fixtures:
            return 0.5, 1.4, 1.2
        wins = 0
        total_goals = 0
        total_conceded = 0
        for fix in fixtures:
            is_home = fix["teams"]["home"]["id"] == team_id
            home_goals = fix["goals"]["home"] or 0
            away_goals = fix["goals"]["away"] or 0
            if is_home:
                team_goals = home_goals
                team_conceded = away_goals
                if home_goals > away_goals:
                    wins +=1
            else:
                team_goals = away_goals
                team_conceded = home_goals
                if away_goals > home_goals:
                    wins +=1
            total_goals += team_goals
            total_conceded += team_conceded
        match_count = len(fixtures)
        win_rate = wins / match_count
        avg_goals = total_goals / match_count
        avg_conceded = total_conceded / match_count
        return round(win_rate, 2), round(avg_goals, 2), round(avg_conceded, 2)
    except Exception:
        return 0.5, 1.4, 1.2

def fetch_h2h_data(home_team: str, away_team: str, api_key: str) -> Tuple[float, float, float]:
    if not valid_key(api_key):
        return 0.45, 0.28, 0.27
    try:
        home_resp = requests.get(
            "https://v3.football.api-sports.io/teams/search",
            headers={"x-apisports-key": api_key},
            params={"name": home_team},
            timeout=15,
        )
        away_resp = requests.get(
            "https://v3.football.api-sports.io/teams/search",
            headers={"x-apisports-key": api_key},
            params={"name": away_team},
            timeout=15,
        )
        home_resp.raise_for_status()
        away_resp.raise_for_status()
        home_teams = home_resp.json().get("response", [])
        away_teams = away_resp.json().get("response", [])
        if not home_teams or not away_teams:
            return 0.45, 0.28, 0.27
        home_id = home_teams[0]["team"]["id"]
        away_id = away_teams[0]["team"]["id"]
        resp = requests.get(
            "https://v3.football.api-sports.io/fixtures/headtohead",
            headers={"x-apisports-key": api_key},
            params={"h2h": f"{home_id}-{away_id}", "last": 10},
            timeout=15,
        )
        resp.raise_for_status()
        fixtures = resp.json().get("response", [])
        if not fixtures:
            return 0.45, 0.28, 0.27
        home_wins = 0
        draws = 0
        away_wins = 0
        for fix in fixtures:
            home_goals = fix["goals"]["home"] or 0
            away_goals = fix["goals"]["away"] or 0
            if home_goals > away_goals:
                home_wins +=1
            elif home_goals == away_goals:
                draws +=1
            else:
                away_wins +=1
        match_count = len(fixtures)
        return round(home_wins/match_count,2), round(draws/match_count,2), round(away_wins/match_count,2)
    except Exception:
        return 0.45, 0.28, 0.27

def fetch_injury_correction(home_team: str, away_team: str, api_key: str) -> Tuple[float, float]:
    if not valid_key(api_key):
        return 0.0, 0.0
    try:
        home_resp = requests.get(
            "https://v3.football.api-sports.io/teams/search",
            headers={"x-apisports-key": api_key},
            params={"name": home_team},
            timeout=15,
        )
        away_resp = requests.get(
            "https://v3.football.api-sports.io/teams/search",
            headers={"x-apisports-key": api_key},
            params={"name": away_team},
            timeout=15,
        )
        home_resp.raise_for_status()
        away_resp.raise_for_status()
        home_teams = home_resp.json().get("response", [])
        away_teams = away_resp.json().get("response", [])
        if not home_teams or not away_teams:
            return 0.0, 0.0
        home_id = home_teams[0]["team"]["id"]
        away_id = away_teams[0]["team"]["id"]
        home_inj_resp = requests.get(
            "https://v3.football.api-sports.io/injuries",
            headers={"x-apisports-key": api_key},
            params={"team": home_id},
            timeout=15,
        )
        away_inj_resp = requests.get(
            "https://v3.football.api-sports.io/injuries",
            headers={"x-apisports-key": api_key},
            params={"team": away_id},
            timeout=15,
        )
        home_inj_resp.raise_for_status()
        away_inj_resp.raise_for_status()
        home_injuries = home_inj_resp.json().get("response", [])
        away_injuries = away_inj_resp.json().get("response", [])
        home_correction = 0.0
        away_correction = 0.0
        for inj in home_injuries:
            if inj.get("type") == "Missing" or inj.get("player", {}).get("pos") in ["Goalkeeper", "Defender", "Midfielder", "Attacker"]:
                home_correction -= 0.02
        for inj in away_injuries:
            if inj.get("type") == "Missing" or inj.get("player", {}).get("pos") in ["Goalkeeper", "Defender", "Midfielder", "Attacker"]:
                away_correction -= 0.02
        return round(home_correction, 4), round(away_correction, 4)
    except Exception:
        return 0.0, 0.0

def get_fixture_corrections(home_team: str, away_team: str, league_name: str, api_key: str) -> Tuple[float, float, float]:
    if not valid_key(api_key):
        return 0.0, 0.0, 0.0
    try:
        home_resp = requests.get(
            "https://v3.football.api-sports.io/teams/search",
            headers={"x-apisports-key": api_key},
            params={"name": home_team},
            timeout=15,
        )
        away_resp = requests.get(
            "https://v3.football.api-sports.io/teams/search",
            headers={"x-apisports-key": api_key},
            params={"name": away_team},
            timeout=15,
        )
        home_resp.raise_for_status()
        away_resp.raise_for_status()
        home_teams = home_resp.json().get("response", [])
        away_teams = away_resp.json().get("response", [])
        if not home_teams or not away_teams:
            return 0.0, 0.0, 0.0
        home_id = home_teams[0]["team"]["id"]
        away_id = away_teams[0]["team"]["id"]
        league_id = None
        league_lower = league_name.lower()
        for key in LEAGUE_BASELINE.keys():
            if key in league_lower:
                league_resp = requests.get(
                    "https://v3.football.api-sports.io/leagues",
                    headers={"x-apisports-key": api_key},
                    params={"search": key, "current": "true"},
                    timeout=15,
                )
                league_resp.raise_for_status()
                leagues = league_resp.json().get("response", [])
                if leagues:
                    league_id = leagues[0]["league"]["id"]
                    break
        morale_correction = 0.0
        if league_id:
            standings_resp = requests.get(
                "https://v3.football.api-sports.io/standings",
                headers={"x-apisports-key": api_key},
                params={"league": league_id, "season": 2025},
                timeout=15,
            )
            standings_resp.raise_for_status()
            standings = standings_resp.json().get("response", [])
            if standings:
                home_rank = 0
                away_rank = 0
                total_teams = 20
                for stand in standings:
                    for row in stand.get("league", {}).get("standings", [])[0]:
                        if row.get("team", {}).get("id") == home_id:
                            home_rank = row.get("rank", 10)
                        if row.get("team", {}).get("id") == away_id:
                            away_rank = row.get("rank", 10)
                        total_teams = len(stand.get("league", {}).get("standings", [])[0])
                if home_rank <= 4 and away_rank > 10:
                    morale_correction += 0.03
                if away_rank <= 4 and home_rank > 10:
                    morale_correction -= 0.03
                if home_rank >= total_teams-3 and away_rank < total_teams-3:
                    morale_correction += 0.04
                if away_rank >= total_teams-3 and home_rank < total_teams-3:
                    morale_correction -= 0.04
        fitness_correction = 0.0
        home_fixtures_resp = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": api_key},
            params={"team": home_id, "last": 1},
            timeout=15,
        )
        away_fixtures_resp = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": api_key},
            params={"team": away_id, "last": 1},
            timeout=15,
        )
        home_fixtures_resp.raise_for_status()
        away_fixtures_resp.raise_for_status()
        home_fixtures = home_fixtures_resp.json().get("response", [])
        away_fixtures = away_fixtures_resp.json().get("response", [])
        if home_fixtures and away_fixtures:
            home_last_date = datetime.fromisoformat(home_fixtures[0]["fixture"]["date"].replace("Z", "+00:00"))
            away_last_date = datetime.fromisoformat(away_fixtures[0]["fixture"]["date"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            home_rest_days = (now - home_last_date).days
            away_rest_days = (now - away_last_date).days
            rest_gap = home_rest_days - away_rest_days
            if rest_gap >= 3:
                fitness_correction += 0.02
            elif rest_gap <= -3:
                fitness_correction -= 0.02
        upset_correction = 0.0
        home_fixtures_resp = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": api_key},
            params={"team": home_id, "last": 10},
            timeout=15,
        )
        away_fixtures_resp = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": api_key},
            params={"team": away_id, "last": 10},
            timeout=15,
        )
        home_fixtures_resp.raise_for_status()
        away_fixtures_resp.raise_for_status()
        home_fixtures = home_fixtures_resp.json().get("response", [])
        away_fixtures = away_fixtures_resp.json().get("response", [])
        home_upset_count = 0
        away_upset_count = 0
        for fix in home_fixtures:
            is_home = fix["teams"]["home"]["id"] == home_id
            is_winner = fix["teams"]["home"]["winner"] if is_home else fix["teams"]["away"]["winner"]
            is_favorite = fix["teams"]["home"]["winner"] is not None and not is_winner
            if is_favorite and not is_winner:
                home_upset_count +=1
        for fix in away_fixtures:
            is_home = fix["teams"]["home"]["id"] == away_id
            is_winner = fix["teams"]["home"]["winner"] if is_home else fix["teams"]["away"]["winner"]
            is_favorite = fix["teams"]["home"]["winner"] is not None and not is_winner
            if is_favorite and not is_winner:
                away_upset_count +=1
        if home_upset_count >= 3:
            upset_correction -= 0.02
        if away_upset_count >= 3:
            upset_correction += 0.02
        return round(morale_correction,4), round(fitness_correction,4), round(upset_correction,4)
    except Exception:
        return 0.0, 0.0, 0.0

def load_odds_cache() -> Optional[Dict]:
    if not ODDS_CACHE_FILE.exists():
        return None
    try:
        cache_data = json.loads(ODDS_CACHE_FILE.read_text(encoding="utf-8"))
        cache_time = datetime.fromisoformat(cache_data["cache_time"])
        if datetime.now(timezone.utc) - cache_time < timedelta(hours=CACHE_EXPIRE_HOURS):
            return cache_data["odds_data"]
        return None
    except Exception:
        return None

def save_odds_cache(odds_data: Dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "cache_time": datetime.now(timezone.utc).isoformat(),
        "odds_data": odds_data
    }
    ODDS_CACHE_FILE.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")

def fetch_all_league_odds(api_key: str) -> Dict:
    cached_odds = load_odds_cache()
    if cached_odds is not None:
        return cached_odds
    if not valid_key(api_key):
        return {}
    all_odds = {}
    total_leagues = len(LEAGUE_ODDS_MAPPING)
    for league_name, sport_code in LEAGUE_ODDS_MAPPING.items():
        try:
            resp = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_code}/odds",
                params={
                    "apiKey": api_key,
                    "regions": "uk,eu,us",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=20,
            )
            if resp.status_code in [401,403,404]:
                continue
            resp.raise_for_status()
            events = resp.json() or []
            for event in events:
                event_home = _norm_team(event.get("home_team", ""))
                event_away = _norm_team(event.get("away_team", ""))
                match_key = f"{event_home}_{event_away}"
                reverse_key = f"{event_away}_{event_home}"
                all_odds[match_key] = event
                all_odds[reverse_key] = event
        except Exception:
            continue
    save_odds_cache(all_odds)
    return all_odds

def get_match_odds_from_cache(home_team: str, away_team: str, all_odds_cache: Dict) -> Tuple[Optional[Tuple[float, float, float]], float, float]:
    norm_home = _norm_team(home_team)
    norm_away = _norm_team(away_team)
    match_key = f"{norm_home}_{norm_away}"
    matched_event = all_odds_cache.get(match_key)
    if not matched_event:
        return None, 0.0, 0.0
    bookmakers = matched_event.get("bookmakers", [])
    bookmaker_count = len(bookmakers)
    if not bookmakers:
        return None, 0.0, 0.0
    opening_odds = None
    first_bk = bookmakers[0]
    for mk in first_bk.get("markets", []):
        if mk.get("key") == "h2h":
            outcomes = mk.get("outcomes", [])
            oh = od = oa = None
            event_home_name = matched_event.get("home_team", "")
            event_away_name = matched_event.get("away_team", "")
            for o in outcomes:
                o_name = o.get("name", "")
                if _norm_team(o_name) == _norm_team(event_home_name):
                    oh = float(o.get("price", 0))
                elif _norm_team(o_name) == _norm_team(event_away_name):
                    oa = float(o.get("price", 0))
                else:
                    od = float(o.get("price", 0))
            if oh and od and oa:
                opening_odds = (oh, od, oa)
                break
    total_oh = total_od = total_oa = 0.0
    valid_count = 0
    consensus_count = 0
    for bk in bookmakers:
        for mk in bk.get("markets", []):
            if mk.get("key") == "h2h":
                outcomes = mk.get("outcomes", [])
                oh = od = oa = None
                event_home_name = matched_event.get("home_team", "")
                event_away_name = matched_event.get("away_team", "")
                for o in outcomes:
                    o_name = o.get("name", "")
                    if _norm_team(o_name) == _norm_team(event_home_name):
                        oh = float(o.get("price", 0))
                    elif _norm_team(o_name) == _norm_team(event_away_name):
                        oa = float(o.get("price", 0))
                    else:
                        od = float(o.get("price", 0))
                if oh and od and oa:
                    total_oh += oh
                    total_od += od
                    total_oa += oa
                    valid_count +=1
                    if opening_odds and oh < opening_odds[0]:
                        consensus_count +=1
                    break
    current_odds = None
    if valid_count > 0:
        current_odds = (
            round(total_oh/valid_count, 2),
            round(total_od/valid_count, 2),
            round(total_oa/valid_count, 2),
        )
    move_correction = 0.0
    consensus_correction = 0.0
    if opening_odds and current_odds and bookmaker_count > 0:
        oh_open, _, _ = opening_odds
        oh_current, _, _ = current_odds
        if oh_current < oh_open:
            move_correction = min((oh_open - oh_current)/oh_open, 0.05)
        elif oh_current > oh_open:
            move_correction = -min((oh_current - oh_open)/oh_open, 0.05)
        consensus_rate = consensus_count / bookmaker_count
        if consensus_rate >= 0.7:
            consensus_correction = 0.03
        elif consensus_rate <= 0.2:
            consensus_correction = -0.03
    return current_odds, move_correction, consensus_correction

def fuse_multi_factor_probs(
    odds_prob: Tuple[float, float, float],
    form_prob: Tuple[float, float, float],
    h2h_prob: Tuple[float, float, float],
    xg_prob: Tuple[float, float, float],
    league_baseline: Dict[str, float],
    total_correction: float,
) -> Tuple[float, float, float, Dict[str, float]]:
    weights = {
        "odds": WEIGHT_ODDS if odds_prob else 0.0,
        "form": WEIGHT_FORM if form_prob else 0.0,
        "h2h": WEIGHT_H2H if h2h_prob else 0.0,
        "xg": WEIGHT_XG if xg_prob else 0.0,
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.45, 0.28, 0.27, weights
    ph = (
        weights["odds"] * (odds_prob[0] if odds_prob else 0)
        + weights["form"] * (form_prob[0] if form_prob else 0)
        + weights["h2h"] * (h2h_prob[0] if h2h_prob else 0)
        + weights["xg"] * (xg_prob[0] if xg_prob else 0)
    ) / total_weight
    pd_ = (
        weights["odds"] * (odds_prob[1] if odds_prob else 0)
        + weights["form"] * (form_prob[1] if form_prob else 0)
        + weights["h2h"] * (h2h_prob[1] if h2h_prob else 0)
        + weights["xg"] * (xg_prob[1] if xg_prob else 0)
    ) / total_weight
    pa = (
        weights["odds"] * (odds_prob[2] if odds_prob else 0)
        + weights["form"] * (form_prob[2] if form_prob else 0)
        + weights["h2h"] * (h2h_prob[2] if h2h_prob else 0)
        + weights["xg"] * (xg_prob[2] if xg_prob else 0)
    ) / total_weight
    ph = ph * (1 + league_baseline["home_bias"] + total_correction)
    pa = pa * (1 - league_baseline["home_bias"] - total_correction)
    pd_ = pd_ * (1 + (league_baseline["draw_rate"] - 0.28))
    ph, pd_, pa = avoid_upset(ph, pd_, pa)
    total = ph + pd_ + pa
    if total <= 0:
        return 0.45, 0.28, 0.27, weights
    return round(ph/total, 4), round(pd_/total, 4), round(pa/total, 4), weights

def probe_external_connections() -> Dict[str, object]:
    out: Dict[str, object] = {}
    api_key = env_value("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY")
    if valid_key(api_key):
        try:
            r = requests.get(
                "https://v3.football.api-sports.io/status",
                headers={"x-apisports-key": api_key},
                timeout=12,
            )
            out["api_football"] = {"ok": r.ok, "status": r.status_code}
        except Exception as exc:
            out["api_football"] = {"ok": False, "error": str(exc)}
    else:
        out["api_football"] = {"ok": False, "error": "missing_key"}
    fdb_key = env_value("FOOTBALL_DATA_KEY", "FOOTBALL_DATA_API_KEY")
    if valid_key(fdb_key):
        try:
            r = requests.get(
                "https://api.football-data.org/v4/matches",
                headers={"X-Auth-Token": fdb_key},
                params={"dateFrom": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "dateTo": (datetime.now(timezone.utc) + timedelta(days=FUTURE_DAYS)).strftime("%Y-%m-%d")},
                timeout=12,
            )
            out["football_data"] = {"ok": r.ok, "status": r.status_code}
        except Exception as exc:
            out["football_data"] = {"ok": False, "error": str(exc)}
    else:
        out["football_data"] = {"ok": False, "error": "missing_key"}
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
        out["odds_api"] = {"ok": False, "error": "missing_key"}
    def _probe_llm(base: str, key: str, model: str) -> Dict[str, object]:
        if not valid_key(key):
            return {"ok": False, "error": "missing_key"}
        try:
            r = requests.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5, "temperature": 0},
                timeout=14,
            )
            return {"ok": r.ok, "status": r.status_code}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    doubao_models = parse_model_candidates(env_value("OPENAI_MODEL", default="doubao-1.5-pro-32k"))
    doubao_probe = {"ok": False, "error": "all_model_failed"}
    for cand in doubao_models:
        probe_one = _probe_llm(
            env_value("OPENAI_BASE_URL", default="https://ark.cn-beijing.volces.com/api/v3"),
            env_value("OPENAI_API_KEY", "OPENAI_KEY"),
            cand,
        )
        probe_one["model"] = cand
        if probe_one.get("error") == "missing_key":
            doubao_probe = probe_one
            break
        if probe_one.get("ok"):
            doubao_probe = probe_one
            break
        doubao_probe = probe_one
    out["doubao_api"] = doubao_probe
    return out

def llm_chat_completion(cfg: LLMConfig, prompt: str) -> Optional[str]:
    if not cfg.base_url or not cfg.api_key:
        return None
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    def clean_text(txt: str) -> Optional[str]:
        t = re.sub(r"\s+", " ", str(txt or "")).strip()
        if not t or len(t) < 10 or len(re.findall(r"[\u4e00-\u9fff]", t)) < 4:
            return None
        return t[:180]
    for m in parse_model_candidates(cfg.model):
        payload = {
            "model": m,
            "messages": [
                {"role": "system", "content": "你是专业简洁的足球赛事分析师，只用中文回复，严格输出2行：第1行战术分析，第2行赔率价值分析。"},
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
                    choices = data.get("choices", [])
                    if not choices:
                        break
                    txt = (choices[0].get("message") or {}).get("content", "")
                    cleaned = clean_text(txt)
                    if cleaned:
                        return cleaned
                    break
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

def build_llm_reason(cfg: LLMConfig, pick: Dict) -> Tuple[str, str, Optional[str]]:
    prompt = (
        f"比赛: {pick['home']} vs {pick['away']}\n"
        f"联赛: {pick['league']}\n"
        f"胜平负概率: 主胜{pick['p_home']:.2f} 平局{pick['p_draw']:.2f} 客胜{pick['p_away']:.2f}\n"
        f"预期进球xG: {pick['xg_home']:.2f}-{pick['xg_away']:.2f}\n"
        f"投注推荐: {pick['pick']} 预期EV值={pick.get('ev', 0)}\n"
        "严格输出2行：第1行战术分析，第2行赔率价值分析。"
    )
    doubao_text = llm_chat_completion(cfg, prompt)
    if doubao_text:
        return f"豆包AI分析: {doubao_text}", "doubao", doubao_text
    fallback = "赛事分析: 球队进攻效率存在差异，预测概率与赔率形成正EV区间。"
    return fallback, "fallback", None

def fetch_football_data_fixtures(start: datetime, end: datetime) -> List[Dict]:
    api_key = env_value("FOOTBALL_DATA_KEY", "FOOTBALL_DATA_API_KEY")
    if not valid_key(api_key):
        return []
    base_url = "https://api.football-data.org/v4"
    headers = {"X-Auth-Token": api_key}
    out: List[Dict] = []
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")
    now_utc = datetime.now(timezone.utc)
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/matches",
            headers=headers,
            params={
                "competitions": ",".join(FOOTBALL_DATA_LEAGUES.keys()),
                "dateFrom": start_date,
                "dateTo": end_date,
                "status": "SCHEDULED,TIMED",
            },
            timeout=20,
        )
        resp_json = resp.json()
        resp.raise_for_status()
        matches = resp_json.get("matches", [])
        valid_count = 0
        for m in matches:
            try:
                fixture_date = datetime.fromisoformat(m.get("utcDate", "").replace("Z", "+00:00"))
            except:
                continue
            if fixture_date <= now_utc:
                continue
            competition_code = (m.get("competition") or {}).get("code")
            league_name = FOOTBALL_DATA_LEAGUES.get(competition_code)
            if not league_name:
                continue
            home = ((m.get("homeTeam") or {}).get("name")) or ""
            away = ((m.get("awayTeam") or {}).get("name")) or ""
            if not home or not away or not _team_name_quality(home) or not _team_name_quality(away):
                continue
            match_date = fixture_date.strftime("%Y-%m-%d")
            match_time = fixture_date.strftime("%H:%M")
            out.append({
                "date": match_date,
                "time": match_time,
                "league": league_name,
                "home": home,
                "away": away,
                "odds_win": None,
                "odds_draw": None,
                "odds_lose": None,
                "source": "football-data",
                "fixture_date_utc": fixture_date,
            })
            valid_count +=1
    except Exception:
        pass
    return out

def load_fixtures() -> pd.DataFrame:
    utc_now = datetime.now(timezone.utc)
    today = datetime.strptime(utc_now.strftime("%Y-%m-%d"), "%Y-%m-%d")
    start = today - timedelta(days=PAST_DAYS)
    upper = today + timedelta(days=max(FUTURE_DAYS, 7))
    rows = fetch_football_data_fixtures(start, upper)
    if not rows:
        return pd.DataFrame()
    fx = pd.DataFrame(rows)
    fx["date"] = fx.get("date", "").astype(str)
    kick = fx.get("time", "").astype(str).str.extract(r"(\d{1,2}:\d{2})")[0].fillna("00:00")
    fx["Date"] = pd.to_datetime(fx["date"] + " " + kick, errors="coerce", utc=True)
    fx = fx.rename(columns={"home": "HomeTeam", "away": "AwayTeam", "league": "League"})
    fx["source"] = fx.get("source", "")
    fx["odds_win"] = pd.to_numeric(fx.get("odds_win"), errors="coerce")
    fx["odds_draw"] = pd.to_numeric(fx.get("odds_draw"), errors="coerce")
    fx["odds_lose"] = pd.to_numeric(fx.get("odds_lose"), errors="coerce")
    fx = fx.sort_values(["Date", "League", "HomeTeam"], ascending=[True, True, True])
    return fx.reset_index(drop=True)

def build_prediction_rows(fx: pd.DataFrame, all_odds_cache: Dict) -> Tuple[List[Dict], Dict]:
    api_football_key = env_value("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY")
    rows: List[Dict] = []
    total_fixtures = len(fx)
    for idx, (_, r) in enumerate(fx.iterrows(), 1):
        home = str(r.get("HomeTeam", "")).strip()
        away = str(r.get("AwayTeam", "")).strip()
        league = str(r.get("League", "")).strip()
        if not home or not away:
            continue
        baseline = get_league_baseline(league)
        odds, move_corr, consensus_corr = get_match_odds_from_cache(home, away, all_odds_cache)
        odds_prob = predict_from_odds(odds) if odds else None
        home_win_rate, home_gf, home_ga = fetch_team_recent_form(home, api_football_key)
        away_win_rate, away_gf, away_ga = fetch_team_recent_form(away, api_football_key)
        total_win = home_win_rate + away_win_rate
        form_ph = home_win_rate / total_win if total_win > 0 else 0.45
        form_pa = away_win_rate / total_win if total_win > 0 else 0.27
        form_pd = 1 - form_ph - form_pa
        form_prob = (form_ph, form_pd, form_pa)
        h2h_ph, h2h_pd, h2h_pa = fetch_h2h_data(home, away, api_football_key)
        h2h_prob = (h2h_ph, h2h_pd, h2h_pa)
        total_gf = home_gf + away_gf
        xg_ph = home_gf / total_gf if total_gf > 0 else 0.45
        xg_pa = away_gf / total_gf if total_gf > 0 else 0.27
        xg_pd = 1 - xg_ph - xg_pa
        xg_prob = (xg_ph, xg_pd, xg_pa)
        injury_home_corr, injury_away_corr = fetch_injury_correction(home, away, api_football_key)
        morale_corr, fitness_corr, upset_corr = get_fixture_corrections(home, away, league, api_football_key)
        total_correction = (
            injury_home_corr - injury_away_corr
            + move_corr + consensus_corr
            + morale_corr + fitness_corr + upset_corr
        )
        total_correction = max(-0.08, min(0.08, total_correction))
        ph, pd_, pa, weights = fuse_multi_factor_probs(
            odds_prob=odds_prob,
            form_prob=form_prob,
            h2h_prob=h2h_prob,
            xg_prob=xg_prob,
            league_baseline=baseline,
            total_correction=total_correction,
        )
        xg_home = round(home_gf + (ph - 0.45)*2, 2)
        xg_away = round(away_gf + (pa - 0.27)*2, 2)
        if pd_ >= max(ph, pa):
            most_likely_score = "1-1" if pd_ > 0.30 else "0-0"
        elif ph >= pa:
            gap = ph - pa
            most_likely_score = "3-1" if gap > 0.22 else "2-1" if gap > 0.10 else "1-0"
        else:
            gap = pa - ph
            most_likely_score = "1-3" if gap > 0.22 else "1-2" if gap > 0.10 else "0-1"
        evv = None
        kellyv = None
        pick = "无推荐"
        pick_score = None
        status = "-"
        if odds and all(odds):
            q1, qx, q2 = implied_prob(odds[0]), implied_prob(odds[1]), implied_prob(odds[2])
            f1, fx_, f2 = remove_overround(q1, qx, q2)
            best = max(
                [calc(ph, odds[0], f1, "主胜"), calc(pd_, odds[1], fx_, "平"), calc(pa, odds[2], f2, "客胜")],
                key=lambda x: x.ev,
            )
            evv = round(best.ev, 4)
            kellyv = round(min(best.kelly, 0.08), 4)
            pick = best.pick
            pick_score = score(best)
            status = label(pick_score)
        dt = pd.to_datetime(r.get("Date"), errors="coerce")
        kick_time = str(r.get("time", "")).strip()
        rows.append({
            "date": dt.strftime("%Y-%m-%d") if not pd.isna(dt) else str(r.get("date", "")),
            "time": kick_time,
            "league": league,
            "home": home,
            "away": away,
            "xg_home": xg_home,
            "xg_away": xg_away,
            "p_home": ph,
            "p_draw": pd_,
            "p_away": pa,
            "form_prob": [round(form_ph,4), round(form_pd,4), round(form_pa,4)],
            "odds_prob": [round(odds_prob[0],4), round(odds_prob[1],4), round(odds_prob[2],4)] if odds_prob else None,
            "h2h_prob": [round(h2h_ph,4), round(h2h_pd,4), round(h2h_pa,4)],
            "xg_prob": [round(xg_ph,4), round(xg_pd,4), round(xg_pa,4)],
            "most_likely_score": most_likely_score,
            "odds_win": odds[0] if odds else None,
            "odds_draw": odds[1] if odds else None,
            "odds_lose": odds[2] if odds else None,
            "ev": evv,
            "kelly": kellyv,
            "pick": pick,
            "score": pick_score,
            "label": status,
            "why": (
                f"融合权重 赔率:{weights['odds']:.2f} 状态:{weights['form']:.2f} 交锋:{weights['h2h']:.2f} xG:{weights['xg']:.2f}; "
                f"主胜率{ph*100:.1f}%, xG差{xg_home - xg_away:.2f}"
            ),
        })
    bt = {
        "matches_used": len(rows),
        "bets": len([x for x in rows if x.get('ev') and x['ev'] > 0]),
        "roi": 0.0,
        "hit_rate": 0.0,
        "avg_ev": 0.0,
        "logloss": 0.0
    }
    return rows, bt

def build_payload(rows: List[Dict], bt: Dict, llm_cfg: LLMConfig) -> Dict:
    ranked = [x for x in rows if x.get("ev") is not None and x.get("ev") > 0]
    ranked.sort(key=lambda x: (x.get("score") or 0, x.get("ev") or -9), reverse=True)
    top = ranked[:TOP_N] if ranked else sorted(rows, key=lambda x: x.get("p_home", 0), reverse=True)[:TOP_N]
    llm_used = {"doubao": 0, "fallback": 0, "skipped": 0}
    for idx, p in enumerate(top):
        if p.get("ev") is None or p.get("pick") == "无推荐":
            base_reason = str(p.get("why", ""))
            p["why"] = f"{base_reason} | 赛事分析: 暂无赔率数据，仅提供基础胜率预测"
            p["llm_status"] = "skipped"
            p["reasons"] = {"base": base_reason, "doubao": None, "fallback": None}
            llm_used["skipped"] +=1
            continue
        base_reason = str(p.get("why", ""))
        llm_reason, llm_status, doubao_reason = build_llm_reason(llm_cfg, p)
        llm_used[llm_status] = llm_used.get(llm_status, 0) + 1
        p["why"] = f"{base_reason} | {llm_reason}"
        p["llm_status"] = llm_status
        p["reasons"] = {"base": base_reason, "doubao": doubao_reason, "fallback": llm_reason if llm_status == "fallback" else None}
    for pick in top:
        pick["home"] = team_name_to_cn(pick["home"])
        pick["away"] = team_name_to_cn(pick["away"])
    for match in rows:
        match["home"] = team_name_to_cn(match["home"])
        match["away"] = team_name_to_cn(match["away"])
    return {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "data_source": "Football-Data + API-Sports + The-Odds-API",
            "league_scope": "五大联赛 + 欧冠 + 欧联 + 国际赛事",
            "fusion": {
                "WEIGHT_ODDS": WEIGHT_ODDS,
                "WEIGHT_FORM": WEIGHT_FORM,
                "WEIGHT_H2H": WEIGHT_H2H,
                "WEIGHT_XG": WEIGHT_XG,
            },
            "llm": {"model": llm_cfg.model, "base_url": llm_cfg.base_url, "usage": llm_used},
            "schedule_bjt": ["09:30", "21:30"],
        },
        "stats": {"fixtures": len(rows), "top": len(top), "backtest": bt},
        "top_picks": top,
        "all": rows,
    }

def write_outputs(payload: Dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    TOP_PATH.write_text(json.dumps(payload.get("top_picks", []), ensure_ascii=False, indent=2), encoding="utf-8")
    PREDICTIONS_PATH.write_text(json.dumps(payload.get("all", []), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "picks_updated.json").write_text(json.dumps(payload.get("top_picks", []), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "complete_predictions.json").write_text(json.dumps(payload.get("all", []), ensure_ascii=False, indent=2), encoding="utf-8")

def load_runtime_env() -> None:
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=True)

def load_llm_config() -> LLMConfig:
    api_key = env_value("OPENAI_API_KEY", "OPENAI_KEY")
    return LLMConfig(
        base_url=env_value("OPENAI_BASE_URL", default="https://ark.cn-beijing.volces.com/api/v3"),
        api_key=api_key if valid_key(api_key) else "",
        model=env_value("OPENAI_MODEL", default="doubao-1.5-pro-32k"),
    )

def run() -> int:
    load_runtime_env()
    Path("site").mkdir(parents=True, exist_ok=True)
    Path("site/.nojekyll").write_text("", encoding="utf-8")
    probe = probe_external_connections()
    fx = load_fixtures()
    odds_api_key = env_value("ODDS_API_KEY", "THE_ODDS_API_KEY")
    all_odds_cache = fetch_all_league_odds(odds_api_key)
    rows, bt = build_prediction_rows(fx, all_odds_cache)
    llm_cfg = load_llm_config()
    payload = build_payload(rows, bt, llm_cfg)
    payload.setdefault("meta", {})["connection_probe"] = probe
    write_outputs(payload)
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
