#!/usr/bin/env python3
"""Daily Top League Football Prediction Pipeline
【最终落地修复版】
核心修复：赔率401报错、过时赛事、过滤bug、无效API调用
权重架构：多机构赔率共识(45%) + 球队近期状态(30%) + 交锋&赛事战意(15%) + xG等效修正(10%)
赛事源：API-Football（免费版可用，修复过滤bug）
辅助API：The-Odds-API（修复401报错）、豆包AI（赛事分析，仅有效赛事调用）
赛事范围：五大联赛 + 欧冠 + 欧联 + 国际赛事
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

# 原有核心计算模块100%完全保留，无任何修改，完全兼容你的项目
from src.backtest.backtest import backtest
from src.engine.value import calc, implied_prob, label, remove_overround, score
from src.models.bookmaker import predict_from_odds
from src.models.upset import avoid_upset

# ====================== 固定配置 无需修改 ======================
# 联赛与赔率API的精准映射（彻底解决之前固定请求英超接口的bug）
LEAGUE_ODDS_MAPPING = {
    "英超": "soccer_epl",
    "西甲": "soccer_spain_la_liga",
    "德甲": "soccer_germany_bundesliga",
    "意甲": "soccer_italy_serie_a",
    "法甲": "soccer_france_ligue_one",
    "欧冠": "soccer_uefa_champs_league",
    "欧联": "soccer_uefa_europa_league",
    "世界杯": "soccer_fifa_world_cup",
    "欧洲杯": "soccer_uefa_european_championship",
    "国际友谊赛": "soccer_international_friendly",
}
# 冷启动专属权重架构 严格落地
WEIGHT_ODDS = 0.45    # 多机构赔率共识+异动 45%
WEIGHT_FORM = 0.30    # 球队近期状态 30%
WEIGHT_H2H = 0.15     # 历史交锋+赛事战意 15%
WEIGHT_XG = 0.10       # xG等效修正 10%
# 五大联赛专属基线
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
# 输出路径 完全兼容原有前端，无需修改
OUT_DIR = Path("site/data")
PICKS_PATH = OUT_DIR / "picks.json"
TOP_PATH = OUT_DIR / "top_picks.json"
PREDICTIONS_PATH = OUT_DIR / "predictions.json"
# 基础配置
FUTURE_DAYS = 7  # 未来7天赛事
PAST_DAYS = 0    # 彻底不包含过去的赛事，只看未来
TOP_N = 4
# 当前赛季（API-Football 2025-2026赛季编码为2025）
CURRENT_SEASON = 2025
# ========================================================================

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
        "豆包pro": "doubao-1.5-pro-32k", "豆包flash": "doubao-1.5-flash-32k",
        "doubao-pro": "doubao-1.5-pro-32k", "doubao-flash": "doubao-1.5-flash-32k",
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
            push("doubao-1.5-flash-32k")
        if "flash" in low:
            push("doubao-1.5-flash-32k")
            push("doubao-1.5-pro-32k")
    return out

def _team_name_quality(name: str) -> bool:
    t = str(name or "").strip()
    if len(t) < 2:
        return False
    alnum = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", t)
    if not alnum:
        return False
    return True

def get_league_baseline(league_name: str) -> Dict[str, float]:
    if not league_name:
        return LEAGUE_BASELINE["default"]
    league_lower = league_name.strip().lower()
    for key, baseline in LEAGUE_BASELINE.items():
        if key in league_lower:
            return baseline
    return LEAGUE_BASELINE["default"]

def _norm_team(name: str) -> str:
    """球队名标准化，解决翻译、简写、特殊符号匹配问题"""
    n = (name or "").strip().lower()
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"fc|cf|ac|sc|1910|1913|1909|1907|1908", "", n)
    n = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", n)
    return n.strip()

# ====================== 核心因子计算模块 100%保留 ======================
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
    except Exception as e:
        print(f"获取球队状态失败 {team_name}: {str(e)}")
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
    except Exception as e:
        print(f"获取交锋数据失败 {home_team} vs {away_team}: {str(e)}")
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
    except Exception as e:
        print(f"获取伤停数据失败 {home_team} vs {away_team}: {str(e)}")
        return 0.0, 0.0

# ====================== 【核心修复】赔率获取函数，彻底解决401报错 ======================
def fetch_fixture_odds_data(home_team: str, away_team: str, league_name: str, api_key: str) -> Tuple[Optional[Tuple[float, float, float]], float, float]:
    """
    【修复版】彻底解决401报错
    1.  联赛与接口精准匹配，不再固定请求英超接口
    2.  修复URL参数编码bug，不再手动编码，避免二次编码导致API读不到密钥
    3.  优化球队名匹配，解决翻译/简写差异
    4.  详细日志打印，方便排查问题
    """
    if not valid_key(api_key):
        print(f"跳过赔率获取：ODDS_API_KEY无效")
        return None, 0.0, 0.0
    
    # 【核心修复1】根据联赛匹配正确的接口，不再固定用英超
    sport_code = None
    for league_key, code in LEAGUE_ODDS_MAPPING.items():
        if league_key in league_name:
            sport_code = code
            break
    if not sport_code:
        print(f"跳过赔率获取：{league_name} 无对应赔率接口")
        return None, 0.0, 0.0

    try:
        norm_home = _norm_team(home_team)
        norm_away = _norm_team(away_team)
        opening_odds = None
        current_odds = None
        bookmaker_count = 0
        consensus_count = 0

        print(f"正在获取【{league_name}】{home_team} vs {away_team} 的赔率，接口: {sport_code}")
        # 【核心修复2】参数直接用原始逗号分隔，不手动编码，requests自动处理，彻底解决编码bug
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
        # 精准处理401报错，直接提示问题
        if resp.status_code == 401:
            print(f"❌ 赔率API授权失败：401 Unauthorized，请检查密钥是否正确、免费额度是否用完")
            print(f"API响应详情: {resp.text}")
            return None, 0.0, 0.0
        if resp.status_code == 403:
            print(f"❌ 赔率API访问被拒绝：403 Forbidden，账号可能被限制")
            return None, 0.0, 0.0
        resp.raise_for_status()
        events = resp.json() or []
        print(f"【{league_name}】接口返回赛事数量: {len(events)}")

        # 【优化】球队名精准匹配+模糊匹配，解决翻译差异
        matched_event = None
        for event in events:
            event_home = _norm_team(event.get("home_team", ""))
            event_away = _norm_team(event.get("away_team", ""))
            # 1. 精准匹配
            if (event_home == norm_home and event_away == norm_away) or (event_home == norm_away and event_away == norm_home):
                matched_event = event
                break
            # 2. 模糊匹配，兼容简写/翻译差异
            if (norm_home in event_home or event_home in norm_home) and (norm_away in event_away or event_away in norm_away):
                matched_event = event
                break

        if not matched_event:
            print(f"未匹配到对应赛事: {home_team} vs {away_team}")
            return None, 0.0, 0.0
        print(f"✅ 成功匹配赛事: {matched_event.get('home_team')} vs {matched_event.get('away_team')}")

        # 提取开盘赔率和多机构实时赔率均值
        bookmakers = matched_event.get("bookmakers", [])
        bookmaker_count = len(bookmakers)
        if not bookmakers:
            print(f"赛事无可用赔率数据")
            return None, 0.0, 0.0
        
        # 取第一个机构的开盘赔率
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
        
        # 计算所有机构的实时赔率均值
        total_oh = total_od = total_oa = 0.0
        valid_count = 0
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
                        # 统计机构共识
                        if opening_odds and oh < opening_odds[0]:
                            consensus_count +=1
                        break
        if valid_count > 0:
            current_odds = (
                round(total_oh/valid_count, 2),
                round(total_od/valid_count, 2),
                round(total_oa/valid_count, 2),
            )
            print(f"✅ 成功获取赔率: 主胜{current_odds[0]} 平{current_odds[1]} 客胜{current_odds[2]}")

        # 计算赔率异动和机构共识修正
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
    except Exception as e:
        print(f"获取赔率数据失败 {home_team} vs {away_team}: {str(e)}")
        return None, 0.0, 0.0

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
                params={"league": league_id, "season": CURRENT_SEASON},
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
    except Exception as e:
        print(f"获取综合修正失败 {home_team} vs {away_team}: {str(e)}")
        return 0.0, 0.0, 0.0

# ====================== 多因子融合核心逻辑 100%保留 ======================
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

# ====================== API连通性探测 100%保留 ======================
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

    doubao_models = env_value("OPENAI_MODEL", default="doubao-1.5-pro-32k")
    doubao_probe = {"ok": False, "error": "all_model_failed"}
    for cand in parse_model_candidates(doubao_models):
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

# ====================== 豆包AI分析模块 【修复】无效赛事不调用，不浪费额度 ======================
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

def build_llm_reason(cfg: LLMConfig, pick: Dict[str, object]) -> Tuple[str, str, Optional[str]]:
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

# ====================== 【核心修复】赛事获取函数，只拿未来未开赛赛事，修复过滤bug ======================
def is_allowed_league(league_name: str) -> bool:
    """【修复版】只过滤明确的低级别联赛，不再误筛带赛季年份的五大联赛"""
    if not league_name:
        return False
    league_lower = league_name.strip().lower()
    # 只过滤明确的低级别联赛，绝对不碰带赛季年份的联赛
    ban_keywords = [
        "serie b", "segunda", "championship", "league 2", "league 3", "league 4",
        "division 2", "division 3", "2nd division", "3rd division", "4th division",
        "reserve", "youth", "amateur", "women", "female"
    ]
    for ban in ban_keywords:
        if ban in league_lower:
            print(f"过滤低级别联赛: {league_name}")
            return False
    # 只保留目标联赛
    allow_keywords = [
        "premier league", "la liga", "bundesliga", "serie a", "ligue 1",
        "champions league", "europa league", "world cup", "euro",
        "international friendly", "world cup qualification", "euro qualification"
    ]
    for allow in allow_keywords:
        if allow in league_lower:
            return True
    print(f"非目标联赛，过滤: {league_name}")
    return False

def fetch_api_fixtures(start: datetime, end: datetime) -> List[Dict[str, object]]:
    """【修复版】只获取未来未开赛的真实赛事，彻底杜绝过时数据"""
    api_key = env_value("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY")
    if not valid_key(api_key):
        print("❌ API_FOOTBALL_KEY无效，无法获取赛事数据")
        return []
    base = "https://v3.football.api-sports.io"
    headers = {"x-apisports-key": api_key}
    out: List[Dict[str, object]] = []

    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")
    print(f"开始获取赛事数据，日期范围: {start_date} 至 {end_date}")
    now_utc = datetime.now(timezone.utc)
    print(f"当前UTC时间: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 一次性拉取日期范围内所有赛事，减少API请求次数，避免触发限流
        resp = requests.get(
            f"{base.rstrip('/')}/fixtures",
            headers=headers,
            params={
                "from": start_date,
                "to": end_date,
                "timezone": "UTC",
                "season": CURRENT_SEASON,
            },
            timeout=20,
        )
        resp_json = resp.json()
        print(f"API响应状态: {resp.status_code} | 错误信息: {resp_json.get('errors', '无')}")
        resp.raise_for_status()
        items = resp_json.get("response", [])
        print(f"API返回总赛事数量: {len(items)}")

        # 【双层严格过滤】只保留未来未开赛的目标赛事
        valid_status = ["NS", "TBD", "PST"] # 未开赛、待定、延期，绝对不包含已结束的
        valid_count = 0
        for m in items:
            # 1. 状态过滤：只保留未开赛的
            fixture = m.get("fixture") or {}
            fixture_status = (fixture.get("status") or {}).get("short") or ""
            if fixture_status not in valid_status:
                continue
            # 2. 时间过滤：开赛时间必须晚于当前时间，双重保险
            try:
                fixture_date = datetime.fromisoformat(fixture.get("date", "").replace("Z", "+00:00"))
            except:
                continue
            if fixture_date <= now_utc:
                continue
            # 3. 联赛过滤：只保留目标联赛，修复之前的误筛bug
            league_name = ((m.get("league") or {}).get("name")) or ""
            if not is_allowed_league(league_name):
                continue
            # 4. 球队名校验
            teams = m.get("teams") or {}
            home = ((teams.get("home") or {}).get("name")) or ""
            away = ((teams.get("away") or {}).get("name")) or ""
            if not home or not away or not _team_name_quality(home) or not _team_name_quality(away):
                continue
            # 格式化时间
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
                "source": "api-football",
                "fixture_date_utc": fixture_date,
            })
            valid_count +=1
        print(f"✅ 最终过滤后有效未来赛事总数: {valid_count}")
    except Exception as e:
        print(f"获取赛事失败: {str(e)}")
    return out

def load_fixtures() -> pd.DataFrame:
    """加载赛事数据，完全兼容原有前端逻辑"""
    utc_now = datetime.now(timezone.utc)
    today = datetime.strptime(utc_now.strftime("%Y-%m-%d"), "%Y-%m-%d")
    start = today - timedelta(days=PAST_DAYS)
    upper = today + timedelta(days=max(FUTURE_DAYS, 7))
    rows = fetch_api_fixtures(start, upper)
    if not rows:
        print("❌ 未获取到任何赛事数据")
        return pd.DataFrame()
    # 转换成原有逻辑兼容的DataFrame格式
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

# ====================== 核心预测流水线 【修复】赔率接口传参、无效赛事过滤 ======================
def build_prediction_rows(fx: pd.DataFrame) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    api_football_key = env_value("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY")
    odds_api_key = env_value("ODDS_API_KEY", "THE_ODDS_API_KEY")
    rows: List[Dict[str, object]] = []

    total_fixtures = len(fx)
    print(f"开始生成预测，总赛事数量: {total_fixtures}")
    for idx, (_, r) in enumerate(fx.iterrows()):
        home = str(r.get("HomeTeam", "")).strip()
        away = str(r.get("AwayTeam", "")).strip()
        league = str(r.get("League", "")).strip()
        print(f"\n[{idx+1}/{total_fixtures}] 处理赛事: {home} vs {away} | {league}")
        if not home or not away:
            print("  跳过：主队/客队名称为空")
            continue
        baseline = get_league_baseline(league)
        # 【修复】传入联赛名称，匹配正确的赔率接口，不再固定英超
        odds, move_corr, consensus_corr = fetch_fixture_odds_data(home, away, league, odds_api_key)
        odds_prob = predict_from_odds(odds) if odds else None
        # 获取球队状态、交锋、伤停数据
        home_win_rate, home_gf, home_ga = fetch_team_recent_form(home, api_football_key)
        away_win_rate, away_gf, away_ga = fetch_team_recent_form(away, api_football_key)
        # 状态概率计算
        total_win = home_win_rate + away_win_rate
        form_ph = home_win_rate / total_win if total_win > 0 else 0.45
        form_pa = away_win_rate / total_win if total_win > 0 else 0.27
        form_pd = 1 - form_ph - form_pa
        form_prob = (form_ph, form_pd, form_pa)
        # 历史交锋数据
        h2h_ph, h2h_pd, h2h_pa = fetch_h2h_data(home, away, api_football_key)
        h2h_prob = (h2h_ph, h2h_pd, h2h_pa)
        # xG等效修正概率
        total_gf = home_gf + away_gf
        xg_ph = home_gf / total_gf if total_gf > 0 else 0.45
        xg_pa = away_gf / total_gf if total_gf > 0 else 0.27
        xg_pd = 1 - xg_ph - xg_pa
        xg_prob = (xg_ph, xg_pd, xg_pa)
        # 综合修正值计算
        injury_home_corr, injury_away_corr = fetch_injury_correction(home, away, api_football_key)
        morale_corr, fitness_corr, upset_corr = get_fixture_corrections(home, away, league, api_football_key)
        total_correction = (
            injury_home_corr - injury_away_corr
            + move_corr + consensus_corr
            + morale_corr + fitness_corr + upset_corr
        )
        total_correction = max(-0.08, min(0.08, total_correction))
        # 多因子融合
        ph, pd_, pa, weights = fuse_multi_factor_probs(
            odds_prob=odds_prob,
            form_prob=form_prob,
            h2h_prob=h2h_prob,
            xg_prob=xg_prob,
            league_baseline=baseline,
            total_correction=total_correction,
        )
        # 预期进球与比分预测
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
        # EV/Kelly计算
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
        # 格式化输出，完全兼容原有前端
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
    # 回测数据兼容
    bt = {
        "matches_used": len(rows),
        "bets": len([x for x in rows if x.get('ev') and x['ev'] > 0]),
        "roi": 0.0,
        "hit_rate": 0.0,
        "avg_ev": 0.0,
        "logloss": 0.0
    }
    print(f"\n✅ 预测生成完成，有效赛事: {len(rows)}，正EV推荐: {bt['bets']}")
    return rows, bt

# ====================== 数据导出&兼容 100%保留 ======================
def build_payload(rows: List[Dict[str, object]], bt: Dict[str, object], llm_cfg: LLMConfig) -> Dict[str, object]:
    # 只保留有正EV值的赛事作为精选推荐
    ranked = [x for x in rows if x.get("ev") is not None and x.get("ev") > 0]
    ranked.sort(key=lambda x: (x.get("score") or 0, x.get("ev") or -9), reverse=True)
    # 没有正EV推荐时，取置信度最高的4场
    top = ranked[:TOP_N] if ranked else sorted(rows, key=lambda x: x.get("p_home", 0), reverse=True)[:TOP_N]
    # 豆包AI分析，仅有效赛事调用，彻底避免浪费额度
    llm_used = {"doubao": 0, "fallback": 0}
    print(f"\n开始生成豆包AI分析，精选推荐数量: {len(top)}")
    for idx, p in enumerate(top):
        # 【核心修复】无效赛事直接跳过，不调用豆包AI，不浪费额度
        if p.get("ev") is None or p.get("pick") == "无推荐":
            print(f"  跳过无效赛事，不调用豆包AI: {p['home']} vs {p['away']}")
            base_reason = str(p.get("why", ""))
            p["why"] = f"{base_reason} | 赛事分析: 暂无赔率数据，仅提供基础胜率预测"
            p["llm_status"] = "skipped"
            p["reasons"] = {"base": base_reason, "doubao": None, "fallback": None}
            continue
        print(f"  生成第{idx+1}场分析: {p['home']} vs {p['away']}")
        base_reason = str(p.get("why", ""))
        llm_reason, llm_status, doubao_reason = build_llm_reason(llm_cfg, p)
        llm_used[llm_status] = llm_used.get(llm_status, 0) + 1
        p["why"] = f"{base_reason} | {llm_reason}"
        p["llm_status"] = llm_status
        p["reasons"] = {"base": base_reason, "doubao": doubao_reason, "fallback": llm_reason if llm_status == "fallback" else None}
    # 输出元数据，完全兼容原有前端
    return {
        "meta": {
            "generated_at_utc": utc_now_str(),
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "data_source": "API-Sports + The-Odds-API",
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

def write_outputs(payload: Dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 主文件，前端核心读取文件
    PICKS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    TOP_PATH.write_text(json.dumps(payload.get("top_picks", []), ensure_ascii=False, indent=2), encoding="utf-8")
    PREDICTIONS_PATH.write_text(json.dumps(payload.get("all", []), ensure_ascii=False, indent=2), encoding="utf-8")
    # 兼容旧版前端路径
    (OUT_DIR / "picks_updated.json").write_text(json.dumps(payload.get("top_picks", []), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "complete_predictions.json").write_text(json.dumps(payload.get("all", []), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 数据已导出到 {OUT_DIR}，文件大小: {PICKS_PATH.stat().st_size} 字节")

def load_runtime_env() -> None:
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=True)

def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def load_llm_config() -> LLMConfig:
    api_key = env_value("OPENAI_API_KEY", "OPENAI_KEY")
    return LLMConfig(
        base_url=env_value("OPENAI_BASE_URL", default="https://ark.cn-beijing.volces.com/api/v3"),
        api_key=api_key if valid_key(api_key) else "",
        model=env_value("OPENAI_MODEL", default="doubao-1.5-pro-32k"),
    )

# ====================== 主入口 完整流水线 ======================
def run() -> int:
    load_runtime_env()
    Path("site").mkdir(parents=True, exist_ok=True)
    Path("site/.nojekyll").write_text("", encoding="utf-8")
    # 环境状态打印
    api_football_on = valid_key(env_value("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY"))
    odds_api_on = valid_key(env_value("ODDS_API_KEY", "THE_ODDS_API_KEY"))
    llm_doubao_on = valid_key(env_value("OPENAI_API_KEY", "OPENAI_KEY"))
    print(f"[env] api_football={api_football_on} odds_api={odds_api_on} doubao_llm={llm_doubao_on}")
    if not (api_football_on or odds_api_on or llm_doubao_on):
        print("WARN 核心密钥缺失，请在GitHub Secrets中配置")
    # API连通性探测
    probe = probe_external_connections()
    print("[probe]", json.dumps(probe, ensure_ascii=False))
    # 流水线执行
    print(f"\n[1/4] 获取赛事数据 {utc_now_str()}")
    fx = load_fixtures()
    print(f"\n[2/4] 多因子融合预测，赛事数量={len(fx)}")
    rows, bt = build_prediction_rows(fx)
    print(f"\n[3/4] 豆包AI赛事分析")
    llm_cfg = load_llm_config()
    payload = build_payload(rows, bt, llm_cfg)
    payload.setdefault("meta", {})["connection_probe"] = probe
    print(f"\n[4/4] 导出前端数据")
    write_outputs(payload)
    print(f"\nDONE 执行完成，精选推荐={len(payload['top_picks'])}，全量赛事={len(payload['all'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
