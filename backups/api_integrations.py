"""
多源足球数据API集成
集成football-data.org, understat, flashscore等数据源
"""

import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballDataAPI:
    """football-data.org 官方API"""
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.headers = {"X-Auth-Token": api_key} if api_key else {}
    
    def get_competitions(self):
        """获取所有支持的联赛"""
        try:
            resp = requests.get(f"{self.BASE_URL}/competitions", headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json().get('competitions', [])
        except Exception as e:
            logger.error(f"Failed to get competitions: {e}")
            return []
    
    def get_matches(self, competition_code: str = "PL", status: str = "SCHEDULED", days: int = 7):
        """
        获取指定联赛的赛程
        PL=英超, SA=西甲, BL1=德甲, FR1=法甲, IT1=意甲
        """
        try:
            params = {
                "status": status,
                "dateFrom": datetime.now().isoformat(),
                "dateTo": (datetime.now() + timedelta(days=days)).isoformat()
            }
            url = f"{self.BASE_URL}/competitions/{competition_code}/matches"
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json().get('matches', [])
        except Exception as e:
            logger.error(f"Failed to get matches: {e}")
            return []
    
    def get_team_standings(self, competition_code: str):
        """获取联赛积分榜"""
        try:
            url = f"{self.BASE_URL}/competitions/{competition_code}/standings"
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json().get('standings', [])
        except Exception as e:
            logger.error(f"Failed to get standings: {e}")
            return []
    
    def get_team_stats(self, team_id: int):
        """获取球队详细统计"""
        try:
            url = f"{self.BASE_URL}/teams/{team_id}"
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get team stats: {e}")
            return {}


class UnderstatAPI:
    """Understat数据（xG、射门等）"""
    BASE_URL = "https://understat.com/api"
    
    @staticmethod
    def get_team_xg_stats(league: str = "EPL") -> Dict:
        """获取球队xG统计"""
        try:
            url = f"{UnderstatAPI.BASE_URL}/get_league_squad_exp_stats/{league}/2024"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get Understat xG: {e}")
            return {}
    
    @staticmethod
    def get_match_data(match_id: int) -> Dict:
        """获取具体比赛的xG数据"""
        try:
            url = f"{UnderstatAPI.BASE_URL}/match/{match_id}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get match xG data: {e}")
            return {}


class OddsAPI:
    """赔率数据API"""
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def get_upcoming_matches(self, sport: str = "soccer_epl", regions: str = "uk"):
        """获取即将进行的比赛赔率"""
        try:
            params = {
                "apiKey": self.api_key,
                "regions": regions,
                "markets": "h2h,spreads,totals"
            }
            url = f"{self.BASE_URL}/sports/{sport}/events"
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json().get('data', [])
        except Exception as e:
            logger.error(f"Failed to get odds: {e}")
            return []


class SofascoreAPI:
    """Sofascore快照数据API"""
    BASE_URL = "https://api.sofascore.com/api"
    
    @staticmethod
    def get_match_statistics(match_id: int) -> Dict:
        """获取比赛统计数据"""
        try:
            url = f"{SofascoreAPI.BASE_URL}/v1/event/{match_id}/statistics"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get Sofascore stats: {e}")
            return {}
    
    @staticmethod
    def get_team_form(team_id: int, limit: int = 10) -> List[Dict]:
        """获取球队最近比赛"""
        try:
            url = f"{SofascoreAPI.BASE_URL}/v1/team/{team_id}/events/last/{limit}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json().get('events', [])
        except Exception as e:
            logger.error(f"Failed to get team form: {e}")
            return []


class DataAggregator:
    """数据聚合器 - 合并多个API源"""
    
    def __init__(self, football_api_key: str = None, odds_api_key: str = None):
        self.fdb = FootballDataAPI(football_api_key)
        self.understat = UnderstatAPI()
        self.odds = OddsAPI(odds_api_key)
        self.sofascore = SofascoreAPI()
    
    def get_comprehensive_match_data(self, match: Dict) -> Dict:
        """获取单场比赛的综合数据"""
        enhanced = {
            "basic": match,
            "odds": [],
            "team_form": {
                "home": {},
                "away": {}
            },
            "xg_stats": {},
            "head_to_head": []
        }
        
        try:
            # 获取赔率数据
            enhanced["odds"] = self.odds.get_upcoming_matches()
            
            # 获取球队数据（如果有ID）
            if "homeTeam" in match:
                team_form = self.sofascore.get_team_form(match["homeTeam"].get("id"))
                enhanced["team_form"]["home"] = team_form
            
            if "awayTeam" in match:
                team_form = self.sofascore.get_team_form(match["awayTeam"].get("id"))
                enhanced["team_form"]["away"] = team_form
        
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
        
        return enhanced
    
    def get_league_data(self, competition_code: str = "PL") -> Dict:
        """获取完整联赛数据"""
        return {
            "standings": self.fdb.get_team_standings(competition_code),
            "matches": self.fdb.get_matches(competition_code),
            "xg_stats": self.understat.get_team_xg_stats()
        }


# 快速工厂函数
def create_data_aggregator(football_api_key: str = None, odds_api_key: str = None) -> DataAggregator:
    """创建数据聚合器实例"""
    return DataAggregator(football_api_key, odds_api_key)
