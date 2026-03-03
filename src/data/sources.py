import pandas as pd
import requests
from io import StringIO
from datetime import date
from typing import Optional, Tuple
import numpy as np

LEAGUES = {
    "E0": "EPL",
    "SP1": "LaLiga",
    "I1": "SerieA",
    "D1": "Bundesliga",
    "F1": "Ligue1",
}

def season_code_for(d: date) -> str:
    if d.month >= 7:
        return f"{d.year%100:02d}{(d.year+1)%100:02d}"
    return f"{(d.year-1)%100:02d}{d.year%100:02d}"

def prev_season(code: str, n: int = 1) -> str:
    s = int(code[:2]); e = int(code[2:])
    s = (s - n) % 100
    e = (e - n) % 100
    return f"{s:02d}{e:02d}"

def _fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def fetch_league(div: str, season: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
    df = _fetch_csv(url)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df

def split_played_future(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]:
        if col not in df.columns:
            df[col] = pd.NA
    base = df.dropna(subset=["Date","HomeTeam","AwayTeam"]).copy()

    played = base.dropna(subset=["FTHG","FTAG"]).copy()
    played = played[(played["FTHG"].astype(str).str.strip()!="") & (played["FTAG"].astype(str).str.strip()!="")]
    played["FTHG"] = played["FTHG"].astype(int)
    played["FTAG"] = played["FTAG"].astype(int)

    future = base[base["FTHG"].isna() | base["FTAG"].isna()].copy()
    return played, future

def fetch_fixtures_fallback() -> pd.DataFrame:
    """
    兜底赛程源：fixtures.csv（通常含未来赛程）
    """
    url = "https://www.football-data.co.uk/fixtures.csv"
    df = _fetch_csv(url)

    for c in ["Div","Date","HomeTeam","AwayTeam"]:
        if c not in df.columns:
            df[c] = pd.NA
    if "Time" not in df.columns:
        df["Time"] = ""

    df = df[df["Div"].isin(list(LEAGUES.keys()))].copy()
    df["League"] = df["Div"].map(LEAGUES)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam"]).copy()
    return df

def _try_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None

def pick_1x2_odds(row: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    优先 B365，其次 PS/WH/VC...
    """
    candidates = [
        ("B365H","B365D","B365A","B365"),
        ("PSH","PSD","PSA","PS"),
        ("WHH","WHD","WHA","WH"),
        ("VCH","VCD","VCA","VC"),
        ("IWH","IWD","IWA","IW"),
        ("BWH","BWD","BWA","BW"),
    ]
    for h,d,a,book in candidates:
        if h in row.index and d in row.index and a in row.index:
            oh, od, oa = _try_float(row.get(h)), _try_float(row.get(d)), _try_float(row.get(a))
            if oh and od and oa and oh>1 and od>1 and oa>1:
                return oh, od, oa, book
    return None, None, None, "-"
