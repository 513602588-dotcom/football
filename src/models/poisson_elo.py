import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HOME_ADV_ELO = 60.0
K_ELO = 20.0
ELO_GOAL_SCALER = 0.12
MAX_GOALS = 10

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

@dataclass
class FitModels:
    home: Pipeline
    away: Pipeline
    elo: Dict[str, float]

def run_elo(df: pd.DataFrame) -> Dict[str, float]:
    ratings: Dict[str, float] = {}
    def get(t): return ratings.get(t, 1500.0)

    df = df.sort_values(["Date","HomeTeam","AwayTeam"])
    for _, r in df.iterrows():
        ht, at = r["HomeTeam"], r["AwayTeam"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        Rh, Ra = get(ht), get(at)

        Eh = 1.0 / (1.0 + 10 ** ((Ra - (Rh + HOME_ADV_ELO)) / 400.0))
        Sh = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        ratings[ht] = Rh + K_ELO * (Sh - Eh)
        ratings[at] = Ra + K_ELO * ((1.0 - Sh) - (1.0 - Eh))
    return ratings

def fit_poisson(df: pd.DataFrame) -> Tuple[Pipeline, Pipeline]:
    X = df[["HomeTeam","AwayTeam"]].copy()
    y_h = df["FTHG"].astype(float)
    y_a = df["FTAG"].astype(float)

    pre = ColumnTransformer(
        [("teams", OneHotEncoder(handle_unknown="ignore"), ["HomeTeam","AwayTeam"])],
        remainder="drop",
    )
    m_h = Pipeline([("pre", pre), ("reg", PoissonRegressor(alpha=1e-4, max_iter=400))])
    m_a = Pipeline([("pre", pre), ("reg", PoissonRegressor(alpha=1e-4, max_iter=400))])

    m_h.fit(X, y_h)
    m_a.fit(X, y_a)
    return m_h, m_a

def predict(models: FitModels, home: str, away: str) -> dict:
    X = pd.DataFrame([{"HomeTeam": home, "AwayTeam": away}])
    lh = float(models.home.predict(X)[0])
    la = float(models.away.predict(X)[0])

    Rh = models.elo.get(home, 1500.0)
    Ra = models.elo.get(away, 1500.0)
    elo_diff = (Rh + HOME_ADV_ELO) - Ra

    adj = math.exp((elo_diff / 400.0) * ELO_GOAL_SCALER)
    lh = max(0.05, lh * adj)
    la = max(0.05, la / adj)

    ph = np.array([poisson_pmf(i, lh) for i in range(MAX_GOALS + 1)])
    pa = np.array([poisson_pmf(i, la) for i in range(MAX_GOALS + 1)])
    mat = np.outer(ph, pa)

    p_home = float(np.tril(mat, k=-1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, k=1).sum())

    best = np.unravel_index(np.argmax(mat), mat.shape)

    lam_total = lh + la
    p_over25 = 1.0 - sum(poisson_pmf(k, lam_total) for k in [0, 1, 2])
    p_btts = 1.0 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh + la))

    return {
        "xg_home": lh,
        "xg_away": la,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "most_likely_score": f"{best[0]}-{best[1]}",
        "elo_diff": float(elo_diff),
        "p_over25": float(p_over25),
        "p_btts": float(p_btts),
    }
