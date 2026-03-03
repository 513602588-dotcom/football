import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# 标签：主胜/平/客胜
LABELS = ["H", "D", "A"]

@dataclass
class MLModels:
    rf: RandomForestClassifier
    mlp: MLPClassifier
    feature_cols: list

def _outcome(hg: int, ag: int) -> str:
    if hg > ag: return "H"
    if hg == ag: return "D"
    return "A"

def build_team_form_features(df_played: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """
    为每场比赛构造“赛前”特征：近N场进球/失球/胜率/平率/负率（主客分别）
    只用比赛结果即可，不依赖额外数据源，Actions 也能跑。
    """
    df = df_played.sort_values("Date").copy()
    df["y"] = [ _outcome(int(h), int(a)) for h, a in zip(df["FTHG"], df["FTAG"]) ]

    # 统一生成“每支球队每场的 stats 行”
    rows = []
    for _, r in df.iterrows():
        rows.append({"Date": r["Date"], "Team": r["HomeTeam"], "GF": r["FTHG"], "GA": r["FTAG"],
                     "W": 1 if r["FTHG"]>r["FTAG"] else 0, "D": 1 if r["FTHG"]==r["FTAG"] else 0, "L": 1 if r["FTHG"]<r["FTAG"] else 0,
                     "is_home": 1, "Opp": r["AwayTeam"], "match_id": r.name})
        rows.append({"Date": r["Date"], "Team": r["AwayTeam"], "GF": r["FTAG"], "GA": r["FTHG"],
                     "W": 1 if r["FTAG"]>r["FTHG"] else 0, "D": 1 if r["FTAG"]==r["FTHG"] else 0, "L": 1 if r["FTAG"]<r["FTHG"] else 0,
                     "is_home": 0, "Opp": r["HomeTeam"], "match_id": r.name})

    form = pd.DataFrame(rows).sort_values(["Team","Date"]).copy()

    # 滚动窗口统计（shift 1 保证“赛前”）
    for col in ["GF","GA","W","D","L"]:
        form[f"{col}_roll"] = form.groupby("Team")[col].transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())

    # 汇总到 match 级：主队特征 + 客队特征
    home_form = form[form["is_home"]==1][["match_id","GF_roll","GA_roll","W_roll","D_roll","L_roll"]].copy()
    away_form = form[form["is_home"]==0][["match_id","GF_roll","GA_roll","W_roll","D_roll","L_roll"]].copy()

    home_form.columns = ["match_id","h_gf","h_ga","h_w","h_d","h_l"]
    away_form.columns = ["match_id","a_gf","a_ga","a_w","a_d","a_l"]

    out = df.reset_index().rename(columns={"index":"match_id"}).merge(home_form, on="match_id", how="left").merge(away_form, on="match_id", how="left")
    # 差分特征
    out["dgf"] = out["h_gf"] - out["a_gf"]
    out["dga"] = out["h_ga"] - out["a_ga"]
    out["dw"]  = out["h_w"]  - out["a_w"]
    out["dd"]  = out["h_d"]  - out["a_d"]
    out["dl"]  = out["h_l"]  - out["a_l"]
    return out

def train_models(df_played: pd.DataFrame) -> Optional[MLModels]:
    """
    训练 RF + MLP（轻量版），输出可用于 predict_proba 的模型。
    """
    feat_df = build_team_form_features(df_played, window=8)
    feat_df = feat_df.dropna(subset=["h_gf","a_gf","h_ga","a_ga","h_w","a_w"]).copy()
    if len(feat_df) < 800:
        return None

    X = feat_df[["h_gf","h_ga","h_w","h_d","h_l","a_gf","a_ga","a_w","a_d","a_l","dgf","dga","dw","dd","dl"]].values
    y = feat_df["y"].values

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=0.002,
        max_iter=300,
        random_state=42,
    )
    mlp.fit(X, y)

    return MLModels(rf=rf, mlp=mlp, feature_cols=["h_gf","h_ga","h_w","h_d","h_l","a_gf","a_ga","a_w","a_d","a_l","dgf","dga","dw","dd","dl"])

def _features_for_fixture(team_form: Dict[str, dict], home: str, away: str) -> Optional[np.ndarray]:
    """
    team_form: 每队近期统计
    """
    if home not in team_form or away not in team_form:
        return None
    h = team_form[home]; a = team_form[away]
    x = np.array([
        h["gf"], h["ga"], h["w"], h["d"], h["l"],
        a["gf"], a["ga"], a["w"], a["d"], a["l"],
        h["gf"]-a["gf"], h["ga"]-a["ga"], h["w"]-a["w"], h["d"]-a["d"], h["l"]-a["l"],
    ], dtype=float)
    return x

def compute_latest_team_form(df_played: pd.DataFrame, window: int = 8) -> Dict[str, dict]:
    """
    用最新 window 场生成每队的近期均值（给未来赛程用）
    """
    df = df_played.sort_values("Date").copy()
    stats = {}
    for team in pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]])):
        # 抽取该队所有比赛（转为该队视角 GF/GA/W/D/L）
        rows = []
        for _, r in df[(df["HomeTeam"]==team) | (df["AwayTeam"]==team)].tail(window).iterrows():
            if r["HomeTeam"] == team:
                gf, ga = int(r["FTHG"]), int(r["FTAG"])
            else:
                gf, ga = int(r["FTAG"]), int(r["FTHG"])
            rows.append((gf, ga, 1 if gf>ga else 0, 1 if gf==ga else 0, 1 if gf<ga else 0))
        if len(rows) < 3:
            continue
        arr = np.array(rows, dtype=float)
        stats[team] = {
            "gf": float(arr[:,0].mean()),
            "ga": float(arr[:,1].mean()),
            "w":  float(arr[:,2].mean()),
            "d":  float(arr[:,3].mean()),
            "l":  float(arr[:,4].mean()),
        }
    return stats

def predict_proba(models: MLModels, team_form: Dict[str, dict], home: str, away: str) -> Optional[Tuple[float,float,float]]:
    x = _features_for_fixture(team_form, home, away)
    if x is None:
        return None

    X = x.reshape(1, -1)
    # 两个模型各出概率，再平均
    prf = models.rf.predict_proba(X)[0]
    pmlp = models.mlp.predict_proba(X)[0]

    # sklearn 返回的 class 顺序按模型 classes_ 排列
    def to_triplet(p, classes):
        out = {"H":0.0,"D":0.0,"A":0.0}
        for v,c in zip(p, classes):
            out[str(c)] = float(v)
        return out["H"], out["D"], out["A"]

    h1,d1,a1 = to_triplet(prf, models.rf.classes_)
    h2,d2,a2 = to_triplet(pmlp, models.mlp.classes_)

    ph = 0.5*h1 + 0.5*h2
    pd = 0.5*d1 + 0.5*d2
    pa = 0.5*a1 + 0.5*a2

    # 归一化
    s = ph+pd+pa
    if s <= 0: return None
    return ph/s, pd/s, pa/s
