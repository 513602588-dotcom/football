import json
from pathlib import Path

# 计算市场隐含概率
def implied_prob(odds):
    try:
        return 1/float(odds) if odds and float(odds) > 0 else 0
    except:
        return 0

# 简单模型：市场 + Poisson
def model_prob(m):
    # 从赔率生成市场概率
    ph = implied_prob(m.get("odds_win"))
    pd = implied_prob(m.get("odds_draw"))
    pa = implied_prob(m.get("odds_lose"))
    s = ph+pd+pa
    if s>0:
        ph, pd, pa = ph/s, pd/s, pa/s
    # Poisson简单预设xG 用市场偏度
    diff = ph - pa
    home_xg = 1.35 + diff*0.5
    away_xg = 1.10 - diff*0.5
    # Poisson outcome
    import math
    def poisson(l,k): return (l**k)*math.exp(-l)/math.factorial(k)
    maxg=6
    hp=pd=ap=0
    for h in range(maxg):
        for a in range(maxg):
            p = poisson(home_xg,h)*poisson(away_xg,a)
            if h>a: hp+=p
            elif h==a: pd+=p
            else: ap+=p
    mp = {"H":hp,"D":pd,"A":ap}
    # normalize
    tot = hp+pd+ap
    if tot>0:
        mp={"H":hp/tot,"D":pd/tot,"A":ap/tot}
    return mp

# Kelly 资金比例
def kelly(p, odds):
    try:
        o = float(odds)
        if o<=1: return 0
        return max((p*(o-1)-(1-p))/(o-1),0)
    except:
        return 0

# load jczq
jczq = json.loads(Path("site/data/jczq.json").read_text(encoding="utf-8"))
matches = jczq.get("matches",[])

picks=[]
for m in matches:
    mp = model_prob(m)
    ph, pd, pa = mp["H"], mp["D"], mp["A"]
    # EV 只对主胜算
    ev_home = ph*float(m.get("odds_win") or 0) - 1 if m.get("odds_win") else 0
    kelly_home = kelly(ph, m.get("odds_win") or 0)
    # 只选 EV>0 情况
    if ev_home>0:
        picks.append({
            "date": m.get("date"),
            "home": m.get("home"),
            "away": m.get("away"),
            "odds": {
                "H": m.get("odds_win"),
                "D": m.get("odds_draw"),
                "A": m.get("odds_lose"),
            },
            "prob": mp,
            "ev_home": round(ev_home,3),
            "kelly_home": round(kelly_home,3)
        })

# write output
outP = Path("site/data/picks.json")
outP.write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")
print("Generated picks.json:",len(picks),"picks")
