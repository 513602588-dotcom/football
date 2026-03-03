from dataclasses import dataclass
from typing import Tuple, Optional

def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds and odds > 1.0 else 0.0

def remove_overround(p1: float, p2: float, p3: float) -> Tuple[float,float,float]:
    s = p1 + p2 + p3
    if s <= 0:
        return 0.0, 0.0, 0.0
    return p1/s, p2/s, p3/s

@dataclass
class BestBet:
    pick: str
    p_model: float
    odds: float
    p_fair: float
    ev: float
    kelly: float
    gap: float

def calc(p_model: float, odds: float, p_fair: float, pick: str) -> BestBet:
    ev = p_model * odds - 1.0
    b = odds - 1.0
    q = 1.0 - p_model
    k = (b*p_model - q) / b if b > 0 else 0.0
    k = max(0.0, k)
    return BestBet(pick, p_model, odds, p_fair, ev, k, p_model - p_fair)

def score(best: BestBet) -> float:
    # 100分：EV 55 + 概率 25 + 稳定性 20
    s = 0.0
    s += max(0.0, min(55.0, best.ev * 120.0))
    s += max(0.0, min(25.0, best.p_model * 25.0))
    s += max(0.0, min(20.0, best.p_model * 20.0))
    return round(s, 2)

def label(s: Optional[float]) -> str:
    if s is None:
        return "-"
    if s >= 80: return "🔥 强推"
    if s >= 65: return "⭐ 主推"
    if s >= 50: return "✔ 可博"
    return "❌ 放弃"
