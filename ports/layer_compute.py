"""Layered ternary compute for inference. Local only.

T0: trit pack score
T1: absmean-style energy
T2: combined score for envelope
"""
from __future__ import annotations
from ports.trit_score import pack_trit, trit_reward

def absmean(trits: list[int]) -> float:
    """Mean absolute value. 0.0 to 1.0."""
    if not trits:
        return 0.0
    return sum(abs(t) for t in trits) / len(trits)

def layer_score(note: str) -> dict:
    trits = pack_trit(note)
    t_score, passed, detail = trit_reward(note)
    energy = absmean(trits)
    combined = (t_score + energy) / 2
    layer = "T0"
    if combined >= 0.6 and len(note) < 128:
        layer = "T0"
    elif combined >= 0.4:
        layer = "T1"
    else:
        layer = "T2"
    return {
        "trit_score": t_score,
        "energy": energy,
        "combined": combined,
        "layer": layer,
        "passed": passed,
        "detail": detail,
        "local": True,
    }
