"""Trit pack scoring for RLVR agent learning. Local only.

Packs a note to {-1,0,1}, scores density and stability, returns a verifiable reward.
"""
from __future__ import annotations
from typing import Callable

def pack_trit(text: str, cap: int = 64) -> list[int]:
    """Clip char codes to {-1,0,1}. Short notes only."""
    if not text:
        return []
    return [max(-1, min(1, (ord(c) % 3) - 1)) for c in text[:cap]]

def density(trits: list[int]) -> float:
    """Non-zero fraction. 0.0 to 1.0."""
    if not trits:
        return 0.0
    return sum(1 for t in trits if t != 0) / len(trits)

def stability(trits: list[int]) -> float:
    """1.0 if no long zero runs. Penalize collapse."""
    if not trits:
        return 0.0
    zeros = 0
    max_run = 0
    for t in trits:
        if t == 0:
            zeros += 1
            max_run = max(max_run, zeros)
        else:
            zeros = 0
    if max_run >= len(trits) // 2:
        return 0.0
    return 1.0 - (max_run / len(trits))

def trit_reward(text: str) -> tuple[float, bool, str]:
    """Verifiable score for a packed note. Used as an RLVR check."""
    trits = pack_trit(text)
    if not trits:
        return 0.0, False, "empty"
    d = density(trits)
    s = stability(trits)
    score = (d + s) / 2
    passed = score >= 0.4 and s > 0.0
    detail = f"density={d:.2f} stability={s:.2f} n={len(trits)}"
    return score, passed, detail

def make_trit_check(text: str) -> Callable[[], bool]:
    """Returns a check function for agent_cycle."""
    def _check() -> bool:
        _, passed, _ = trit_reward(text)
        return passed
    return _check
