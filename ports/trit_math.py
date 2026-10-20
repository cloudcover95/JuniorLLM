"""Efficient trit math for agent workflows. Local only.

Caches the pack and absmean so goldend, terraform, and OSai share one computation.
"""
from __future__ import annotations
from functools import lru_cache
from ports.trit_score import pack_trit, trit_reward

@lru_cache(maxsize=256)
def cached_pack(note: str) -> tuple[int, ...]:
    return tuple(pack_trit(note))

@lru_cache(maxsize=256)
def cached_absmean(note: str) -> float:
    trits = cached_pack(note)
    if not trits:
        return 0.0
    return sum(abs(t) for t in trits) / len(trits)

@lru_cache(maxsize=256)
def cached_score(note: str) -> tuple[float, bool, str]:
    return trit_reward(note)

def clear() -> None:
    cached_pack.cache_clear()
    cached_absmean.cache_clear()
    cached_score.cache_clear()

def once(note: str) -> dict:
    """One call. Agents reuse this instead of recomputing."""
    score, passed, detail = cached_score(note)
    return {
        "pack_n": len(cached_pack(note)),
        "absmean": cached_absmean(note),
        "score": score,
        "passed": passed,
        "detail": detail,
        "local": True,
    }
