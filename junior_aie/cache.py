"""Semantic cache — embedding similarity + hit-rate."""
from __future__ import annotations

from dataclasses import dataclass, field

from junior_aie.embed import cosine, embed


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0


class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self.store: list[tuple[list[float], str, str]] = []
        self.stats = CacheStats()

    def get(self, query: str) -> str | None:
        qv = embed(query)
        best: tuple[float, str] | None = None
        for vec, _q, val in self.store:
            s = cosine(qv, vec)
            if best is None or s > best[0]:
                best = (s, val)
        if best and best[0] >= self.threshold:
            self.stats.hits += 1
            return best[1]
        self.stats.misses += 1
        return None

    def put(self, query: str, value: str) -> None:
        self.store.append((embed(query), query, value))
