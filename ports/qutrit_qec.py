"""3-symbol majority. Toy. Not a qutrit stabilizer code."""
from __future__ import annotations

from collections import Counter


def encode(u: int) -> tuple[int, int, int]:
    u = int(u) % 3
    return (u, u, u)


def decode(trip: tuple[int, int, int]) -> dict:
    c = Counter(trip)
    sym, n = c.most_common(1)[0]
    return {"sym": sym, "votes": n, "corrected": n >= 2, "code": "rep3", "steane": False}
