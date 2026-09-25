"""Center or delta a numeric series then pack. No numpy."""
from __future__ import annotations


def center(xs: list[float]) -> list[float]:
    if not xs:
        return []
    mu = sum(xs) / len(xs)
    return [x - mu for x in xs]


def delta(xs: list[float]) -> list[float]:
    if not xs:
        return []
    out = [0.0]
    for i in range(1, len(xs)):
        out.append(xs[i] - xs[i - 1])
    return out
