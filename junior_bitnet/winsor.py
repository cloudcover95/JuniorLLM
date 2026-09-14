"""Winsor AbsMean → trit {-1,0,1}. Stdlib only. MLX optional later."""
from __future__ import annotations

from math import floor


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, floor((p / 100.0) * (len(s) - 1))))
    return s[i]


def pack(weights: list[float], percentile: float = 95.0) -> dict:
    abs_w = [abs(float(w)) for w in weights]
    tau = _pct(abs_w, percentile)
    winsor = [min(a, tau) for a in abs_w]
    gamma = sum(winsor) / max(1, len(winsor))
    g = gamma + 1e-7
    trit = [max(-1, min(1, int(round(float(w) / g)))) for w in weights]
    zeros = sum(1 for t in trit if t == 0)
    return {
        "n": len(trit),
        "tau": tau,
        "gamma": gamma,
        "sparsity": zeros / max(1, len(trit)),
        "trit": trit,
        "ok": True,
        "backend": "stdlib",
    }
