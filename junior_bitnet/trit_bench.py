"""Compare AbsMean vs Winsor p90/p95/p99. Reconstruction L1 / n."""
from __future__ import annotations

import math
import time

from junior_bitnet.math import absmean
from junior_bitnet.winsor import pack


def _signal(n: int = 256) -> list[float]:
    return [math.sin(i * 0.17) * 2.0 + ((i * 13) % 7 - 3) * 0.4 for i in range(n)]


def _l1(xs: list[float], trit: list[int], g: float) -> float:
    return sum(abs(x - t * g) for x, t in zip(xs, trit)) / max(1, len(xs))


def run(n: int = 256) -> dict:
    xs = _signal(n)
    rows = []
    t0 = time.perf_counter()
    t, g = absmean(xs)
    rows.append({"kind": "absmean", "gamma": g, "l1": _l1(xs, t, g), "zeros": t.count(0) / n})
    for p in (90.0, 95.0, 99.0):
        q = pack(xs, p)
        rows.append({"kind": f"winsor-p{int(p)}", "gamma": q["gamma"], "l1": _l1(xs, q["trit"], q["gamma"] + 1e-7), "zeros": q["sparsity"]})
    return {"n": n, "ms": round((time.perf_counter() - t0) * 1000, 3), "rows": rows, "pick": "winsor-p95"}
