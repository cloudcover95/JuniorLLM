"""48x48 retain-k. numpy.linalg.svd when present; else power energy."""
from __future__ import annotations

import math
import time


def _mat(n: int = 48, seed: int = 7) -> list[list[float]]:
    a = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(math.sin((i + 1) * (j + seed) * 0.11) * 0.7 + 0.05 * ((i * 13 + j) % 5))
        a.append(row)
    return a


def _energy_numpy(n: int, k: int) -> dict:
    import numpy as np

    a = np.array(_mat(n), dtype=float)
    t0 = time.perf_counter()
    _, s, _ = np.linalg.svd(a, full_matrices=False)
    ms = (time.perf_counter() - t0) * 1000
    tot = float((s * s).sum()) or 1.0
    kept = float((s[:k] * s[:k]).sum())
    return {"backend": "numpy", "n": n, "k": k, "energy": round(kept / tot, 4), "ms": round(ms, 3), "s0": float(s[0])}


def _energy_power(n: int, k: int) -> dict:
    a = _mat(n)
    t0 = time.perf_counter()
    tot = sum(x * x for row in a for x in row) or 1.0
    kept = 0.0
    work = [row[:] for row in a]
    for _ in range(min(k, n)):
        v = [1.0] * n
        for _it in range(8):
            w = [sum(work[i][j] * v[j] for j in range(n)) for i in range(n)]
            nrm = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / nrm for x in w]
        lam = sum(v[i] * sum(work[i][j] * v[j] for j in range(n)) for i in range(n))
        kept += lam * lam
        for i in range(n):
            for j in range(n):
                work[i][j] -= lam * v[i] * v[j]
    ms = (time.perf_counter() - t0) * 1000
    return {"backend": "power", "n": n, "k": k, "energy": round(min(1.0, kept / tot), 4), "ms": round(ms, 3)}


def tick(n: int = 48, k: int = 30) -> dict:
    try:
        return _energy_numpy(n, k)
    except Exception:
        return _energy_power(min(n, 24), min(k, 8))
