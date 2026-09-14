"""Retain-k. Dense numpy SVD is opt-in. Handshake must never call this."""
from __future__ import annotations

import math
import os
import time
from typing import Any

from ports.terrain_spine import flagstaff_mesh


def _energy_numpy(mesh: list[list[float]], k: int) -> dict[str, Any]:
    import numpy as np

    a = np.asarray(mesh, dtype=float)
    t0 = time.perf_counter()
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    approx = (u[:, :k] * s[:k]) @ vt[:k, :]
    ms = (time.perf_counter() - t0) * 1000
    tot = float((s * s).sum()) or 1.0
    kept = float((s[:k] * s[:k]).sum())
    return {"backend": "numpy", "n": int(a.shape[0]), "k": k, "energy": round(kept / tot, 4), "ms": round(ms, 3), "mesh": approx.tolist()}


def _energy_power(mesh: list[list[float]], k: int) -> dict[str, Any]:
    n = len(mesh)
    kk = min(k, 8, n)
    work = [row[:] for row in mesh]
    t0 = time.perf_counter()
    tot = sum(x * x for row in work for x in row) or 1.0
    kept = 0.0
    for _ in range(kk):
        v = [1.0] * n
        for _it in range(6):
            w = [sum(work[i][j] * v[j] for j in range(n)) for i in range(n)]
            nrm = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / nrm for x in w]
        lam = sum(v[i] * sum(work[i][j] * v[j] for j in range(n)) for i in range(n))
        kept += lam * lam
        for i in range(n):
            for j in range(n):
                work[i][j] -= lam * v[i] * v[j]
    ms = (time.perf_counter() - t0) * 1000
    return {"backend": "power", "n": n, "k": kk, "energy": round(min(1.0, kept / tot), 4), "ms": round(ms, 3), "mesh": work}


def tick(n: int = 32, k: int = 8, mesh: list[list[float]] | None = None, mode: str | None = None) -> dict[str, Any]:
    src = mesh or flagstaff_mesh(min(n, 48))
    mode = mode or os.environ.get("JUNIOR_SVD", "auto")
    if mode == "full" or (mode == "auto" and n <= 32):
        try:
            return _energy_numpy(src, k)
        except Exception:
            pass
    return _energy_power([row[:n] for row in src[:n]], k)
