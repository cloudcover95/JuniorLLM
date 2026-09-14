"""A = U Σ V^T retain-k. Reconstructs mesh for OBJ."""
from __future__ import annotations

import math
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
    return {
        "backend": "numpy",
        "n": int(a.shape[0]),
        "k": k,
        "energy": round(kept / tot, 4),
        "ms": round(ms, 3),
        "mesh": approx.tolist(),
    }


def _energy_power(mesh: list[list[float]], k: int) -> dict[str, Any]:
    n = len(mesh)
    work = [row[:] for row in mesh]
    t0 = time.perf_counter()
    tot = sum(x * x for row in work for x in row) or 1.0
    kept = 0.0
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
    return {
        "backend": "power",
        "n": n,
        "k": k,
        "energy": round(min(1.0, kept / tot), 4),
        "ms": round(ms, 3),
        "mesh": work,
    }


def tick(n: int = 48, k: int = 30, mesh: list[list[float]] | None = None) -> dict[str, Any]:
    src = mesh or flagstaff_mesh(n)
    try:
        return _energy_numpy(src, k)
    except Exception:
        small = [row[:24] for row in src[:24]]
        return _energy_power(small, min(k, 8))
