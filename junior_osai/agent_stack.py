"""OSai agent path into the layer stack.

Lets a workflow call pick(m, k) and run a ternary matvec without a live bench.
T0/T1/T2 chosen by size only. No bench. No LUT. No GGUF. No model pull. Local only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from junior_osai.kernels.stack import pick, run

__all__ = ["pick", "plan", "step"]


def plan(m: int, k: int) -> dict[str, Any]:
    """Return the layer the stack would use for an (m, k) weight, without running it."""
    if m <= 0 or k <= 0:
        raise ValueError("m and k must be positive")
    return {"m": int(m), "k": int(k), "layer": pick(m, k), "bench": False, "lut": False}


def step(job: dict[str, Any]) -> dict[str, Any]:
    """Run one OSai workflow step through the stack.

    job keys:
      T      ternary weights (-1/0/1), shape (m, k)
      x      input vector, shape (k,)
      gamma  optional scale, default 1.0
    """
    T = np.asarray(job["T"], dtype=np.int8)
    x = np.asarray(job["x"], dtype=np.float32)
    if T.ndim != 2 or x.shape != (T.shape[1],):
        raise ValueError("T must be (m, k) and x must be (k,)")
    if not np.isin(T, (-1, 0, 1)).all():
        raise ValueError("T must be ternary")
    out = run(T, x, float(job.get("gamma", 1.0)))
    return {
        "layer": out["layer"],
        "planned": pick(*T.shape),
        "y": out["y"],
        "packed_bytes": out["packed_bytes"],
        "bench": False,
        "lut": False,
    }


if __name__ == "__main__":
    import json

    rng = np.random.default_rng(0)
    T = rng.choice([-1, 0, 1], size=(64, 64)).astype(np.int8)
    x = rng.standard_normal(64).astype(np.float32)
    res = step({"T": T, "x": x, "gamma": 0.02})
    ref = (T.astype(np.float32) * 0.02) @ x
    print(json.dumps({
        "plans": [plan(64, 64)["layer"], plan(600, 600)["layer"], plan(2048, 2048)["layer"]],
        "layer": res["layer"],
        "ok": bool(np.allclose(res["y"], ref, atol=1e-4)),
    }))
