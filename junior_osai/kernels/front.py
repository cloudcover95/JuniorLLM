"""Trit pack front. Float weights → ternary before the stack. No bench. No LUT."""
from __future__ import annotations
from typing import Any
import numpy as np
from .stack import pick, run

def to_trit(W: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.asarray(W, dtype=np.float32)
    g = float(np.mean(np.abs(a))) or 1.0
    t = np.clip(np.rint(a / g), -1, 1).astype(np.int8)
    return t, g

def front(W: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    T, g = to_trit(W)
    out = run(T, x, g)
    return {"layer": out["layer"], "gamma": g, "y": out["y"], "packed_bytes": out["packed_bytes"], "lut": False}

if __name__ == "__main__":
    import json
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 64)).astype(np.float32) * 0.02
    x = rng.standard_normal(64).astype(np.float32)
    out = front(W, x)
    print(json.dumps({"layer": out["layer"], "gamma": round(out["gamma"], 6)}, indent=2))
