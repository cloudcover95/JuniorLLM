#!/usr/bin/env python3
"""OSai stack one-shot. No bench. No LUT. No GGUF."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from junior_osai.agent_stack import plan, step
from junior_osai.kernels.front import to_trit
import numpy as np

def main():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 64)).astype(np.float32) * 0.02
    x = rng.standard_normal(64).astype(np.float32)
    T, g = to_trit(W)
    p = plan(*T.shape)
    s = step({"T": T, "x": x, "gamma": g})
    print(json.dumps({"planned": p["layer"], "layer": s["layer"], "n": int(s["y"].size), "bench": False}, indent=2))

if __name__ == "__main__":
    main()
