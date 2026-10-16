"""OSai deck workflow with the layer stack wired in.

Runs the existing OSai deck port, then plans and runs one trit step through
junior_osai (front -> plan/step). No live bench. No LUT. No GGUF. No model pull.
Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from junior_osai.agent_stack import plan, step
from junior_osai.kernels.front import to_trit
from ports.osai_deck import run as deck_run

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def run(W: np.ndarray | None = None, x: np.ndarray | None = None, task: str = "deck audio") -> dict[str, Any]:
    deck = deck_run(task)
    if W is None or x is None:
        rng = np.random.default_rng(0)
        W = rng.standard_normal((64, 64)).astype(np.float32) * 0.02
        x = rng.standard_normal(64).astype(np.float32)
    T, gamma = to_trit(W)
    p = plan(*T.shape)
    s = step({"T": T, "x": x, "gamma": gamma})
    return {
        **deck,
        "stack": {
            "planned": p["layer"],
            "layer": s["layer"],
            "gamma": gamma,
            "packed_bytes": s["packed_bytes"],
            "n_out": int(s["y"].size),
            "bench": False,
            "lut": False,
        },
        "ports": list(PORTS),
        "model_pull": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
