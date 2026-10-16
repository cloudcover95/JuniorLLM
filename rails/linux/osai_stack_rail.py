"""Linux rail for the OSai stack prod. Loopback only. No bench. No LUT. No GGUF. No model pull.

Runs scripts/osai_stack_prod.py, then walks one shape per tier (T0/T1/T2) through
plan/step and reports whether the native bit-plane .so is present on this rail.
Usage: python3 -m rails.linux.osai_stack_rail
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from junior_osai.agent_stack import plan, step  # noqa: E402
from junior_osai.kernels.front import to_trit  # noqa: E402

SHAPES = ((64, 64), (600, 600), (1024, 1024))


def prod() -> dict:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "osai_stack_prod.py")],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads(out.stdout)


def tiers() -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for m, k in SHAPES:
        W = rng.standard_normal((m, k)).astype(np.float32) * 0.02
        x = rng.standard_normal(k).astype(np.float32)
        T, g = to_trit(W)
        s = step({"T": T, "x": x, "gamma": g})
        ref = (T.astype(np.float32) * g) @ x
        rows.append({
            "m": m, "k": k,
            "planned": plan(m, k)["layer"],
            "layer": s["layer"],
            "ok": bool(np.allclose(s["y"], ref, atol=1e-3)),
        })
    return rows


def run() -> dict:
    native = (Path(__file__).with_name("libjunior_bitplane.so")).is_file()
    rows = tiers()
    return {
        "rail": "linux",
        "prod": prod(),
        "native_so": native,
        "tiers": rows,
        "ok": all(r["ok"] for r in rows),
        "bind": "127.0.0.1",
        "bench": False,
        "lut": False,
        "model_pull": False,
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
