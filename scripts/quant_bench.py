#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptations.omega_cad.draft import interpret
from adaptations.omega_cad.interpolate import interpolate
from adaptations.omega_cad.quant import absmean, i2s_pack, sign

XS = [40.0, 20.0, 8.0, 6.0, 0.2, -3.0] * 20


def _us(fn, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def main() -> dict:
    t, sc = absmean(XS)
    out = {
        "absmean_us": round(_us(lambda: absmean(XS), 400), 3),
        "sign_us": round(_us(lambda: sign(XS), 400), 3),
        "i2s_us": round(_us(lambda: i2s_pack(t), 400), 3),
        "interp_explicit_us": round(_us(lambda: interpolate("40 20", "6", "HEIGHT: 8"), 80), 3),
        "interp_misc_us": round(_us(lambda: interpolate("40 20", "6 6", "TITLE"), 40), 3),
        "interpret_us": round(_us(lambda: interpret("TITLE ELEV HEIGHT: 8"), 20), 3),
        "sparsity": t.count(0) / len(t),
        "scale": sc,
        "packed_bytes": len(i2s_pack(t)),
        "finding": "explicit HEIGHT skips IQ; sign ~8x absmean; interp cost is IQ not quant",
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
