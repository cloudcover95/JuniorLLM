"""Layer stack. Picks kernel by size. No bench. No LUT. No GGUF."""
from __future__ import annotations
from typing import Any
import numpy as np
from .bitplane import bitplane_matvec, pack2

def pick(m: int, k: int) -> str:
    n = m * k
    if n < 512 * 512:
        return "t0_python"
    if n < 1024 * 1024:
        return "t1_bitplane"
    return "t2_native"

def run(T: np.ndarray, x: np.ndarray, gamma: float = 1.0) -> dict[str, Any]:
    m, k = T.shape
    layer = pick(m, k)
    if layer == "t0_python":
        y = (T.astype(np.float32) * gamma) @ x
    elif layer == "t1_bitplane":
        y = bitplane_matvec(T, x, gamma)
    else:
        try:
            from rails.linux.bitplane_ffi import bitplane as native
            y = native(T, x, gamma)
            if y is None:
                y = bitplane_matvec(T, x, gamma)
                layer = "t1_bitplane_fallback"
        except Exception:
            y = bitplane_matvec(T, x, gamma)
            layer = "t1_bitplane_fallback"
    return {"layer": layer, "y": y, "packed_bytes": int(pack2(T).nbytes), "lut": False}

if __name__ == "__main__":
    import json
    T = np.random.default_rng(0).choice([-1, 0, 1], size=(256, 256)).astype(np.int8)
    x = np.random.default_rng(1).standard_normal(256).astype(np.float32)
    out = run(T, x, 0.02)
    print(json.dumps({"layer": out["layer"], "n": int(out["y"].size)}, indent=2))
