"""Bit-plane matvec on already-ternary weights. No LUT. No GGUF."""
from __future__ import annotations
import time
from typing import Any
import numpy as np

def pack2(t: np.ndarray) -> np.ndarray:
    """t in {-1,0,1} -> 2-bit codes {0,1,2}, 4 per byte."""
    c = (t.astype(np.int8) + 1).astype(np.uint8)
    flat = c.reshape(-1)
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    packed = flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    return packed

def bitplane_matvec(T: np.ndarray, x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """T is int8 {-1,0,1}. y = gamma * (pos @ x - neg @ x)."""
    pos = (T == 1)
    neg = (T == -1)
    return gamma * (pos.astype(np.float32) @ x - neg.astype(np.float32) @ x)

def bench(m: int = 512, k: int = 512, repeats: int = 5, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    T = rng.choice([-1, 0, 1], size=(m, k)).astype(np.int8)
    x = rng.standard_normal(k).astype(np.float32)
    gamma = 0.02
    ref = (T.astype(np.float32) * gamma) @ x
    def run(fn):
        ts = []
        for _ in range(repeats):
            a = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - a)
        return float(np.median(ts)) * 1e3
    fp32_ms = run(lambda: (T.astype(np.float32) * gamma) @ x)
    bp_ms = run(lambda: bitplane_matvec(T, x, gamma))
    y = bitplane_matvec(T, x, gamma)
    err = float(np.linalg.norm(y - ref) / (np.linalg.norm(ref) + 1e-12))
    packed = pack2(T)
    return {
        "m": m, "k": k,
        "fp32_ms": round(fp32_ms, 3),
        "bitplane_ms": round(bp_ms, 3),
        "speed_vs_fp32": round(fp32_ms / bp_ms, 3) if bp_ms else 0.0,
        "rel_err": round(err, 6),
        "packed_bytes": int(packed.nbytes),
        "fp32_bytes": int(T.astype(np.float32).nbytes),
        "mem_vs_fp32": round(T.astype(np.float32).nbytes / packed.nbytes, 2),
        "viable": bool(bp_ms > 0 and fp32_ms / bp_ms >= 1.5 and err <= 0.01),
        "lut": False,
    }

if __name__ == "__main__":
    import json
    print(json.dumps(bench(), indent=2))
