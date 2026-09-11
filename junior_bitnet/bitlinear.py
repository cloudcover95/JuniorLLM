"""y ≈ (X_q @ W_q) * (Δ_x * Δ_w) with ternary W and int8-ish X."""
from __future__ import annotations

from junior_bitnet.math import absmax_act, absmean


def dot(a: list[int], b: list[int]) -> int:
    return sum(x * y for x, y in zip(a, b))


def bitlinear(x: list[float], w: list[float]) -> dict:
    wq, dw = absmean(w)
    xq, dx = absmax_act(x)
    n = min(len(xq), len(wq))
    acc = dot(xq[:n], wq[:n])
    y = acc * (dw / 127.0)
    return {"y": y, "acc": acc, "dw": dw, "dx": dx, "wq": wq, "xq": xq[:n]}
