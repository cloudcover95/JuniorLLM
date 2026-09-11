"""Tensor parallel for 1.58-bit BitLinear.
Shard W_q columns. Each worker: integer dot. Reduce acc, then * Δ.
One box, N logical workers. Not NCCL.
"""
from __future__ import annotations

from junior_bitnet.bitlinear import bitlinear, dot
from junior_bitnet.math import absmax_act, absmean


def shard(xs: list, n: int) -> list[list]:
    n = max(1, n)
    out = [[] for _ in range(n)]
    for i, v in enumerate(xs):
        out[i % n].append(v)
    return out


def tp_bitlinear(x: list[float], w: list[float], workers: int = 2) -> dict:
    ref = bitlinear(x, w)
    wq, dw = absmean(w)
    xq, dx = absmax_act(x)
    n = min(len(xq), len(wq))
    xq, wq = xq[:n], wq[:n]
    acc = 0
    for xs, ws in zip(shard(xq, workers), shard(wq, workers)):
        acc += dot(xs, ws)
    y = acc * (dw / 127.0)
    return {"y": y, "acc": acc, "workers": workers, "match": abs(y - ref["y"]) < 1e-9, "ref": ref["y"]}
