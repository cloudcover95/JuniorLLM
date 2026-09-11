"""Quant maps. No autograd.

AbsMean (1.58):  W_q = clip(round(W / mean(|W|)), -1, 1)
AbsMax act:      X_q = clip(round(X * 127 / max(|X|)), -127, 127)
Sign-binarize:   W_b = +1 if W>=0 else -1   (no zero; 1-bit)
"""
from __future__ import annotations


def clip_round(x: float, lo: int, hi: int) -> int:
    q = int(round(x))
    return lo if q < lo else (hi if q > hi else q)


def mean_abs(xs: list[float]) -> float:
    return (sum(abs(x) for x in xs) / len(xs)) if xs else 1.0


def absmean(xs: list[float]) -> tuple[list[int], float]:
    d = mean_abs(xs) or 1.0
    return [clip_round(x / d, -1, 1) for x in xs], d


def absmax_act(xs: list[float]) -> tuple[list[int], float]:
    am = max((abs(x) for x in xs), default=1.0) or 1.0
    s = 127.0 / am
    return [clip_round(x * s, -127, 127) for x in xs], s


def binarize(xs: list[float]) -> list[int]:
    return [1 if x >= 0 else -1 for x in xs]


def sparsity(trits: list[int]) -> float:
    return trits.count(0) / len(trits) if trits else 1.0
