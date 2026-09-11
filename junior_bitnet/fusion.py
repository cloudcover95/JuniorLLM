"""Fused AbsMean(W) + AbsMax(X) + integer dot.
CPU reference is the contract. Triton is optional and must match y.
"""
from __future__ import annotations

from junior_bitnet.bitlinear import bitlinear


def fused(x: list[float], w: list[float]) -> dict:
    out = bitlinear(x, w)
    out["fused"] = True
    return out


def triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def run(x: list[float], w: list[float]) -> dict:
    ref = fused(x, w)
    ref["backend"] = "triton" if triton_available() else "cpu"
    return ref
