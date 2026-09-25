"""Rotations. Cyclic permute is not SU(3). Ry on levels 0,1 is a real 2-level embed."""
from __future__ import annotations

import math


def cyclic(amp: tuple[float, float, float], k: int = 1) -> tuple[float, float, float]:
    k %= 3
    return amp[-k:] + amp[:-k] if k else amp


def ry01(amp: tuple[float, float, float], theta: float) -> tuple[float, float, float]:
    a0, a1, a2 = amp
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return (c * a0 - s * a1, s * a0 + c * a1, a2)


def norm2(amp: tuple[float, ...]) -> float:
    return sum(x * x for x in amp)


def demo(theta: float = 1.5707963267948966) -> dict:
    psi = (1.0, 0.0, 0.0)
    rot = ry01(psi, theta)
    return {
        "psi": psi,
        "ry01_pi2": rot,
        "n2": round(norm2(rot), 6),
        "cyclic_of_e0": cyclic(psi, 1),
        "su3": False,
        "complex_phase": False,
    }
