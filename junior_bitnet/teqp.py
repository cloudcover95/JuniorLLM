"""JuniorTeqp — Templated Equation of Quantized Properties.

Flip of NIST teqp (Bell et al.): residual + ideal Helmholtz, reduced
variables, derivatives without handwritten analytic forms.

Domain is a trit vector z in {-1,0,1}^N — FieldCore embed, night mesh,
zkVM register, AbsMean weights — not a fluid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Corresponding-states criticals for *this* stack (AbsMean ~50% zeros; night review T).
RHO_C = 0.5
T_C = 0.22
J_COUPLE = 0.5


def _counts(z: list[int]) -> tuple[int, int, int, int]:
    n = len(z) or 1
    n1 = sum(1 for x in z if x == 1)
    nm = sum(1 for x in z if x == -1)
    n0 = n - n1 - nm
    return n, n1, nm, n0


def rho(z: list[int]) -> float:
    n, n1, nm, _ = _counts(z)
    return (n1 + nm) / n


def mag(z: list[int]) -> float:
    n, n1, nm, _ = _counts(z)
    return (n1 - nm) / n


def _entropy(z: list[int]) -> float:
    n, n1, nm, n0 = _counts(z)
    s = 0.0
    for c in (n1, nm, n0):
        if c:
            p = c / n
            s -= p * math.log(p)
    return s


def _pair(z: list[int]) -> float:
    if len(z) < 2:
        return 0.0
    return sum(z[i] * z[(i + 1) % len(z)] for i in range(len(z))) / len(z)


def a_ideal(z: list[int], T: float) -> float:
    return T * _entropy(z)


def a_residual(z: list[int]) -> float:
    return -J_COUPLE * _pair(z)


def a_helmholtz(z: list[int], T: float) -> float:
    return a_ideal(z, T) + a_residual(z)


def reduce(z: list[int], T: float) -> tuple[float, float]:
    return rho(z) / RHO_C, T / T_C


def fd(fn, x: float, h: float = 1e-4) -> float:
    """Central difference — teqp used AD; we stay stdlib."""
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def ar01(z: list[int], T: float) -> float:
    """Like teqp get_Ar01: rho * d(A_res)/d(rho) via scale of nonzeros."""
    r0 = rho(z)
    if r0 <= 0:
        return 0.0

    def a_at(scale: float) -> float:
        zz = [int(max(-1, min(1, round(v * scale)))) for v in z]
        return a_residual(zz)

    return r0 * fd(a_at, 1.0)


@dataclass
class Props:
    n: int
    rho: float
    mag: float
    T: float
    rho_r: float
    T_r: float
    A: float
    A_id: float
    A_res: float
    Ar01: float
    pressure: float
    phase: str


def phase_of(r: float, T: float) -> str:
    if r < 0.25:
        return "sparse"
    if r > 0.75:
        return "dense"
    if T > T_C:
        return "mixed"
    return "coexist"


def props(z: list[int], T: float = T_C) -> Props:
    r = rho(z)
    rr, tr = reduce(z, T)
    aid = a_ideal(z, T)
    ares = a_residual(z)
    a = aid + ares
    ar = ar01(z, T)
    p = r * T * (1.0 + ar) if r > 0 else 0.0
    return Props(
        len(z),
        round(r, 6),
        round(mag(z), 6),
        T,
        round(rr, 6),
        round(tr, 6),
        round(a, 6),
        round(aid, 6),
        round(ares, 6),
        round(ar, 6),
        round(p, 6),
        phase_of(r, T),
    )
