"""Eight Gell-Mann generators. Proofs = algebra checks. Not a fridge."""
from __future__ import annotations

import math
from typing import Sequence

C = complex
Mat = tuple[tuple[C, C, C], tuple[C, C, C], tuple[C, C, C]]
Vec = tuple[C, C, C]

I: Mat = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

LAMBDAS: dict[int, Mat] = {
    1: ((0, 1, 0), (1, 0, 0), (0, 0, 0)),
    2: ((0, -1j, 0), (1j, 0, 0), (0, 0, 0)),
    3: ((1, 0, 0), (0, -1, 0), (0, 0, 0)),
    4: ((0, 0, 1), (0, 0, 0), (1, 0, 0)),
    5: ((0, 0, -1j), (0, 0, 0), (1j, 0, 0)),
    6: ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
    7: ((0, 0, 0), (0, 0, -1j), (0, 1j, 0)),
    8: (
        (1 / math.sqrt(3), 0, 0),
        (0, 1 / math.sqrt(3), 0),
        (0, 0, -2 / math.sqrt(3)),
    ),
}


def _dag(m: Mat) -> Mat:
    return tuple(tuple(m[j][i].conjugate() for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _mul(a: Mat, b: Mat) -> Mat:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def _tr(m: Mat) -> C:
    return m[0][0] + m[1][1] + m[2][2]


def apply(m: Mat, v: Vec) -> Vec:
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def hermite_ok() -> bool:
    return all(
        all(abs(LAMBDAS[k][i][j] - _dag(LAMBDAS[k])[i][j]) < 1e-12 for i in range(3) for j in range(3))
        for k in range(1, 9)
    )


def fubini_ok(tol: float = 1e-9) -> bool:
    """Tr(λi λj) = 2 δij."""
    for i in range(1, 9):
        for j in range(1, 9):
            t = _tr(_mul(LAMBDAS[i], LAMBDAS[j])).real
            want = 2.0 if i == j else 0.0
            if abs(t - want) > tol:
                return False
    return True


def proofs() -> dict:
    return {
        "hermitian": hermite_ok(),
        "tr_lambda_lambda": fubini_ok(),
        "n_gen": 8,
        "su3_hw": False,
        "pulse_hw": False,
    }
