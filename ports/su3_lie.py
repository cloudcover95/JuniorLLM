"""su(3) commutator sample. [\u03bb1,\u03bb2]=2i \u03bb3."""
from __future__ import annotations

from ports.gell_mann import LAMBDAS, _mul, proofs


def _sub(a, b):
    return tuple(tuple(a[i][j] - b[i][j] for j in range(3)) for i in range(3))


def commutator(i: int, j: int):
    return _sub(_mul(LAMBDAS[i], LAMBDAS[j]), _mul(LAMBDAS[j], LAMBDAS[i]))


def su2_ok(tol: float = 1e-9) -> bool:
    c = commutator(1, 2)
    want = tuple(tuple(2j * LAMBDAS[3][i][j] for j in range(3)) for i in range(3))
    return all(abs(c[i][j] - want[i][j]) < tol for i in range(3) for j in range(3))


def lie_proofs() -> dict:
    p = proofs()
    p["su2_subalgebra"] = su2_ok()
    p["structure_full"] = False
    return p
