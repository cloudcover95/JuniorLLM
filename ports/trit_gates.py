"""Classical balanced-ternary gates. Not qutrits."""
from __future__ import annotations

T = (-1, 0, 1)


def _t(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def neg(a: int) -> int:
    return -_t(a)


def tmin(a: int, b: int) -> int:
    return min(_t(a), _t(b))


def tmax(a: int, b: int) -> int:
    return max(_t(a), _t(b))


def mul(a: int, b: int) -> int:
    return _t(a) * _t(b)


def consensus(a: int, b: int) -> int:
    a, b = _t(a), _t(b)
    if a == b:
        return a
    return 0


def cycle(a: int, step: int = 1) -> int:
    return T[(_t(a) + 1 + step) % 3]
