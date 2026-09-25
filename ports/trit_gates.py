"""Balanced ternary + Kleene / Lukasiewicz. Classical."""
from __future__ import annotations

T = (-1, 0, 1)


def clip(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def neg(a: int) -> int:
    return clip(-int(a))


def tmin(a: int, b: int) -> int:
    return a if a < b else b


def tmax(a: int, b: int) -> int:
    return a if a > b else b


def mul(a: int, b: int) -> int:
    return clip(int(a) * int(b))


def consensus(a: int, b: int) -> int:
    return a if a == b else 0


def kleene_and(a: int, b: int) -> int:
    return tmin(a, b)


def kleene_or(a: int, b: int) -> int:
    return tmax(a, b)


def lukasiewicz_imp(a: int, b: int) -> int:
    # clip(1 - a + b) on {-1,0,1} after shift? use clip(b - a + 1) then clip
    return clip(int(b) - int(a) + 1)


def apply(name: str, xs: list[int]) -> list[int]:
    fns = {
        "neg": None,
        "min": tmin,
        "max": tmax,
        "mul": mul,
        "cons": consensus,
        "kand": kleene_and,
        "kor": kleene_or,
        "limp": lukasiewicz_imp,
    }
    if name == "neg":
        return [neg(x) for x in xs]
    fn = fns[name]
    out = []
    for i in range(0, max(0, len(xs) - 1), 2):
        out.append(fn(xs[i], xs[i + 1]))
    if len(xs) % 2:
        out.append(xs[-1])
    return out


def status() -> dict:
    return {
        "alphabet": list(T),
        "kleene": True,
        "lukasiewicz": True,
        "qubit": False,
        "qutrit": False,
        "cmos": False,
    }
