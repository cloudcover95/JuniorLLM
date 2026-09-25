"""Balanced ternary gates on {-1,0,1}. Classical. Not qutrits."""
from __future__ import annotations

T = (-1, 0, 1)


def clip(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def neg(a: int) -> int:
    return clip(-a)


def tmin(a: int, b: int) -> int:
    return a if a < b else b


def tmax(a: int, b: int) -> int:
    return a if a > b else b


def mul(a: int, b: int) -> int:
    return clip(a * b)


def consensus(a: int, b: int) -> int:
    return a if a == b else 0


def txor(a: int, b: int) -> int:
    return clip(a - b) if a != b else 0


def apply(name: str, xs: list[int]) -> list[int]:
    fn = {"neg": lambda a, b=0: neg(a), "min": tmin, "max": tmax, "mul": mul, "cons": consensus, "xor": txor}[name]
    if name == "neg":
        return [fn(x) for x in xs]
    out = []
    for i in range(0, len(xs) - 1, 2):
        out.append(fn(xs[i], xs[i + 1]))
    if len(xs) % 2:
        out.append(xs[-1])
    return out


def status() -> dict:
    return {"alphabet": list(T), "qubit": False, "qutrit": False, "sti_pti_nti": False}
