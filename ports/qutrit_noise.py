"""Classical noise on {0,1,2}. Not a Lindblad solver."""
from __future__ import annotations

import random


def flip(u: int, p: float, rng: random.Random | None = None) -> int:
    r = rng or random.Random(0)
    if r.random() >= p:
        return int(u) % 3
    return r.choice([x for x in (0, 1, 2) if x != int(u) % 3])


def depolarize(u: int, p: float, rng: random.Random | None = None) -> int:
    r = rng or random.Random(0)
    if r.random() >= p:
        return int(u) % 3
    return r.choice((0, 1, 2))


def damp(u: int, p: float, rng: random.Random | None = None) -> int:
    """Toward |0>."""
    r = rng or random.Random(0)
    if int(u) % 3 == 0 or r.random() >= p:
        return int(u) % 3
    return 0


def thread(seq: list[int], p: float = 0.1, kind: str = "flip") -> dict:
    fn = {"flip": flip, "dep": depolarize, "damp": damp}[kind]
    rng = random.Random(7)
    out = [fn(s, p, rng) for s in seq]
    flips = sum(a != b for a, b in zip(seq, out))
    return {"kind": kind, "p": p, "in": seq, "out": out, "flips": flips, "lindblad": False}
