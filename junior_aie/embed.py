"""Shared local embedding — hashed bag, no hosted encoder."""
from __future__ import annotations

import math
import re

DIM = 64
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def embed(text: str, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = hash(tok) % dim
        sign = -1.0 if (hash(tok + "#") % 2) else 1.0
        vec[h] += sign
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
