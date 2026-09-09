"""Shared local embedding — stable hashed bag (hashlib, not PYTHONHASHSEED)."""
from __future__ import annotations

import hashlib
import math
import re

DIM = 64
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _h(tok: str) -> int:
    return int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)


def embed(text: str, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in tokenize(text):
        hv = _h(tok)
        idx = hv % dim
        sign = -1.0 if (hv >> 8) & 1 else 1.0
        vec[idx] += sign
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
