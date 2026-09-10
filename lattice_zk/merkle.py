"""Polynomial-ish commitment = Merkle over coefficient bytes."""
from __future__ import annotations

import hashlib


def _h(a: bytes, b: bytes | None = None) -> bytes:
    x = a if b is None else a + b
    return hashlib.sha256(x).digest()


def root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return _h(b"empty")
    layer = [_h(x) for x in leaves]
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer), 2):
            r = layer[i + 1] if i + 1 < len(layer) else layer[i]
            nxt.append(_h(layer[i], r))
        layer = nxt
    return layer[0]
