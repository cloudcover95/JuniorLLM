"""Fallback SIS commit if MemSys package is not on path."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

Q = 8380417
N = 16
M = 32


def _mat(seed: bytes) -> list[list[int]]:
    rows = []
    h = seed
    for _ in range(N):
        row = []
        for _j in range(M):
            h = hashlib.sha256(h).digest()
            row.append(int.from_bytes(h[:4], "big") % Q)
        rows.append(row)
    return rows


def _clamp3(xs: list[int]) -> list[int]:
    out = [(1 if x > 0 else (-1 if x < 0 else 0)) for x in xs[:M]]
    return out + [0] * (M - len(out))


@dataclass(frozen=True)
class Commit:
    seed: bytes
    c: tuple[int, ...]


def commit(z: list[int], seed: bytes = b"junior-msis-v0") -> Commit:
    A = _mat(seed)
    zz = _clamp3(z)
    return Commit(seed, tuple(sum(a * x for a, x in zip(row, zz)) % Q for row in A))
