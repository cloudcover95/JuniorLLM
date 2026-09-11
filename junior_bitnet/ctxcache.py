"""Trit-keyed context cache. Key is I2_S of the ask, not raw text."""
from __future__ import annotations

from junior_bitnet.i2s import pack
from junior_bitnet.math import absmean


def key(ask: str) -> str:
    xs = [float(ord(c) % 97) for c in (ask or "x")[:64]]
    z, _ = absmean(xs)
    return pack(z).hex()


class Cache:
    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self.hits = 0
        self.miss = 0

    def get(self, ask: str) -> dict | None:
        k = key(ask)
        if k in self.store:
            self.hits += 1
            return self.store[k]
        self.miss += 1
        return None

    def put(self, ask: str, pack_: dict) -> dict:
        self.store[key(ask)] = pack_
        return pack_
