"""I2_S pack: two bits per trit. 0	o0, 1	o1, -1	o2."""
from __future__ import annotations

from junior_bitnet.math import absmean


def pack(trits: list[int]) -> bytes:
    acc = 0
    n = 0
    out = bytearray()
    for t in trits:
        c = 0 if t == 0 else (1 if t == 1 else 2)
        acc = (acc << 2) | c
        n += 2
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def unpack(blob: bytes, count: int) -> list[int]:
    bits = []
    for b in blob:
        for shift in (6, 4, 2, 0):
            bits.append((b >> shift) & 3)
    inv = {0: 0, 1: 1, 2: -1}
    return [inv.get(bits[i], 0) for i in range(count)]


def pack_floats(xs: list[float]) -> tuple[bytes, int, float]:
    z, scale = absmean(xs)
    return pack(z), len(z), scale
