"""Pack ternary z into 2 bits per coeff."""
from __future__ import annotations


def pack(z: list[int]) -> bytes:
    bits = 0
    acc = 0
    out = bytearray()
    for x in z:
        v = 0 if x < 0 else (1 if x == 0 else 2)
        acc |= v << bits
        bits += 2
        if bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack(buf: bytes, n: int) -> list[int]:
    z = []
    vals = []
    for b in buf:
        vals.extend([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3])
    for v in vals[:n]:
        z.append(-1 if v == 0 else (0 if v == 1 else 1))
    return z
