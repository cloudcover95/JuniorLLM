"""Local I2_S pack for an inbox row. Not a cloud sidecar."""
from __future__ import annotations

from junior_bitnet.i2s import pack
from junior_bitnet.math import absmean


def edgepack(text: str) -> dict:
    xs = [float(ord(c) % 97) for c in (text or "x")[:64]]
    z, scale = absmean(xs)
    return {"i2s": pack(z).hex(), "n": len(z), "scale": scale, "alphabet": all(t in (-1, 0, 1) for t in z), "where": "local"}
