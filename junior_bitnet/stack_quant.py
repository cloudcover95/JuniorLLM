"""Quant stats for a stack note. AbsMean + I2_S size vs raw chars."""
from __future__ import annotations

from junior_bitnet.edgepack import edgepack
from junior_bitnet.i2s import pack
from junior_bitnet.math import absmean


def stats(text: str) -> dict:
    xs = [float(ord(c) % 97) for c in (text or "x")[:64]]
    z, scale = absmean(xs)
    blob = pack(z)
    raw = len(text or "")
    return {
        "sparsity": round(z.count(0) / max(len(z), 1), 3),
        "i2s_bytes": len(blob),
        "raw_chars": raw,
        "ratio": round(len(blob) / max(raw, 1), 3),
        "scale": scale,
        "alphabet": all(t in (-1, 0, 1) for t in z),
        "edgepack": edgepack(text)["i2s"],
    }
