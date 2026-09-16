"""Bit Hamming + normalized distance on I2_S hex."""
from __future__ import annotations

from ports.gaia_proto import handshake
from ports.ham_pq import ham


def hex_of(note: str) -> str:
    env = handshake(note, job="dash-viewport")
    return str((env.get("note") or {}).get("i2s_hex") or "")


def metric(a: str, b: str) -> dict:
    ha, hb = hex_of(a), hex_of(b)
    d = ham(ha, hb)
    nbits = max(len(ha), len(hb)) * 4
    return {
        "d_bits": d,
        "nbits": nbits,
        "d_norm": round(d / nbits, 6) if nbits else 1.0,
        "same_pack": ha == hb and bool(ha),
        "a": a,
        "b": b,
    }
