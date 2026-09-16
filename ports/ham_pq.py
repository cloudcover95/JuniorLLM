"""Bit Hamming on I2_S + SHA3-256 tag. Not SPHINCS+."""
from __future__ import annotations

import hashlib

from ports.gaia_proto import handshake
from ports.osai_engines import run as engine


def _bits(hx: str) -> str:
    try:
        raw = bytes.fromhex(hx)
    except ValueError:
        raw = (hx or "").encode()
    return "".join(f"{b:08b}" for b in raw)


def ham(a: str, b: str) -> int:
    x, y = _bits(a), _bits(b)
    n = min(len(x), len(y))
    return sum(p != q for p, q in zip(x[:n], y[:n])) + abs(len(x) - len(y))


def tool(note: str, other: str | None = None) -> dict:
    env = handshake(note, job="dash-viewport")
    hx = str((env.get("note") or {}).get("i2s_hex") or "")
    digest = hashlib.sha3_256((note or "").encode()).hexdigest()
    wrote = engine("notes", note) if env.get("schema_ok") else None
    row = {
        "i2s_hex": hx,
        "sha3_256": digest,
        "pq_sig": False,
        "sphincs": False,
        "onboard": ["hashlib.sha256", "hashlib.sha3_256", "hashlib.blake2b"],
        "skip": ["SPHINCS+", "ML-DSA", "liboqs"],
        "brain_write": bool(wrote and wrote.get("wrote")),
    }
    if other is not None:
        env2 = handshake(other, job="dash-viewport")
        hx2 = str((env2.get("note") or {}).get("i2s_hex") or "")
        row["hamming_bits"] = ham(hx, hx2)
        row["other"] = other
    return row
