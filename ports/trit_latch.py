"""Latch last I2_S key. Bench pack vs sha256."""
from __future__ import annotations

import hashlib
import time

from ports.gaia_proto import handshake
from ports.trit_cache import hit, put

LATCH = {"hex": None, "note": None}


def encode(note: str) -> dict:
    env = handshake(note, job="dash-viewport")
    hx = (env.get("note") or {}).get("i2s_hex")
    LATCH["hex"] = hx
    LATCH["note"] = note
    put(note)
    return {"i2s_hex": hx, "latched": True, "schema_ok": env.get("schema_ok")}


def bench(note: str = "buy oats", n: int = 30) -> dict:
    t0 = time.perf_counter()
    for _ in range(n):
        handshake(note, job="dash-viewport")
    ms_trit = (time.perf_counter() - t0) * 1000 / n
    b = (note or "").encode()
    t0 = time.perf_counter()
    for _ in range(n):
        hashlib.sha256(b).digest()
    ms_sha = (time.perf_counter() - t0) * 1000 / n
    enc = encode(note)
    return {
        "ms_handshake": round(ms_trit, 4),
        "ms_sha256": round(ms_sha, 4),
        "sha_faster": ms_sha < ms_trit,
        "latch": enc.get("i2s_hex"),
        "near": hit(note).get("near"),
        "hash_role": "integrity",
        "trit_role": "semantic-cache",
        "crypto": False,
    }
