"""Approximate note cache. Not a cryptographic hash."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from ports.gaia_proto import handshake

STORE: dict[str, dict] = {}


def key(note: str) -> str:
    env = handshake(note, job="dash-viewport")
    return str((env.get("note") or {}).get("i2s_hex") or "")


def sha(note: str) -> str:
    return hashlib.sha256((note or "").encode()).hexdigest()


def put(note: str) -> dict:
    k = key(note)
    STORE[k] = {"note": note, "sha": sha(note)}
    return {"i2s_hex": k, "sha256": STORE[k]["sha"], "i2s_bytes": len(k) // 2, "sha_bytes": 32, "crypto": False}


def hit(note: str) -> dict:
    k = key(note)
    found = k in STORE
    same_sha = found and STORE[k]["sha"] == sha(note)
    return {
        "hit_trit": found,
        "hit_sha": same_sha,
        "why": "same quant ≠ same bytes",
        "crypto": False,
        "use": "semantic cache / agent replay, not signatures",
    }
