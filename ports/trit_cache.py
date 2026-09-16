"""Semantic note cache. Not a cryptographic hash."""
from __future__ import annotations

import hashlib

from ports.gaia_proto import handshake

STORE: dict[str, str] = {}


def _hex(note: str) -> str:
    env = handshake(note, job="dash-viewport")
    return str((env.get("note") or {}).get("i2s_hex") or "")


def _hamming(a: str, b: str) -> int:
    n = min(len(a), len(b))
    d = sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))
    return d


def put(note: str) -> dict:
    k = _hex(note)
    STORE[k] = note
    return {"i2s_hex": k, "sha256": hashlib.sha256(note.encode()).hexdigest(), "n": len(STORE), "crypto": False}


def hit(note: str, *, max_d: int = 2) -> dict:
    k = _hex(note)
    exact = k in STORE
    near = []
    if not exact and k:
        for old, txt in STORE.items():
            d = _hamming(k, old)
            if d <= max_d:
                near.append({"d": d, "note": txt})
    return {
        "exact": exact,
        "near": near[:5],
        "sha_eq": exact and STORE.get(k) == note,
        "crypto": False,
    }


def replay(notes: list[str]) -> dict:
    out = []
    for n in notes:
        put(n)
        out.append(hit(n))
    return {"stored": len(STORE), "rows": out, "crypto": False}
