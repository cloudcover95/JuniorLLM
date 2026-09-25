"""FIPS 202 SHA3-256 of a note. Not I2_S."""
from __future__ import annotations

import hashlib


def digest(note: str) -> dict:
    raw = (note or "").encode("utf-8")
    return {
        "sha3_256": hashlib.sha3_256(raw).hexdigest(),
        "n": len(raw),
        "trit_is_not_hash": True,
    }
