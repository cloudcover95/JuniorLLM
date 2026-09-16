"""Stamp FIPS names on each runtime domain. KEM stays off."""
from __future__ import annotations

from ports.osai_six import NOTES, six

PQ = {
    "fips": ["203", "204", "205"],
    "stdlib": ["sha3_256", "blake2b"],
    "operator": ["liboqs", "ML-KEM-768", "ML-DSA-65"],
    "ml_kem": False,
}


def domains() -> dict:
    base = six()
    return {
        **PQ,
        "ok": base.get("ok"),
        "domains": {k: {"ok": v.get("ok"), **PQ} for k, v in (base.get("rows") or {}).items()},
        "notes": NOTES,
    }
