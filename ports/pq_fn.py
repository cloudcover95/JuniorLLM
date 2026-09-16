"""What each domain would do with PQ — when liboqs exists."""
from __future__ import annotations

import hashlib

from ports.osai_six import NOTES
from ports.osai_engines import run

FN = {
    "vault": {"now": "sha3_256 tag", "later": "ML-DSA on jsonl export"},
    "media": {"now": "blake2b of note", "later": "ML-DSA on asset list"},
    "notes": {"now": "sha3_256 + trit cache", "later": "SLH-DSA long-term"},
    "stock": {"now": "sha3 ticker row", "later": "ML-KEM session to peer node"},
    "cad": {"now": "blake2b obj name", "later": "ML-DSA mesh receipt"},
    "os": {"now": "zero-trust metric", "later": "liboqs on operator box"},
}


def fn(engine: str | None = None) -> dict:
    keys = [engine] if engine in FN else list(FN)
    rows = {}
    for k in keys:
        note = NOTES.get(k, "home dash")
        r = run(k, note)
        digest = hashlib.sha3_256(note.encode()).hexdigest()[:16]
        rows[k] = {**FN[k], "ok": r.get("ok"), "sha3_16": digest, "ml_kem": False}
    return {"fips": ["203", "204", "205"], "rows": rows}
