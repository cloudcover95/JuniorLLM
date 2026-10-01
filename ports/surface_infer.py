"""Infer from a spun surface ticket. No model pull."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ports.surface_spin import load

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "surface_infer.jsonl"


def _trits(note: str) -> list[int]:
    xs = [((ord(c) % 17) / 17.0) * 2 - 1 for c in (note or "x")[:32]]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return out


def infer(kind: str = "web") -> dict:
    row = load(kind)
    if not row.get("ok"):
        return row
    trits = _trits(row.get("sha3") or kind)
    body = {
        "ok": True,
        "kind": kind,
        "sha3": row.get("sha3"),
        "zeros": trits.count(0),
        "pack_sha": hashlib.sha3_256(bytes(t + 1 for t in trits)).hexdigest()[:16],
        "inference": "ticket",
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "pack_sha": body["pack_sha"]}) + "\n")
    return body
