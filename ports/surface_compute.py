"""Trit compute on a spun surface ticket. No model pull."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ports.surface_spin import load

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "surface_compute.jsonl"


def absmean(xs: list[float]) -> tuple[list[int], float]:
    if not xs:
        return [], 1.0
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return out, gamma


def pack5(trits: list[int]) -> str:
    acc = 0
    n = 0
    out = bytearray()
    for t in trits:
        acc = acc * 3 + (t + 1)
        n += 1
        if n == 5:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc)
    return out.hex()


def compute(kind: str = "web") -> dict:
    row = load(kind)
    if not row.get("ok"):
        return row
    xs = [float(ord(c) % 17) / 17.0 for c in (row.get("sha3") or "0")]
    wq, gamma = absmean(xs)
    body = {
        "protocol": "goldend-osai-omega/1",
        "kind": kind,
        "sha3": row.get("sha3"),
        "gamma": gamma,
        "zeros": wq.count(0),
        "pack5": pack5(wq),
        "bind": "127.0.0.1",
        "model_pull": False,
        "inference": "ticket",
    }
    body["id"] = hashlib.sha3_256(body["pack5"].encode()).hexdigest()[:16]
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": body["id"], "kind": kind}) + "\n")
    return body
