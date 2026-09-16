"""I2_S cache + SHA3 identity. Collisions are aliases, not receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ports.flagstaff_balance import check
from ports.gaia_proto import handshake
from ports.osai_engines import run as engine
from ports.zero_trust import metric as zt

STORE: dict[str, str] = {}
TRAIN = Path.home() / ".juniorhome" / "gaia_mesh" / "train_cache.jsonl"


def _sha(note: str) -> str:
    return hashlib.sha3_256((note or "").encode()).hexdigest()


def put(note: str) -> dict:
    gate = check(note)
    z = zt(note)
    if not gate.get("ok") or not z.get("ok"):
        return {"ok": False, "collision": False, "zero_trust": z.get("ok")}
    hx = str(((handshake(note, job="dash-viewport").get("note") or {}).get("i2s_hex") or ""))
    digest = _sha(note)
    prior = STORE.get(hx)
    collision = bool(prior and prior != digest)
    STORE[hx] = digest
    row = {
        "ok": not collision,
        "i2s_hex": hx,
        "sha3": digest,
        "collision": collision,
        "alias_of": prior if collision else None,
        "crypto_hash": "sha3_256",
        "trit_is_cr": False,
    }
    if row["ok"]:
        engine("notes", note)
        TRAIN.parent.mkdir(parents=True, exist_ok=True)
        TRAIN.open("a", encoding="utf-8").write(json.dumps(row) + "\n")
        row["trained"] = True
    return row
