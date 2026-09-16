"""Home receipt: Flagstaff → handshake → I2_S cache → optional neighbor. Not a tx."""
from __future__ import annotations

import json
from pathlib import Path

from ports.ham_pq import ham, tool
from ports.sol_trit import receipt
from ports.trit_cache import hit, put

LEDGER = Path.home() / ".juniorhome" / "gaia_mesh" / "receipts.jsonl"


def issue(note: str, neighbor: str | None = None) -> dict:
    rec = receipt(note)
    if not rec.get("ok"):
        return {**rec, "cached": False, "crypto": False, "submit": False}
    put(note)
    near = hit(neighbor or note)
    tag = tool(note, neighbor)
    row = {
        "ok": True,
        "i2s_hex": rec.get("i2s_hex"),
        "gamma": rec.get("gamma"),
        "exact": near.get("exact"),
        "near": near.get("near"),
        "hamming_bits": tag.get("hamming_bits"),
        "sha3_16": (tag.get("sha3_256") or "")[:16],
        "ledger": str(LEDGER),
        "rpc": False,
        "submit": False,
        "crypto": False,
        "protocol": rec.get("protocol"),
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.open("a", encoding="utf-8").write(json.dumps(row) + "\n")
    row["wrote"] = True
    return row
