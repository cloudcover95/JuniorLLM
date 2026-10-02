"""Read Home web3 receipt. No RPC."""
from __future__ import annotations

import json
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "web3_receipt.jsonl"


def read() -> dict:
    if not MESH.exists():
        return {"ok": False, "rpc": False, "address": False, "model_pull": False}
    row = json.loads(MESH.read_text(encoding="utf-8").strip().splitlines()[-1])
    return {"ok": True, "receipt": row.get("receipt"), "ledger": row.get("ledger"), "rpc": False, "address": False, "model_pull": False}
