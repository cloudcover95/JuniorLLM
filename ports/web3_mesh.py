"""Read the Home web3 mesh. No RPC."""
from __future__ import annotations

import json
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "web3_mesh.jsonl"


def read() -> dict:
    if not MESH.exists():
        return {"ok": False, "reason": "no web3_mesh", "rpc": False, "model_pull": False}
    line = MESH.read_text(encoding="utf-8").strip().splitlines()[-1]
    row = json.loads(line)
    row["ok"] = True
    row["rpc"] = False
    row["bind"] = "127.0.0.1"
    return row
