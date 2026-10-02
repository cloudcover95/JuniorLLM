"""Agent read of the Home integrity ticket."""
from __future__ import annotations

import json
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "mesh_integrity.jsonl"


def read() -> dict:
    if not MESH.exists():
        return {"ok": False, "reason": "no ticket", "svd_1024": False}
    row = json.loads(MESH.read_text(encoding="utf-8").strip().splitlines()[-1])
    row["svd_1024"] = False
    row["bind"] = "127.0.0.1"
    return row
