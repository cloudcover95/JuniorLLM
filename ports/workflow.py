"""Read the Home workflow log."""
from __future__ import annotations

import json
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "workflow.jsonl"


def read() -> dict:
    if not MESH.exists():
        return {"ok": False, "reason": "no workflow", "boot": False}
    lines = MESH.read_text(encoding="utf-8").strip().splitlines()
    hops = [json.loads(line).get("hop") for line in lines[-10:]]
    return {"ok": True, "n": len(lines), "last": hops, "boot": False, "bind": "127.0.0.1"}
