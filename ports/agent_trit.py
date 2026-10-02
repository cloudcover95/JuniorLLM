"""Agent workflow step: trit energy. svd stays off."""
from __future__ import annotations

import json
from pathlib import Path

from ports.trit_mesh import read

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "agent_trit.jsonl"


def step(note: str = "JuniorOSai") -> dict:
    row = read()
    body = {
        "protocol": "goldend-osai-omega/1",
        "step": "trit-energy",
        "note": note[:160],
        "energy": row.get("energy"),
        "ok": bool(row.get("ok")),
        "svd_1024": False,
        "next": "flagstaff" if row.get("ok") else "trit_mesh_prod",
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": body["step"], "ok": body["ok"]}) + "\n")
    return body
