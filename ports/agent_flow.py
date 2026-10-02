"""Agent step. Trit energy is the score. No weight update."""
from __future__ import annotations

import json
from pathlib import Path

from ports.goldend_fetch import read

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "agent_flow.jsonl"


def step(note: str = "JuniorOSai") -> dict:
    best = read()
    energy = float(best.get("energy") or 0) if best.get("ok") else 0.0
    body = {
        "protocol": "goldend-osai-omega/1",
        "hops": ["Goldend", "JuniorFetch", "JuniorLLM", "JuniorOSai"],
        "note": note[:160],
        "energy": energy,
        "kept": best.get("note"),
        "train": False,
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"energy": energy}) + "\n")
    return body
