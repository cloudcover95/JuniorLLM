"""Apply a named envelope to an OSai route."""
from __future__ import annotations

import json
from pathlib import Path

from ports.ecosystem import list_cores, route
from ports.envelope import fit, select, slice_rows

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "envelope.jsonl"


def run(task: str = "JuniorOSai", env: str = "host") -> dict:
    budget = select(env)
    port = route(task, 0.0)
    cores = slice_rows(list_cores(), budget["notes"])
    note = fit(task, budget["chars"])
    row = {
        "protocol": "goldend-osai-omega/1",
        "port": port.name,
        "note": note,
        "cores": [c["name"] for c in cores],
        "budget": budget,
        "model_pull": False,
        "ue5_launch": False,
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"env": budget["name"], "port": port.name, "n": len(cores)}) + "\n")
    return row
