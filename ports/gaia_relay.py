"""OSai → Gaia → mesh relay. Ticket only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ports.envelope import fit, select
from ports.osai_envelope import run

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "relay.jsonl"


def activate(task: str = "Gaia", env: str = "t4") -> dict:
    budget = select(env)
    osai = run(task, env)
    note = fit(task, budget["chars"])
    hops = ["JuniorOSai", "Gaia", "mesh"]
    body = {
        "protocol": "goldend-osai-omega/1",
        "hops": hops,
        "note": note,
        "port": osai["port"],
        "sha3": hashlib.sha3_256(note.encode()).hexdigest()[:16],
        "env": budget["name"],
        "bind": "127.0.0.1",
        "model_pull": False,
        "ue5_launch": False,
        "inference": "ticket",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sha3": body["sha3"], "hops": hops, "env": body["env"]}) + "\n")
    return body


if __name__ == "__main__":
    print(json.dumps(activate(), indent=2))
