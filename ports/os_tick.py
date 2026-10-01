"""JuniorOS tick. One pass. No model pull. Envelope is a budget, not a wattmeter."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ports.ecosystem import list_cores, route

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "os_tick.jsonl"
BUDGET = {"watts": 45, "t4_watts": 12, "notes": 8, "download_gb": 0.0, "bind": "127.0.0.1"}


def tick(task: str = "JuniorOS") -> dict:
    t0 = time.perf_counter()
    cores = list_cores()[: BUDGET["notes"]]
    port = route(task, 0.0)
    row = {
        "protocol": "goldend-osai-omega/1",
        "port": port.name,
        "cores_n": len(list_cores()),
        "shown": [c["name"] for c in cores],
        "budget": BUDGET,
        "ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "model_pull": False,
        "ue5_launch": False,
        "bind": "127.0.0.1",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"port": row["port"], "ms": row["ms"]}) + "\n")
    return row
