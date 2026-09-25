"""Pulse schedule jsonl. Not an AWG."""
from __future__ import annotations

import json
from pathlib import Path

LAKE = Path.home() / ".juniorhome" / "lake" / "pulses.jsonl"


def append(axis: int, theta: float, dt: float = 1.0) -> dict:
    LAKE.parent.mkdir(parents=True, exist_ok=True)
    row = {"axis": int(axis), "theta": float(theta), "dt": float(dt), "awg": False, "fridge": False}
    with LAKE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return {"wrote": True, "path": str(LAKE), **row}


def tail(n: int = 8) -> list:
    if not LAKE.is_file():
        return []
    lines = LAKE.read_text(encoding="utf-8").splitlines()[-n:]
    return [json.loads(x) for x in lines if x.strip()]
