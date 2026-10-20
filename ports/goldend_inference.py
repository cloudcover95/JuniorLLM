"""Goldend inference entry. Records a note and runs the hill-climb path. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.inference_engine import record
from ports.goldend_hillclimb import run as climb_run

def run(note: str = "goldend inference") -> dict:
    rec = record(note, source="goldend")
    climb = climb_run(note)
    body = {"record": rec, "climb": climb["climb"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_inference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
