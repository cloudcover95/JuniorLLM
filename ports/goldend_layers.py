"""Goldend layered inference. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.layer_compute import layer_score
from ports.inference_engine import record

def run(note: str = "goldend layered") -> dict:
    layers = layer_score(note)
    rec = record(note, source="goldend_layer")
    body = {"layers": layers, "record": rec, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_layers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
