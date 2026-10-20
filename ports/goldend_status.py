"""Goldend status writer. Combines layer score and latest pack. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.layer_compute import layer_score
from ports.pack_ledger import store

def write(note: str = "goldend status") -> dict:
    layers = layer_score(note)
    pack = store(note, source="goldend")
    body = {
        "layers": layers,
        "pack_score": pack["score"],
        "pack_n": len(pack["trits"]),
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "goldend_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
