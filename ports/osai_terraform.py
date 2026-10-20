"""OSai terraform hook. Reads Home tile and decides keep or handoff. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.layer_compute import layer_score

TILE = Path.home() / ".juniorhome" / "os" / "home_inference_tile.json"

def decide(note: str) -> dict:
    score = layer_score(note)
    tile = {}
    if TILE.exists():
        try:
            tile = json.loads(TILE.read_text(encoding="utf-8"))
        except Exception:
            tile = {}
    keep = score["layer"] == "T0" and score["combined"] >= 0.6
    body = {
        "note_layer": score["layer"],
        "combined": score["combined"],
        "keep": keep,
        "handoff": "" if keep else "repack",
        "tile_present": bool(tile),
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "osai_terraform.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(decide("os tick"), indent=2))
