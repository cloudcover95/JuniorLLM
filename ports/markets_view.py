"""Local market tiles. Same orient/scale as Gaia dash. No live scrape."""
from __future__ import annotations

import json
from pathlib import Path

from ports.gaia_view import view

SEED = Path(__file__).resolve().parents[1] / "junior_osai" / "goldens" / "markets.json"


def board(note: str = "stock tape", orient: str = "landscape", scale: float = 1.0) -> dict:
    v = view(note, orient, scale)
    tiles = json.loads(SEED.read_text(encoding="utf-8")).get("tiles") or []
    v["tiles"] = tiles
    v["tape"] = " ".join(f"{t['sym']} {t['px']}" for t in tiles)
    v["live_feed"] = False
    v["omega"] = {"job": "dash-viewport", "w": v["px"][0], "h": v["px"][1], "mesh": "obj", "surface": "markets"}
    return v
