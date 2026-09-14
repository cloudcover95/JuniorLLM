"""Local StoneField seed. No MP/KAYA scrape."""
from __future__ import annotations

import json
from pathlib import Path

SEED = Path(__file__).resolve().parents[1] / "junior_osai" / "goldens" / "stonefield.json"


def load() -> dict:
    return json.loads(SEED.read_text(encoding="utf-8"))


def find(q: str) -> dict | None:
    t = (q or "").lower()
    for n in load().get("nodes") or []:
        blob = " ".join([n.get("id", ""), n.get("name", ""), n.get("area", "")]).lower()
        if any(p in blob for p in t.split()) or n["id"] in t:
            return n
    return None


def list_public() -> list[dict]:
    return [n for n in load().get("nodes") or [] if n.get("public")]
