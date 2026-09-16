"""Live imager tick. Do not rewrite tree.jsonl. Do not invent trit."""
from __future__ import annotations

from ports.imager import image
from ports.imager_auto import tick as auto_tick


def process(note: str = "gaia spine") -> dict:
    row = auto_tick(note)
    img = image(note)
    return {
        **row,
        "rewrote_tree": False,
        "numpy": False,
        "custom_p95_over_3": False,
        "i2s_from": "handshake-winsor",
        "gamma": img.get("gamma"),
    }
