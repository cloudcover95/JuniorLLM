"""One terraform ticket. No editor launch. No mesh append."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "terraform.json"
SURFACES = ("code", "blender", "llm")


def run(note: str = "JuniorOSai", surface: str = "code") -> dict:
    surface = surface if surface in SURFACES else "code"
    xs = [((ord(c) % 5) - 2) / 2.0 for c in note[:32]] or [0.0]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    zeros = sum(1 for x in xs if abs(round(x / gamma)) == 0)
    body = {
        "protocol": "goldend-osai-omega/1",
        "hops": ["JuniorOSai", "Goldend", surface],
        "surface": surface,
        "note": note[:160],
        "trit_energy": round(1.0 - zeros / len(xs), 3),
        "launch": False,
        "bpy": False,
        "model_pull": False,
        "writes": 1,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body
