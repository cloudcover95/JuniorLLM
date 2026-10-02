"""One surface per call. The other two are not written."""
from __future__ import annotations

import json
from pathlib import Path

OS = Path.home() / ".juniorhome" / "os"
SURFACES = ("code", "blender", "llm")


def run(note: str = "JuniorOSai", surface: str = "code") -> dict:
    if surface not in SURFACES:
        return {"ok": False, "reason": "surface", "surfaces": list(SURFACES)}
    xs = [((ord(c) % 5) - 2) / 2.0 for c in note[:32]] or [0.0]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    zeros = sum(1 for x in xs if abs(round(x / gamma)) == 0)
    body = {
        "ok": True,
        "protocol": "goldend-osai-omega/1",
        "surface": surface,
        "note": note[:160],
        "trit_energy": round(1.0 - zeros / len(xs), 3),
        "launch": False,
        "bpy": False,
        "model_pull": False,
        "writes": 1,
        "others": False,
        "bind": "127.0.0.1",
    }
    OS.mkdir(parents=True, exist_ok=True)
    (OS / f"terraform_{surface}.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body
