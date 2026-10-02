"""Read the Home terraform ticket. Does not launch."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path.home() / ".juniorhome" / "os" / "terraform.json"


def read() -> dict:
    if not PATH.exists():
        return {"ok": False, "reason": "no terraform", "bpy": False}
    row = json.loads(PATH.read_text(encoding="utf-8"))
    row["ok"] = True
    row["bpy"] = False
    return row
