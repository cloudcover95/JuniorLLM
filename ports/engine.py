"""Read the Home engine receipt."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path.home() / ".juniorhome" / "os" / "engine.json"


def read() -> dict:
    if not PATH.exists():
        return {"ok": False, "reason": "no engine", "boot": False}
    row = json.loads(PATH.read_text(encoding="utf-8"))
    row["ok"] = True
    return row
