"""Read the lean receipt."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path.home() / ".juniorhome" / "os" / "lean.json"


def read() -> dict:
    if not PATH.exists():
        return {"ok": False, "reason": "no lean"}
    row = json.loads(PATH.read_text(encoding="utf-8"))
    row["ok"] = True
    return row
