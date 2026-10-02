"""Read the harvest critique. measured stays false."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path.home() / ".juniorhome" / "os" / "harvest.json"


def read() -> dict:
    if not PATH.exists():
        return {"ok": False, "reason": "no harvest", "measured": False}
    row = json.loads(PATH.read_text(encoding="utf-8"))
    row["ok"] = True
    row["measured"] = False
    return row
