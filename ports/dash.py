"""Read Home dash.json."""
from __future__ import annotations

import json
from pathlib import Path

DASH = Path.home() / ".juniorhome" / "os" / "dash.json"


def read() -> dict:
    if not DASH.exists():
        return {"ok": False, "reason": "no dash", "boot": False}
    row = json.loads(DASH.read_text(encoding="utf-8"))
    row["ok"] = True
    return row
