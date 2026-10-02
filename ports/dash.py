"""Read the Home dash ticket."""
from __future__ import annotations

import json
from pathlib import Path

DASH = Path.home() / ".juniorhome" / "surfaces" / "web" / "dash.json"


def read() -> dict:
    if not DASH.exists():
        return {"ok": False, "reason": "no dash", "svd": False, "model_pull": False}
    row = json.loads(DASH.read_text(encoding="utf-8"))
    row["ok"] = True
    return row
