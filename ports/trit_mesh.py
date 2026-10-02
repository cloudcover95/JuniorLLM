"""Read trit mesh integrity. svd_1024 stays false."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path.home() / ".juniorhome" / "os" / "trit_mesh.json"


def read() -> dict:
    if not PATH.exists():
        return {"ok": False, "reason": "no trit_mesh", "svd_1024": False}
    row = json.loads(PATH.read_text(encoding="utf-8"))
    row["ok"] = True
    row["svd_1024"] = False
    return row
