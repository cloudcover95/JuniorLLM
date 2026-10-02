"""Read the kept Goldend note."""
from __future__ import annotations

import json
from pathlib import Path

BEST = Path.home() / ".juniorhome" / "os" / "goldend_best.json"


def read() -> dict:
    if not BEST.exists():
        return {"ok": False, "reason": "no best", "train": False}
    row = json.loads(BEST.read_text(encoding="utf-8"))
    row["ok"] = True
    row["train"] = False
    return row
