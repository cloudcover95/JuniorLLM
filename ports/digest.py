"""Read the Home digest index."""
from __future__ import annotations

import json
from pathlib import Path

INDEX = Path.home() / ".juniorhome" / "digest" / "index.json"


def read() -> dict:
    if not INDEX.exists():
        return {"ok": False, "reason": "no digest", "model_pull": False}
    row = json.loads(INDEX.read_text(encoding="utf-8"))
    row["ok"] = row.get("n", 0) > 0
    row["bind"] = "127.0.0.1"
    return row
