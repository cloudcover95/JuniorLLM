"""Read a spun surface ticket. No bundler."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.home() / ".juniorhome" / "surfaces"


def load(kind: str = "web") -> dict:
    path = ROOT / kind / "ticket.json"
    if not path.exists():
        return {"ok": False, "reason": "no ticket", "kind": kind, "model_pull": False}
    row = json.loads(path.read_text(encoding="utf-8"))
    row["ok"] = row.get("bind") == "127.0.0.1" and row.get("model_pull") is False
    return row
