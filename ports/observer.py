"""Read the Home ledger. Does not post an order."""
from __future__ import annotations

import json
from pathlib import Path

LEDGER = Path.home() / ".juniorhome" / "gaia_mesh" / "ledger.jsonl"


def read(n: int = 8) -> dict:
    if not LEDGER.exists():
        return {"ok": False, "reason": "no ledger", "order": False, "model_pull": False}
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").strip().splitlines()[-n:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {
        "ok": True,
        "n": len(rows),
        "orders": sum(1 for r in rows if r.get("order")),
        "bind": "127.0.0.1",
        "model_pull": False,
    }
