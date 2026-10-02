"""Read Home vault notes. No sync."""
from __future__ import annotations

from pathlib import Path

VAULT = Path.home() / ".juniorhome" / "vault" / "bitnetCloud" / "notes"


def read(n: int = 8) -> dict:
    if not VAULT.exists():
        return {"ok": False, "reason": "no vault", "sync": False, "model_pull": False}
    notes = sorted(VAULT.glob("*.md"))[-n:]
    return {
        "ok": True,
        "n": len(notes),
        "names": [p.name for p in notes],
        "sync": False,
        "bind": "127.0.0.1",
        "model_pull": False,
    }
