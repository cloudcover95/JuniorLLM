"""Three extra OSai engines. Train = jsonl after votes. No fake GGUF."""
from __future__ import annotations

import json
from pathlib import Path

from ports.agent_pipe import step

DIR = Path.home() / ".juniorhome" / "gaia_mesh"
ENGINES = ("vault", "media", "notes")


def run(engine: str, note: str) -> dict:
    engine = engine if engine in ENGINES else "notes"
    row = step(note, write=False)
    path = DIR / f"engine_{engine}.jsonl"
    wrote = False
    if row.get("ok"):
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"engine": engine, "note": note, "profit": row.get("profit"), "train": "jsonl", "gguf_mock": False}
        path.open("a", encoding="utf-8").write(json.dumps(rec) + "\n")
        wrote = True
    return {"engine": engine, "ok": row.get("ok"), "votes": row.get("votes"), "wrote": wrote, "path": str(path) if wrote else None, "download": False}


def all_three(note: str = "journal field note") -> dict:
    return {"engines": ENGINES, "rows": [run(e, note) for e in ENGINES]}
