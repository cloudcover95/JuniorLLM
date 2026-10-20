"""Trit pack ledger for inference. Local only.

Stores the packed trits alongside the score so later stages can reuse them.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from ports.trit_score import pack_trit, trit_reward

PACKS = Path.home() / ".juniorhome" / "os" / "trit_packs.jsonl"

def store(note: str, source: str = "agent") -> dict:
    trits = pack_trit(note)
    score, passed, detail = trit_reward(note)
    body = {
        "source": source,
        "note_len": len(note),
        "trits": trits,
        "score": score,
        "passed": passed,
        "detail": detail,
        "ts": datetime.utcnow().isoformat(),
        "local": True,
    }
    PACKS.parent.mkdir(parents=True, exist_ok=True)
    with PACKS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body) + "\n")
    return body

def latest(n: int = 3) -> list[dict]:
    if not PACKS.exists():
        return []
    lines = PACKS.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out
