"""Inference data engine for agent and Home loops. Local only.

Records notes, applies trit score, optional Winsor clip, writes a ledger.
No model pull.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from ports.trit_score import trit_reward, pack_trit

LEDGER = Path.home() / ".juniorhome" / "os" / "inference_ledger.jsonl"

def winsor_clip(text: str, cap: int = 64) -> str:
    """Simple Winsor-style clip on length. Domain math can replace this."""
    return text[:cap]

def record(note: str, source: str = "agent") -> dict[str, Any]:
    clipped = winsor_clip(note)
    score, passed, detail = trit_reward(clipped)
    trits = pack_trit(clipped)
    body = {
        "source": source,
        "note_len": len(note),
        "clipped_len": len(clipped),
        "trit_score": score,
        "passed": passed,
        "detail": detail,
        "trit_n": len(trits),
        "ts": datetime.utcnow().isoformat(),
        "local": True,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body) + "\n")
    return body

def latest(n: int = 5) -> list[dict]:
    if not LEDGER.exists():
        return []
    lines = LEDGER.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out
