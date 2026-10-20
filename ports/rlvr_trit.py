"""RLVR cycle with trit pack scoring. Local only."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from ports.rlvr import agent_cycle
from ports.trit_score import trit_reward, make_trit_check

def cycle_with_trit(note: str, hardware: str = "cpu", next_agent: str = "") -> dict[str, Any]:
    """One agent cycle that includes a trit pack reward."""
    score, passed, detail = trit_reward(note)
    checks = [
        ("trit_pack", make_trit_check(note)),
        ("non_empty", lambda: len(note) > 0),
    ]
    result = agent_cycle(checks, note_len=len(note), hardware=hardware, next_agent=next_agent)
    result["trit"] = {"score": score, "passed": passed, "detail": detail}
    if score < 0.4:
        result["envelope"]["layer"] = "T0"
        result["envelope"]["max_tokens"] = 256
        result["envelope"]["handoff"] = "repack"
    out = Path.home() / ".juniorhome" / "os" / "rlvr_trit.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "note_len": len(note),
            "trit": result["trit"],
            "envelope": result["envelope"],
            "ts": datetime.utcnow().isoformat(),
            "local": True,
        }) + "\n")
    return result
