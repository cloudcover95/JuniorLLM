"""Local hill-climb using trit score as verifiable reward. No model pull.

One change at a time. Keep if score rises. Held-out note catches collapse.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Callable
from ports.trit_score import trit_reward

LOG = Path.home() / ".juniorhome" / "os" / "hillclimb.jsonl"

def climb(
    note: str,
    mutate: Callable[[str], str],
    held_out: str = "",
    rounds: int = 4,
) -> dict:
    """Mutate, score, keep if better. Held-out must not drop."""
    best = note
    best_score, _, _ = trit_reward(best)
    history = [{"round": 0, "score": best_score, "kept": True}]
    for i in range(1, rounds + 1):
        candidate = mutate(best)
        score, passed, detail = trit_reward(candidate)
        keep = score > best_score
        if keep and held_out:
            h_score, _, _ = trit_reward(held_out)
            c_held = mutate(held_out)
            c_h, _, _ = trit_reward(c_held)
            if c_h < h_score:
                keep = False
        if keep:
            best = candidate
            best_score = score
        history.append({"round": i, "score": score, "kept": keep, "detail": detail})
    body = {
        "best": best,
        "best_score": best_score,
        "history": history,
        "ts": datetime.utcnow().isoformat(),
        "local": True,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body) + "\n")
    return body

def trim_mutate(text: str) -> str:
    """Example mutate: drop trailing space or last char if long."""
    if len(text) > 32:
        return text[:-1]
    return text.strip() or text
