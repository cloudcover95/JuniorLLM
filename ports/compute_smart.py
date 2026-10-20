"""Compute-smart path selector for JuniorOS agent. Local only.

Chooses T0/T1/T2 from bench numbers and note length. No model pull.
"""
from __future__ import annotations
import json
from pathlib import Path
from ports.rlvr import scale_envelope, Reward
from ports.trit_score import trit_reward

def choose(note: str, hardware: str = "cpu") -> dict:
    score, passed, detail = trit_reward(note)
    rewards = [Reward("trit", score, passed, detail)]
    env = scale_envelope(rewards, note_len=len(note), hardware=hardware)
    if score >= 0.7 and len(note) < 128:
        env.layer = "T0"
        env.max_tokens = 256
        env.handoff = ""
    body = {
        "note_len": len(note),
        "trit_score": score,
        "layer": env.layer,
        "max_tokens": env.max_tokens,
        "hardware": hardware,
        "handoff": env.handoff,
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "compute_smart.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(choose("os tick"), indent=2))
