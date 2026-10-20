"""RLVR for LLM and OSai agent workflows. Local only.

Verifiable reward: a check that returns pass/fail with a scalar score.
No external trainer. No model pull. Envelope scales by score and note length.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

@dataclass
class Reward:
    name: str
    score: float  # 0.0 to 1.0
    passed: bool
    detail: str = ""

@dataclass
class Envelope:
    layer: str          # T0, T1, T2
    max_tokens: int
    max_ms: int
    hardware: str       # cpu, metal, cuda, asahi
    handoff: str = ""   # next layer or agent

def verify(name: str, check: Callable[[], bool], detail: str = "") -> Reward:
    try:
        ok = bool(check())
    except Exception as e:
        ok = False
        detail = str(e)
    return Reward(name=name, score=1.0 if ok else 0.0, passed=ok, detail=detail)

def scale_envelope(rewards: list[Reward], note_len: int, hardware: str = "cpu") -> Envelope:
    """Auto-scale by mean score and note length. Local only."""
    if not rewards:
        mean = 0.5
    else:
        mean = sum(r.score for r in rewards) / len(rewards)
    if note_len < 256 and mean >= 0.8:
        return Envelope(layer="T0", max_tokens=512, max_ms=200, hardware=hardware)
    if note_len < 2048 and mean >= 0.5:
        return Envelope(layer="T1", max_tokens=2048, max_ms=800, hardware=hardware, handoff="T2")
    return Envelope(layer="T2", max_tokens=8192, max_ms=4000, hardware=hardware, handoff="local_gguf")

def handoff(env: Envelope, next_agent: str) -> dict[str, Any]:
    return {
        "from": env.layer,
        "to": next_agent,
        "max_tokens": env.max_tokens,
        "max_ms": env.max_ms,
        "hardware": env.hardware,
        "ts": datetime.utcnow().isoformat(),
        "local": True,
    }

def log_cycle(rewards: list[Reward], env: Envelope, handoff_to: str = "") -> Path:
    body = {
        "rewards": [asdict(r) for r in rewards],
        "envelope": asdict(env),
        "handoff": handoff(env, handoff_to) if handoff_to else None,
        "ts": datetime.utcnow().isoformat(),
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "rlvr.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body) + "\n")
    return out

def agent_cycle(checks: list[tuple[str, Callable[[], bool]]], note_len: int, hardware: str = "cpu", next_agent: str = "") -> dict[str, Any]:
    """One RLVR cycle. Verifies, scales, optionally hands off."""
    rewards = [verify(name, fn) for name, fn in checks]
    env = scale_envelope(rewards, note_len, hardware)
    path = log_cycle(rewards, env, next_agent)
    return {
        "rewards": [asdict(r) for r in rewards],
        "envelope": asdict(env),
        "log": str(path),
        "local": True,
    }
