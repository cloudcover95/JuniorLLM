"""Run an overnight dream cycle and persist a morning brief."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.guardrails import scan_prompt
from agent.overnight import enqueue
from bitnet_night.dream_mesh import DIM, DreamMesh, tick
from ports.registry import pick


def seed_from_text(text: str) -> list[int]:
    from agent.guardrails import hash_text

    h = hash_text(text)
    bits = []
    for i, ch in enumerate(h):
        v = int(ch, 16)
        bits.append(-1 if v < 5 else (0 if v < 10 else 1))
    return (bits * (DIM // len(bits) + 1))[:DIM]


def run_cycle(root: Path, objective: str, ticks: int = 128) -> dict:
    rail = scan_prompt(objective)
    if not rail.ok:
        brief = {"status": "blocked", "reasons": rail.reasons}
        root.mkdir(parents=True, exist_ok=True)
        (root / "MORNING.json").write_text(json.dumps(brief, indent=2), encoding="utf-8")
        return brief
    mesh = DreamMesh(seed_from_text(objective))
    night = mesh.run(ticks)
    port = pick(objective, ram_gb=8.0)
    queued = enqueue(root, objective)
    payload = {
        "status": "complete",
        "at": datetime.now(timezone.utc).isoformat(),
        "port": port.name,
        "queue": queued.get("status"),
        "night": night.to_dict(),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "MORNING.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "LAST.md").write_text(
        f"night ticks={night.ticks} drift={night.final_drift} rec={night.recommendation}\n",
        encoding="utf-8",
    )
    return payload


def warmup(n: int = 8) -> list[int]:
    s = [1, 0, -1] * 10 + [0, 0]
    for i in range(n):
        s = tick(s[:DIM] if len(s) >= DIM else (s + [0] * DIM)[:DIM], i)
    return s
