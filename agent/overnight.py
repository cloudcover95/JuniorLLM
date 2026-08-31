"""JuniorOvernightAgent — queue jobs, self-prompt, guardrail, write receipt.

Does not spawn unbounded subprocesses. Jobs are JSON files in data/agent/overnight/.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.guardrails import scan_prompt, scan_skill_source
from agent.self_prompt import compose
from ports.registry import pick


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def enqueue(root: Path, objective: str, skill_src: str = "") -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    pr = scan_prompt(objective)
    sr = scan_skill_source(skill_src) if skill_src else None
    if not pr.ok or (sr and not sr.ok):
        receipt = {
            "status": "blocked",
            "at": _now(),
            "prompt_rail": pr.to_dict(),
            "skill_rail": None if sr is None else sr.to_dict(),
        }
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return receipt
    nxt = compose(
        (root / "STATE.md").read_text(encoding="utf-8") if (root / "STATE.md").exists() else "",
        (root / "LAST.md").read_text(encoding="utf-8") if (root / "LAST.md").exists() else "",
        objective,
    )
    port = pick(objective, ram_gb=8.0)
    receipt = {
        "status": "queued",
        "at": _now(),
        "objective": objective,
        "port": port.name,
        "next_prompt_ok": nxt.ok,
        "self_prompt_head": nxt.text.splitlines()[0],
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (root / "NEXT_PROMPT.md").write_text(nxt.text, encoding="utf-8")
    return receipt
