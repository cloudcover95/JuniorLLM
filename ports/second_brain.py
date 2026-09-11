"""Vault markdown for the localLLM card + inject log."""
from __future__ import annotations

from pathlib import Path

from ports.enduser_llm import spec


def note(vault: Path, body: str = "") -> Path:
    vault = Path(vault)
    brain = vault / "second_brain"
    brain.mkdir(parents=True, exist_ok=True)
    s = spec()
    p = brain / "local_llm.md"
    p.write_text(
        f"# {s['name']}\n\n- runtime: {s['runtime']}\n- ctx: {s['ctx']}\n- accel: {s['accel']}\n\n{body}\n",
        encoding="utf-8",
    )
    return p


def log(vault: Path, area: str, text: str) -> Path:
    vault = Path(vault)
    brain = vault / "second_brain"
    brain.mkdir(parents=True, exist_ok=True)
    p = brain / "inject_log.md"
    with p.open("a", encoding="utf-8") as f:
        f.write(f"- [{area}] {text}\n")
    return p
