"""Write a vault markdown note from the localLLM card."""
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
