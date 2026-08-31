"""Junior self-prompt — agent writes the next prompt from STATE.md + last receipt."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.guardrails import scan_prompt


@dataclass
class NextPrompt:
    text: str
    ok: bool
    reasons: list[str]


TEMPLATE = """You are JuniorOvernight on BitNet-local rails.
STATE:
{state}
LAST:
{last}
OBJECTIVE:
{objective}
RULES:
- Do not publish private-land pins without owner consent.
- Do not call hosted APIs unless port.kind allows it.
- Stop if TDA recommendation is low_confidence.
- Write a receipt JSON when done.
NEXT ACTION (one paragraph):
"""


def compose(state: str, last: str, objective: str) -> NextPrompt:
    text = TEMPLATE.format(state=state.strip() or "(empty)", last=last.strip() or "(none)", objective=objective.strip())
    rail = scan_prompt(text + " " + objective)
    return NextPrompt(text=text, ok=rail.ok, reasons=rail.reasons)


def compose_from_dir(root: Path, objective: str) -> NextPrompt:
    state = (root / "STATE.md").read_text(encoding="utf-8") if (root / "STATE.md").exists() else ""
    last = (root / "LAST.md").read_text(encoding="utf-8") if (root / "LAST.md").exists() else ""
    return compose(state, last, objective)
