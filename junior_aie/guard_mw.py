"""Guardrails middleware — injection detection + PII redaction."""
from __future__ import annotations

import re
from dataclasses import dataclass

from agent.guardrails import scan_prompt

_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")


@dataclass
class Guarded:
    ok: bool
    text: str
    reasons: list[str]


def apply(text: str) -> Guarded:
    rail = scan_prompt(text)
    redacted = _EMAIL.sub("[EMAIL]", text or "")
    redacted = _PHONE.sub("[PHONE]", redacted)
    reasons = list(rail.reasons)
    if redacted != (text or ""):
        reasons.append("pii_redacted")
    return Guarded(ok=rail.ok, text=redacted, reasons=reasons)
