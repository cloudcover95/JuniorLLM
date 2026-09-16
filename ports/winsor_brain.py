"""P95 wire. Brain jsonl only if firing in (0.1, 0.9)."""
from __future__ import annotations

from ports.osai_engines import run as engine
from ports.sparsity_ops import ops


def tick(note: str = "journal field note") -> dict:
    s = ops(note)
    fire = float(s.get("firing") or 0)
    band = 0.1 < fire < 0.9
    wrote = None
    if s.get("ok") and band:
        wrote = engine("notes", note)
    return {
        **s,
        "band": band,
        "brain_write": bool(wrote and wrote.get("wrote")),
        "wire": "winsor-p95",
        "agent": "flagstaff-then-band",
    }
