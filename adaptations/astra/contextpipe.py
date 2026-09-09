"""JuniorAstra ContextPipe — budgeted context assembly (open Astra idea).

Does not stream vendor tokens. Packs local notes under a hard budget.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PackedContext:
    text: str
    tokens_est: int
    dropped: int
    provenance: list[str]


def _est_tokens(s: str) -> int:
    return max(1, len(s.split()))


def pack(parts: list[tuple[str, str]], budget: int = 512) -> PackedContext:
    """parts = [(source_name, text), ...] highest priority first."""
    kept: list[str] = []
    prov: list[str] = []
    used = 0
    dropped = 0
    for name, text in parts:
        t = (text or "").strip()
        if not t:
            continue
        cost = _est_tokens(t)
        if used + cost > budget:
            dropped += 1
            continue
        kept.append(f"[{name}] {t}")
        prov.append(name)
        used += cost
    return PackedContext(text="\n".join(kept), tokens_est=used, dropped=dropped, provenance=prov)
