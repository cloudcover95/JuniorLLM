"""Web3node-style absmean step for the labelled path. Local only.

Domain math can replace absmean. Current is mean absolute trit.
"""
from __future__ import annotations
from ports.trit_score import pack_trit
from ports.provenance import modelled

def absmean_step(note: str) -> dict:
    trits = pack_trit(note)
    if not trits:
        energy = 0.0
    else:
        energy = sum(abs(t) for t in trits) / len(trits)
    return modelled("absmean", energy, assumption="mean absolute trit")

def feed(note: str) -> dict:
    """Entry the labelled path can call."""
    return absmean_step(note)
