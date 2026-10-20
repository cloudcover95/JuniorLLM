"""Web3node math adapter stub. Local only.

Replace score() with a real Winsor or absmean from web3node when available.
"""
from __future__ import annotations
from ports.layer_compute import layer_score

def score(note: str) -> dict:
    """Current: layered ternary. Swap body for web3node math."""
    return layer_score(note)
