"""Web3node-style math hook for the inference engine. Local only.

Placeholder absmean. Domain math can replace this.
"""
from __future__ import annotations
from ports.layer_compute import layer_score

def web3_score(note: str) -> dict:
    """Entry that web3node can later override with Winsor or absmean."""
    return layer_score(note)
