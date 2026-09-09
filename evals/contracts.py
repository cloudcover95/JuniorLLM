"""Required evals: covenant vocab + TDA recommendation labels. FLOPS optional."""
from __future__ import annotations

RECS = {"high_confidence", "review_needed", "low_confidence"}


def flops_proxy(params: int, bits: float = 1.58) -> float:
    """Not a hardware counter. Scale params by ternary width."""
    return float(params) * bits / 16.0


def rec_ok(label: str) -> bool:
    return label in RECS
