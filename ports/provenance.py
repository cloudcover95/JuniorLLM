"""Fact vs modelled provenance for inference and agent state. Local only.

Every numeric claim is labelled. Prevents treating a heuristic as ground truth.
"""
from __future__ import annotations
from typing import Any

def fact(name: str, value: Any, unit: str = "") -> dict:
    return {"name": name, "value": value, "unit": unit, "kind": "fact", "local": True}

def modelled(name: str, value: Any, unit: str = "", assumption: str = "") -> dict:
    return {"name": name, "value": value, "unit": unit, "kind": "modelled", "assumption": assumption, "local": True}

def label(score: float, source: str = "trit") -> dict:
    """Trit and envelope scores are modelled. Hardware probe is fact."""
    if source in ("trit", "envelope", "combined", "hillclimb"):
        return modelled(source, score, assumption="heuristic from packed note")
    return fact(source, score)
