"""Eval harness — trajectory grading + CI regression gate."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Grade:
    name: str
    score: float
    pass_: bool
    notes: str


def grade_trajectory(steps: list[str], must_include: list[str], floor: float = 0.7) -> Grade:
    blob = " ".join(steps).lower()
    hits = sum(1 for m in must_include if m.lower() in blob)
    score = hits / max(1, len(must_include))
    return Grade("trajectory", score, score >= floor, f"{hits}/{len(must_include)}")


def ci_gate(grades: list[Grade]) -> bool:
    return all(g.pass_ for g in grades)
