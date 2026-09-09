"""Eval harness — trajectory grading + CI regression gate."""
from __future__ import annotations

from dataclasses import dataclass

from evals.contracts import rec_ok


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


def rec_vocab_grade(label: str) -> Grade:
    """CI gate piece: TDA/covenant recommendation vocab from evals.contracts."""
    ok = rec_ok(label)
    return Grade("rec_vocab", 1.0 if ok else 0.0, ok, label)


def ci_gate(grades: list[Grade], recs: list[str] | None = None) -> bool:
    extra = [rec_vocab_grade(r) for r in recs] if recs is not None else []
    return all(g.pass_ for g in list(grades) + extra)
