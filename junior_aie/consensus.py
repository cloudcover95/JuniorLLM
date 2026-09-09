"""Multi-agent consensus — weighted voting + judge + escalation."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter


@dataclass
class Ballot:
    agent: str
    choice: str
    weight: float


@dataclass
class Verdict:
    choice: str
    confidence: float
    escalate: bool
    reason: str


def decide(ballots: list[Ballot], judge: str | None = None, threshold: float = 0.6) -> Verdict:
    tallies: dict[str, float] = {}
    for b in ballots:
        tallies[b.choice] = tallies.get(b.choice, 0.0) + b.weight
    total = sum(tallies.values()) or 1.0
    choice, score = max(tallies.items(), key=lambda kv: kv[1])
    conf = score / total
    if judge and judge in tallies:
        choice = judge if conf < threshold else choice
        reason = "judge_override" if conf < threshold else "majority"
    else:
        reason = "majority"
    return Verdict(choice, round(conf, 4), conf < threshold, reason)
