"""JuniorNightTernary — original overnight BitNet-native state machine.

Weights stay in {-1,0,1}. Night ticks apply a sparse neighbor vote so most
dims sleep (stay 0). Drift is an EMA of |delta|. Morning brief maps drift
to the shared recommendation vocab.

This is JuniorCloud original tech, not a hosted model runner.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

DIM = 32


def _clamp3(v: int) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def tick(state: list[int], step: int) -> list[int]:
    n = len(state)
    nxt = [0] * n
    for i, v in enumerate(state):
        left = state[(i - 1) % n]
        right = state[(i + 1) % n]
        vote = left + v + right
        # sleep: every third dim on even steps stays 0 (overnight sparsity)
        if (i + step) % 3 == 0:
            nxt[i] = 0
        else:
            nxt[i] = _clamp3(vote if vote != 0 else v)
    return nxt


def drift(a: list[int], b: list[int]) -> float:
    if not a or len(a) != len(b):
        return 1.0
    return sum(abs(x - y) for x, y in zip(a, b)) / (2.0 * len(a))


@dataclass
class NightBrief:
    ticks: int
    final_drift: float
    mean_drift: float
    sparsity: float
    recommendation: str
    state: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DreamMesh:
    def __init__(self, seed: list[int] | None = None, decay: float = 0.9):
        self.state = list(seed) if seed is not None else [1, -1, 0] * (DIM // 3) + [0] * (DIM % 3)
        if len(self.state) != DIM:
            self.state = (self.state + [0] * DIM)[:DIM]
        self.ema = 0.0
        self.decay = decay
        self.history: list[float] = []

    def run(self, ticks: int = 64) -> NightBrief:
        ticks = max(1, min(int(ticks), 10_000))
        for t in range(ticks):
            nxt = tick(self.state, t)
            d = drift(self.state, nxt)
            self.ema = self.decay * self.ema + (1.0 - self.decay) * d
            self.history.append(d)
            self.state = nxt
        zeros = self.state.count(0)
        sparsity = zeros / len(self.state)
        mean_d = sum(self.history) / len(self.history)
        if self.ema < 0.08:
            rec = "high_confidence"
        elif self.ema < 0.22:
            rec = "review_needed"
        else:
            rec = "low_confidence"
        return NightBrief(
            ticks=ticks,
            final_drift=round(self.ema, 4),
            mean_drift=round(mean_d, 4),
            sparsity=round(sparsity, 4),
            recommendation=rec,
            state=list(self.state),
        )
