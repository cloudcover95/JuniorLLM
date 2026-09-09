"""Data flywheel — feedback → synthetic pair → LoRA job receipt (no silent train)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pair:
    prompt: str
    completion: str
    score: float


@dataclass
class LoraJob:
    n_pairs: int
    status: str
    notes: str


class Flywheel:
    def __init__(self) -> None:
        self.pairs: list[Pair] = []

    def feedback(self, prompt: str, completion: str, score: float) -> None:
        self.pairs.append(Pair(prompt, completion, score))
        if score >= 0.8:
            self.pairs.append(Pair(prompt, completion, score))  # upsample good

    def lora_receipt(self, min_pairs: int = 4) -> LoraJob:
        good = [p for p in self.pairs if p.score >= 0.6]
        if len(good) < min_pairs:
            return LoraJob(len(good), "blocked", "not_enough_pairs")
        return LoraJob(len(good), "queued", "local LoRA job queued — run on-device, no 1.5TB pull")
