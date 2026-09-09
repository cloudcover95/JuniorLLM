"""Model router — cost / latency / quality + fallbacks onto Junior ports."""
from __future__ import annotations

from dataclasses import dataclass

from ports.registry import LLMPort, pick

PROFILE = {
    "JuniorBitNetFieldCore": {"cost": 0.0, "latency_ms": 8, "quality": 0.45},
    "BitNet-2B4T": {"cost": 0.2, "latency_ms": 40, "quality": 0.62},
    "JuniorGemma4-4B": {"cost": 0.4, "latency_ms": 80, "quality": 0.78},
    "JuniorFable": {"cost": 0.0, "latency_ms": 5, "quality": 0.50},
    "JuniorAstra": {"cost": 0.1, "latency_ms": 20, "quality": 0.70},
    "JuniorAstraReason": {"cost": 0.5, "latency_ms": 90, "quality": 0.80},
    "JuniorKimiK3-edge": {"cost": 0.9, "latency_ms": 140, "quality": 0.84},
    "Qwen-local": {"cost": 0.6, "latency_ms": 110, "quality": 0.79},
}

FALLBACKS = {
    "JuniorAstraReason": ["JuniorGemma4-4B", "JuniorBitNetFieldCore"],
    "JuniorGemma4-4B": ["BitNet-2B4T", "JuniorBitNetFieldCore"],
    "JuniorKimiK3-edge": ["JuniorAstraReason", "JuniorGemma4-4B"],
    "Qwen-local": ["JuniorGemma4-4B", "JuniorBitNetFieldCore"],
}


@dataclass
class Route:
    primary: LLMPort
    fallbacks: list[str]
    score: float
    reason: str


class ModelRouter:
    def choose(self, task: str, ram_gb: float = 8.0, prefer: str = "balanced") -> Route:
        port = pick(task, ram_gb)
        prof = PROFILE.get(port.name, {"cost": 0.5, "latency_ms": 50, "quality": 0.5})
        if prefer == "cheap":
            score = 1.0 - prof["cost"]
        elif prefer == "fast":
            score = 1.0 - min(prof["latency_ms"], 200) / 200.0
        else:
            score = 0.5 * prof["quality"] + 0.3 * (1 - prof["cost"]) + 0.2 * (1 - min(prof["latency_ms"], 200) / 200)
        return Route(port, FALLBACKS.get(port.name, ["JuniorBitNetFieldCore"]), round(score, 4), prefer)
