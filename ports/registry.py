"""Junior custom LLM ports — local / high-quant first."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class LLMPort:
    name: str
    kind: str  # bitnet-native | high-quant | behavioral | edge-moe
    quant: str
    backend: str  # mlx | bitnet.cpp | gguf | torch-ternary | none
    max_download_gb: float
    notes: str


PORTS = [
    LLMPort("JuniorBitNetFieldCore", "bitnet-native", "ternary-1.58", "torch-ternary", 0.0, "Crowd/field scorer"),
    LLMPort("BitNet-2B4T", "bitnet-native", "I2_S", "bitnet.cpp", 1.5, "microsoft/bitnet-b1.58-2B-4T-gguf"),
    LLMPort("JuniorGemma4-4B", "high-quant", "Q4_K_M", "mlx", 4.0, "Apache-2.0 interactive portal"),
    LLMPort("JuniorFable", "behavioral", "n/a", "none", 0.0, "SafetyClassifier rigidity"),
    LLMPort("JuniorKimiK3-edge", "edge-moe", "pruned-ternary", "mlx", 8.0, "Never pull full 1.5TB"),
    LLMPort("Qwen-local", "high-quant", "Q4_K_M", "gguf", 20.0, "Quality baseline, not BitNet-native"),
]


def list_ports() -> list[dict]:
    return [asdict(p) for p in PORTS]


def pick(task: str, ram_gb: float) -> LLMPort:
    t = (task or "").lower()
    if "field" in t or "beta" in t or "access" in t:
        return PORTS[0]
    if "safety" in t or "fable" in t:
        return PORTS[3]
    if ram_gb < 6:
        return PORTS[1] if ram_gb >= 2 else PORTS[0]
    if "chat" in t or "gemma" in t:
        return PORTS[2]
    return PORTS[0]
