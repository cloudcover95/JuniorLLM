"""Junior custom LLM ports — local / high-quant first."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LLMPort:
    name: str
    kind: str
    quant: str
    backend: str
    max_download_gb: float
    notes: str


PORTS = [
    LLMPort("JuniorBitNetFieldCore", "bitnet-native", "ternary-1.58", "torch-ternary", 0.0, "Crowd/field scorer"),
    LLMPort("BitNet-2B4T", "bitnet-native", "I2_S", "bitnet.cpp", 1.5, "microsoft/bitnet-b1.58-2B-4T-gguf"),
    LLMPort("JuniorGemma4-4B", "high-quant", "Q4_K_M", "mlx", 4.0, "Apache-2.0 interactive portal"),
    LLMPort("JuniorFable", "behavioral", "n/a", "none", 0.0, "SafetyClassifier rigidity"),
    LLMPort("JuniorAstra", "agent-runtime", "n/a", "astra-runner", 0.0, "Open Astra durable Work + ContextPipe; BYO LLM"),
    LLMPort(
        "JuniorAstraReason",
        "high-quant",
        "Q4_K_M+ternary-loops",
        "mlx-or-gguf",
        8.0,
        "Astra-class stand-in: Gemma4/Qwen Q4 + rigid IQ layers + MemSys palace. Not GPT-6 weights.",
    ),
    LLMPort("JuniorKimiK3-edge", "edge-moe", "pruned-ternary", "mlx", 8.0, "Never pull full 1.5TB"),
    LLMPort("Qwen-local", "high-quant", "Q4_K_M", "gguf", 8.0, "Quality baseline under 8GB cap"),
    LLMPort(
        "JuniorBitNetDraft",
        "bitnet-native",
        "ternary-1.58",
        "rigid-iq",
        0.0,
        "CAD sidecar / title-block / layer intent. Not a generic chat hook.",
    ),
]


def list_ports() -> list[dict]:
    return [asdict(p) for p in PORTS]


def _named(name: str) -> LLMPort:
    return next(p for p in PORTS if p.name == name)


def pick(task: str, ram_gb: float) -> LLMPort:
    t = (task or "").lower()
    if any(k in t for k in ("cad", "dxf", "dwg", "drawing", "title block", "omega", "draft")):
        return _named("JuniorBitNetDraft")
    if "astra-class" in t or "astra reason" in t or "reason" in t or "qwen" in t:
        return _named("JuniorAstraReason")
    if "durable" in t or "checkpoint" in t or t.strip() == "astra" or t.startswith("astra ") and "class" not in t:
        return _named("JuniorAstra")
    if "field" in t or "beta" in t or "access" in t:
        return PORTS[0]
    if "safety" in t or "fable" in t:
        return _named("JuniorFable")
    if ram_gb < 6:
        return PORTS[1] if ram_gb >= 2 else PORTS[0]
    if "chat" in t or "gemma" in t:
        return _named("JuniorGemma4-4B")
    return PORTS[0]
