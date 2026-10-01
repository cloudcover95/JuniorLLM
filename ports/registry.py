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
    LLMPort("JuniorOSai", "bitnet-native", "ternary-1.58", "flagstaff", 0.0, "Home kernel + Flagstaff. Ticket only."),
    LLMPort("JuniorOmega", "mesh-cad", "ternary-1.58", "omega-in", 0.0, "Sheet ingest. launch false."),
    LLMPort("AGI_SDK", "agent-port", "ternary-1.58", "agi-sdk", 0.0, "Agent job list. No weight pull."),
    LLMPort("JuniorStoneField", "field-note", "ternary-1.58", "stonefield", 0.0, "Local seed. Public nodes only."),
    LLMPort("web3node", "trit-quant", "ternary-1.58", "absmean", 0.0, "AbsMean ticket. Not a chain address."),
    LLMPort("JuniorBitNetFieldCore", "bitnet-native", "ternary-1.58", "torch-ternary", 0.0, "Crowd/field scorer"),
    LLMPort("BitNet-2B4T", "bitnet-native", "I2_S", "bitnet.cpp", 1.5, "local GGUF header only if already on disk"),
    LLMPort("JuniorGemma4-4B", "high-quant", "Q4_K_M", "mlx", 4.0, "Apache-2.0 interactive portal"),
    LLMPort("JuniorFable", "behavioral", "n/a", "none", 0.0, "SafetyClassifier rigidity"),
    LLMPort("JuniorAstra", "agent-runtime", "n/a", "astra-runner", 0.0, "Open Astra durable Work + ContextPipe; BYO LLM"),
    LLMPort(
        "JuniorAstraReason",
        "high-quant",
        "Q4_K_M+ternary-loops",
        "mlx-or-gguf",
        8.0,
        "Astra-class stand-in. Not a weight download from this port.",
    ),
    LLMPort("JuniorKimiK3-edge", "edge-moe", "pruned-ternary", "mlx", 8.0, "Never pull full 1.5TB"),
    LLMPort("Qwen-local", "high-quant", "Q4_K_M", "gguf", 8.0, "Quality baseline under 8GB cap"),
    LLMPort("JuniorBitNetDraft", "bitnet-native", "ternary-1.58", "rigid-iq", 0.0, "CAD sidecar. Not a generic chat hook."),
    LLMPort("JuniorGaia", "companion-spine", "ternary-1.58", "home-portrait", 0.0, "Local HUD. Not a vendor companion."),
    LLMPort("JuniorDeck", "audio-ticket", "ternary-1.58", "osai-join", 0.0, "Deck inbox + tritquant join."),
]


def list_ports() -> list[dict]:
    return [asdict(p) for p in PORTS]


def _named(name: str) -> LLMPort:
    return next(p for p in PORTS if p.name == name)


def pick(task: str, ram_gb: float) -> LLMPort:
    t = (task or "").lower()
    if any(k in t for k in ("omega", "blender", "obj", "mesh job")):
        return _named("JuniorOmega")
    if any(k in t for k in ("agi", "agent sdk", "agi_sdk")):
        return _named("AGI_SDK")
    if any(k in t for k in ("stone", "boulder", "crag", "stonefield")):
        return _named("JuniorStoneField")
    if any(k in t for k in ("web3", "absmean", "trit quant", "web3node")):
        return _named("web3node")
    if any(k in t for k in ("deck", "audio", "tritquant", "osai join")):
        return _named("JuniorDeck")
    if any(k in t for k in ("osai", "flagstaff", "home kernel")):
        return _named("JuniorOSai")
    if any(k in t for k in ("gaia", "companion", "portrait", "goldend")):
        return _named("JuniorGaia")
    if any(k in t for k in ("cad", "dxf", "dwg", "drawing", "title block", "draft")):
        return _named("JuniorBitNetDraft")
    if "astra-class" in t or "astra reason" in t or "qwen" in t:
        return _named("JuniorAstraReason")
    if "durable" in t or "checkpoint" in t or t.strip() == "astra":
        return _named("JuniorAstra")
    if "field" in t or "beta" in t or "access" in t:
        return _named("JuniorBitNetFieldCore")
    if "safety" in t or "fable" in t:
        return _named("JuniorFable")
    if ram_gb < 6:
        return _named("BitNet-2B4T") if ram_gb >= 2 else _named("JuniorOSai")
    if "chat" in t or "gemma" in t:
        return _named("JuniorGemma4-4B")
    return _named("JuniorOSai")
