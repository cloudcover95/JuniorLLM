"""Every Home-suite core is an OSai port. Download 0 unless noted."""
from __future__ import annotations

from ports.registry import LLMPort, pick

CORES = [
    LLMPort("JuniorHome", "suite", "ternary-1.58", "mesh", 0.0, "Operator box."),
    LLMPort("JuniorOS", "os-rails", "ternary-1.58", "junioros", 0.0, "Local OS rails."),
    LLMPort("JuniorOSai", "bitnet-native", "ternary-1.58", "flagstaff", 0.0, "Agent layer."),
    LLMPort("JuniorLLM", "ticket", "ternary-1.58", "ports", 0.0, "This repo."),
    LLMPort("JuniorOmega", "mesh-cad", "ternary-1.58", "omega-job", 0.0, "CAD job. launch false."),
    LLMPort("AGI_SDK", "agent-port", "ternary-1.58", "agi-sdk", 0.0, "Agent port list."),
    LLMPort("JuniorAGI_SDK", "agent-port", "ternary-1.58", "agi-sdk", 0.0, "AGI fork. No weight pull."),
    LLMPort("FieldCore", "jsonl-intent", "ternary-1.58", "fieldcore", 0.0, "n=32 cap 64."),
    LLMPort("Flagstaff", "vote", "ternary-1.58", "6-and", 0.0, "6-vote AND."),
    LLMPort("Gaia", "spine", "ternary-1.58", "gaia", 0.0, "Terrain ticket."),
    LLMPort("Goldend", "engine", "ternary-1.58", "goldens", 0.0, "Engine species."),
    LLMPort("JuniorDeck", "audio-ticket", "ternary-1.58", "osai-join", 0.0, "Board + audio drop."),
    LLMPort("BitNet-mlx", "kernel", "ternary-1.58", "mlx", 0.0, "Metal kernels. No second packer."),
    LLMPort("JuniorMemSys-Suite", "notes", "ternary-1.58", "memsys", 0.0, "Note store."),
    LLMPort("web3node", "trit-quant", "ternary-1.58", "absmean", 0.0, "AbsMean pack. Not a chain address."),
    LLMPort("crispy-mouse", "input", "ternary-1.58", "hid", 0.0, "Sparse input."),
    LLMPort("JuniorStock", "quant-node", "ternary-1.58", "stock", 0.0, "Quant node."),
    LLMPort("stocksnode", "quant-node", "ternary-1.58", "stock", 0.0, "Preview node. JuniorStock is live."),
    LLMPort("JuniorQuant", "quant-math", "ternary-1.58", "quant", 0.0, "45 W quant pack."),
    LLMPort("JuniorStoneField", "field-notes", "ternary-1.58", "stonefield", 0.0, "Local seed. No scrape."),
    LLMPort("JuniorClimbs", "gym", "ternary-1.58", "climbs", 0.0, "Gym pack."),
    LLMPort("JuniorEngrTools", "calc", "ternary-1.58", "engr", 0.0, "Calc pack."),
    LLMPort("JuniorPoker", "table", "ternary-1.58", "felt", 0.0, "Local felt. live false."),
    LLMPort("JuniorSOL", "ledger", "ternary-1.58", "sol", 0.0, "Live ledger pack."),
    LLMPort("JuniorSolana", "ledger-archive", "ternary-1.58", "sol", 0.0, "Archive. JuniorSOL is the name."),
    LLMPort("JuniorTeqp", "property", "ternary-1.58", "teqp", 0.0, "Property pack."),
    LLMPort("JuniorFetch", "search", "ternary-1.58", "fetch", 0.0, "Local file search."),
    LLMPort("JuniorDrive", "sim", "ternary-1.58", "drive", 0.0, "Sim ticket."),
    LLMPort("JuniorCoach", "roster", "ternary-1.58", "coach", 0.0, "Local roster pack."),
    LLMPort("JuniorPiThon", "sbc", "ternary-1.58", "pi", 0.0, "SBC host pack."),
    LLMPort("JuniorPiPython", "runtime", "ternary-1.58", "python", 0.0, "Ternary runtime pack."),
    LLMPort("JuniorPython-Suite", "tools", "ternary-1.58", "python", 0.0, "Tool registry."),
    LLMPort("FrameForge", "scene", "ternary-1.58", "forge", 0.0, "Arena pack. ue5_launch false."),
    LLMPort("FrameForge2D", "scene", "ternary-1.58", "forge2d", 0.0, "2D pack."),
]

BY = {c.name.lower(): c for c in CORES}


def route(task: str, ram_gb: float = 0.0):
    t = (task or "").lower()
    for name, port in BY.items():
        if name in t:
            return port
    return pick(t, ram_gb)


def list_cores() -> list[dict]:
    return [{"name": c.name, "kind": c.kind, "quant": c.quant, "download_gb": c.max_download_gb} for c in CORES]
