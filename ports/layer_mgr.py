"""Active layer manager — ports keyed off JuniorTeqp phase, then task.

sparse   → FieldCore (cheap trit)
coexist  → Draft if CAD else FieldCore
mixed    → Gemma/Qwen if RAM else FieldCore
dense    → AstraReason (rigid loops)
Task keywords still win (cad/fable/astra).
"""
from __future__ import annotations

from ports.registry import LLMPort, _named, pick

PHASE_PORT = {
    "sparse": "JuniorBitNetFieldCore",
    "coexist": "JuniorBitNetFieldCore",
    "mixed": "JuniorGemma4-4B",
    "dense": "JuniorAstraReason",
}


def _phase(fluid: str, table: dict | None) -> str:
    if table:
        try:
            from junior_bitnet.coolstore import props_si
            from junior_bitnet.teqp import T_C

            return str(props_si(table, "PHASE", fluid, T_C))
        except Exception:
            pass
    try:
        from junior_bitnet.refprop import Library

        return str(Library().props_si("PHASE", fluid))
    except Exception:
        return "coexist"


def pick_eos(task: str, ram_gb: float, fluid: str = "NIGHT", table: dict | None = None) -> LLMPort:
    t = (task or "").lower()
    if any(k in t for k in ("cad", "dxf", "dwg", "drawing", "omega", "draft")):
        return _named("JuniorBitNetDraft")
    if "safety" in t or "fable" in t:
        return _named("JuniorFable")
    if "durable" in t or (t.strip() == "astra"):
        return _named("JuniorAstra")
    phase = _phase(fluid, table)
    name = PHASE_PORT.get(phase, "JuniorBitNetFieldCore")
    if name == "JuniorGemma4-4B" and ram_gb < 6:
        name = "JuniorBitNetFieldCore"
    if name == "JuniorAstraReason" and ram_gb < 8:
        name = "JuniorBitNetFieldCore" if ram_gb < 4 else pick(task or "reason", ram_gb).name
        if ram_gb < 8:
            name = "JuniorBitNetFieldCore" if phase != "dense" else "JuniorAstraReason"
    return _named(name)


def report(table: dict | None = None, ram_gb: float = 8.0) -> list[dict]:
    fluids = ["SPARSE", "NIGHT", "FIELD", "ABSMEAN", "ZK", "DENSE"]
    out = []
    for f in fluids:
        p = pick_eos("", ram_gb, f, table)
        out.append({"fluid": f, "phase": _phase(f, table), "port": p.name})
    return out
