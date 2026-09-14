"""GGUF families vs Junior 1.58 wire. No download."""
from __future__ import annotations

from ports.gguf_t3 import find, header, run as t3

FAMILIES = {
    "classic": ["Q4_0", "Q5_0", "Q8_0", "F16", "F32"],
    "k": ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    "iq": ["IQ1_S", "IQ2_XXS", "IQ3_XXS", "IQ4_NL"],
    "tq": ["TQ1_0", "TQ2_0"],
    "bitnet": ["I2_S", "TL1", "TL2"],
}

PICK = {
    "T0": "winsor-p95-python",
    "T3_file": "name-suffix",
    "prefer_local": ["I2_S", "TQ1_0", "TQ2_0", "Q4_K_M"],
    "never_pull": True,
}


def guess(path: str) -> str:
    u = path.upper()
    for fam, names in FAMILIES.items():
        for n in names:
            if n in u:
                return n
    return "unknown"


def report() -> dict:
    p = find()
    h = header(p) if p else {"ok": False, "reason": "no-local-gguf"}
    kind = guess(h.get("path") or "") if h.get("ok") else None
    return {
        "families": FAMILIES,
        "pick": PICK,
        "file": h,
        "guess": kind,
        "junior_wire": "{-1,0,1} winsor-p95 / I2_S pack",
        "gguf_tq": "llama.cpp TQ1_0/TQ2_0 ≈ ternary weights, not our note pack",
        "download": False,
    }
