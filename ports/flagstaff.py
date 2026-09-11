"""Local Flagstaff LLM context — budget chars per locked Layer-1 tier."""
from __future__ import annotations

import json
from pathlib import Path

from ports.layer_mgr import pick_eos
from ports.terraform import terraform

BUDGET = {
    "probe": 256,
    "i2s": 128,
    "teqp": 512,
    "ports": 256,
    "terraform": 1024,
}


def _fit(text: str, n: int) -> str:
    t = text or ""
    return t if len(t) <= n else t[: n - 1] + "…"


def assemble(ask: str, lock: Path | None = None) -> dict:
    locked = []
    if lock and Path(lock).is_file():
        locked = list(json.loads(Path(lock).read_text(encoding="utf-8")).get("locked") or [])
    tf = terraform(ask)
    sheets = ""
    try:
        from junior_bitnet.coolstore import build, props_si

        tab = build()
        sheets = f"FIELD {props_si(tab,'PHASE','FIELD')} NIGHT {props_si(tab,'PHASE','NIGHT')}"
    except Exception:
        pass
    layers = {}
    for name, cap in BUDGET.items():
        if locked and name not in locked:
            continue
        raw = " ".join(x for x in (ask, tf["text"], sheets, name) if x)
        layers[name] = {"budget": cap, "ctx": _fit(raw, cap), "n": min(len(raw), cap)}
    return {
        "port": pick_eos(ask or "flagstaff", 8).name,
        "layers": layers,
        "total": sum(v["n"] for v in layers.values()),
        "tf": tf,
    }
