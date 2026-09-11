"""Local Flagstaff LLM context — fill each locked tier to budget, pack I2_S."""
from __future__ import annotations

import json
from pathlib import Path

from junior_bitnet.i2s import pack
from junior_bitnet.math import absmean
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
    return t if len(t) <= n else t[: n - 1] + "\u2026"


def _i2s(text: str) -> str:
    xs = [float(ord(c) % 97) for c in (text or "x")[:64]]
    z, _ = absmean(xs)
    return pack(z).hex()


def assemble(ask: str, lock: Path | None = None) -> dict:
    locked = []
    if lock and Path(lock).is_file():
        locked = list(json.loads(Path(lock).read_text(encoding="utf-8")).get("locked") or [])
    tf = terraform(ask)
    sheets = ""
    try:
        from junior_bitnet.coolstore import build

        tab = build()
        sheets = json.dumps(tab["fluids"], separators=(",", ":"))
    except Exception:
        pass
    layers = {}
    for name, cap in BUDGET.items():
        if locked and name not in locked:
            continue
        raw = " ".join(x for x in (ask, tf["text"], sheets, name) if x)
        ctx = _fit(raw, cap)
        layers[name] = {
            "budget": cap,
            "ctx": ctx,
            "n": len(ctx),
            "fill": round(len(ctx) / cap, 3),
            "i2s": _i2s(ctx),
        }
    return {
        "port": pick_eos(ask or "flagstaff", 8).name,
        "layers": layers,
        "total": sum(v["n"] for v in layers.values()),
        "tf": tf,
    }
