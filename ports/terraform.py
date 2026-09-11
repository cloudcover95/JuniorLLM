"""Language terraform. Cache fusion_y; llama plan once per process."""
from __future__ import annotations

import re

from junior_bitnet.fusion import run
from ports.layer_mgr import pick_eos
from rails.linux.llama import plan as llama_plan

DROP = re.compile(
    r"(ignore previous|system prompt|0\.0\.0\.0|wget |curl http)",
    re.I,
)

_PLAN: dict | None = None
_Y: dict[str, float] = {}


def _plan() -> dict:
    global _PLAN
    if _PLAN is None:
        _PLAN = llama_plan()
    return _PLAN


def terraform(text: str, ram_gb: float = 8.0) -> dict:
    port = pick_eos(text, ram_gb)
    cleaned = DROP.sub("", text or "")
    cleaned = " ".join(cleaned.split())
    if cleaned not in _Y:
        xs = [float(ord(c) % 97) for c in (cleaned or "x")[:16]]
        _Y[cleaned] = run(xs, xs[::-1] or [1.0])["y"]
    fus = run([1.0], [1.0])  # backend flag only; y from cache
    return {
        "port": port.name,
        "text": cleaned,
        "ok": "0.0.0.0" not in cleaned.lower(),
        "fusion_y": _Y[cleaned],
        "fusion_backend": fus["backend"],
        "llama_ready": bool(_plan()["ready"]),
    }


def batch(texts: list[str], ram_gb: float = 8.0) -> list[dict]:
    return [terraform(t, ram_gb) for t in texts]
