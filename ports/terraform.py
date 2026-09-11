"""Language terraform through local trit ports + fused infer sidecar."""
from __future__ import annotations

import re

from junior_bitnet.fusion import run
from ports.layer_mgr import pick_eos
from rails.linux.llama import plan as llama_plan

DROP = re.compile(
    r"(ignore previous|system prompt|0\.0\.0\.0|wget |curl http)",
    re.I,
)


def terraform(text: str, ram_gb: float = 8.0) -> dict:
    port = pick_eos(text, ram_gb)
    cleaned = DROP.sub("", text or "")
    cleaned = " ".join(cleaned.split())
    xs = [float(ord(c) % 97) for c in (cleaned or "x")[:16]]
    fus = run(xs, xs[::-1] or [1.0])
    return {
        "port": port.name,
        "text": cleaned,
        "ok": "0.0.0.0" not in cleaned.lower(),
        "fusion_y": fus["y"],
        "fusion_backend": fus["backend"],
        "llama_ready": llama_plan()["ready"],
    }
