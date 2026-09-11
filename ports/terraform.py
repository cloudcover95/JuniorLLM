"""Language terraform through local trit ports. No cloud LLM."""
from __future__ import annotations

import re

from ports.layer_mgr import pick_eos

DROP = re.compile(
    r"(ignore previous|system prompt|0\.0\.0\.0|wget |curl http)",
    re.I,
)


def terraform(text: str, ram_gb: float = 8.0) -> dict:
    port = pick_eos(text, ram_gb)
    cleaned = DROP.sub("", text or "")
    cleaned = " ".join(cleaned.split())
    return {
        "port": port.name,
        "text": cleaned,
        "ok": "0.0.0.0" not in cleaned.lower(),
    }
