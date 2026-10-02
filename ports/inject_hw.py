"""Hardware class injection. Launch only if llama plan is ready. This call does not launch."""
from __future__ import annotations

import json
from pathlib import Path

from rails.linux.llama import plan

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "inject_hw.jsonl"
CLASSES = ("cpu", "mlx", "cuda", "vulkan", "asahi")


def inject(note: str = "JuniorOS", hw: str = "cpu") -> dict:
    lp = plan()
    body = {
        "protocol": "goldend-osai-omega/1",
        "note": note[:160],
        "hw": hw if hw in CLASSES else "cpu",
        "classes": list(CLASSES),
        "llama_ready": bool(lp.get("ready")),
        "launch": False,
        "model_pull": False,
        "bind": "127.0.0.1",
        "fallback": lp.get("fallback"),
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"hw": body["hw"], "ready": body["llama_ready"]}) + "\n")
    return body
