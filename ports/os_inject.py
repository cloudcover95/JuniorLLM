"""JuniorOS hardware inject. Probe plus class ticket. No launch."""
from __future__ import annotations

import json
from pathlib import Path

from ports.inject_hw import inject
from ports.os_infra import probe

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "os_inject.jsonl"


def run(hw: str = "cpu") -> dict:
    infra = probe()
    row = inject("JuniorOS", hw)
    body = {
        "protocol": "goldend-osai-omega/1",
        "os": "JuniorOS",
        "hw": row["hw"],
        "llama_ready": row["llama_ready"],
        "models": infra["models"],
        "asahi": infra["asahi"],
        "launch": False,
        "fetch_driver": False,
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"hw": body["hw"], "ready": body["llama_ready"]}) + "\n")
    return body
