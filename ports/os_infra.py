"""OS infra probe. No driver fetch. No model pull."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "os_infra.jsonl"
MODELS = ("gguf", "mlx", "bitnet.cpp", "ticket")


def probe() -> dict:
    gguf = os.environ.get("JUNIOR_GGUF")
    body = {
        "protocol": "goldend-osai-omega/1",
        "models": {
            "gguf": bool(gguf and Path(gguf).is_file()),
            "mlx": False,
            "bitnet.cpp": bool(shutil.which("llama-cli") or os.environ.get("JUNIOR_LLAMA")),
            "ticket": True,
        },
        "asahi": {
            "driver": "honeykrisp-class",
            "fetch_driver": False,
            "iso": False,
            "submit": False,
        },
        "hw": ["cpu", "mlx", "cuda", "vulkan", "asahi"],
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"gguf": body["models"]["gguf"], "fetch_driver": False}) + "\n")
    return body
