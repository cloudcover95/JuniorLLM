"""Home onboarding for a local GGUF. No pull."""
from __future__ import annotations

from pathlib import Path

from ports.fieldcore_spine import expand
from ports.gguf_quants import report
from ports.gguf_t3 import find, run as t3

MODELS = Path.home() / ".juniorhome" / "models"
MESH = Path.home() / ".juniorhome" / "gaia_mesh"


def onboard() -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    MESH.mkdir(parents=True, exist_ok=True)
    hit = find()
    return {
        "models_dir": str(MODELS),
        "mesh_dir": str(MESH),
        "gguf": t3("onboard"),
        "quants": report(),
        "fieldcore": expand(n=32, k=8),
        "how": "copy a .gguf into ~/.juniorhome/models then rerun",
        "rust_ffi": False,
        "download": False,
    }
