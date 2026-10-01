"""Route Omega, AGI, StoneField, web3node. No weight pull."""
from __future__ import annotations

import json
from pathlib import Path

from ports.registry import pick
from ports.stonefield import find, list_public

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "tritquant.jsonl"


def run(task: str) -> dict:
    port = pick(task, 0.0)
    extra: dict = {}
    if port.name == "JuniorStoneField":
        extra["hit"] = find(task)
        extra["public_n"] = len(list_public())
    if port.name == "web3node":
        extra["mesh"] = str(MESH)
        extra["mesh_exists"] = MESH.exists()
    if port.name == "JuniorOmega":
        extra["launch"] = False
    if port.name == "AGI_SDK":
        extra["weights"] = False
    return {
        "port": port.name,
        "kind": port.kind,
        "quant": port.quant,
        "backend": port.backend,
        "download_gb": port.max_download_gb,
        "model_pull": False,
        **extra,
    }


if __name__ == "__main__":
    print(json.dumps(run("omega mesh"), indent=2))
