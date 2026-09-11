"""Custom localLLM for the computer at hand. Written into Home, not a repo clone."""
from __future__ import annotations

import json
from pathlib import Path

from ports.inject import write_vault
from ports.terraform import terraform
from rails.linux.backend import probe
from rails.linux.llama import plan as llama_plan


def spec() -> dict:
    p = probe()
    if p.asahi:
        name, runtime, ctx = "JuniorHome-Asahi", "mlx", 4096
    elif p.accel == "mlx":
        name, runtime, ctx = "JuniorHome-MLX", "mlx", 4096
    elif p.accel == "cuda":
        name, runtime, ctx = "JuniorHome-CUDA", "llama.cpp", 8192
    elif p.machine.lower() in {"aarch64", "arm64"}:
        name, runtime, ctx = "JuniorHome-ARM", "i2sd", 2048
    else:
        name, runtime, ctx = "JuniorHome-CPU", "i2sd", 2048
    lp = llama_plan()
    if lp["ready"]:
        runtime = "llama.cpp"
    return {
        "name": name,
        "runtime": runtime,
        "ctx": ctx,
        "accel": p.accel,
        "machine": p.machine,
        "asahi": p.asahi,
        "llama_ready": lp["ready"],
        "port": "JuniorBitNetFieldCore",
    }


def build(vault: Path, note: str = "home local llm") -> dict:
    vault = Path(vault)
    vault.mkdir(parents=True, exist_ok=True)
    card = spec()
    tf = terraform(note)
    card["tf"] = {"text": tf["text"], "fusion_y": tf["fusion_y"], "ok": tf["ok"]}
    (vault / "local_llm.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    inj = write_vault(note, vault)
    card["inject_ok"] = inj.get("ok")
    return card
