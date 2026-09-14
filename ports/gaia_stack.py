"""One Gaia pass: gate, spine, I2_S, SVD, terrain OBJ, UE5 JSON, llama/bitnet probe."""
from __future__ import annotations

import os
from pathlib import Path

from junior_bitnet.i2s import pack as i2s_pack
from ports.gaia import spine
from ports.gaia_terrain import write as terrain_write
from ports.svd_tick import tick as svd_tick
from ports.ue5_port import write as ue5_write
from rails.linux.bitnet_cpp import plan as bitnet_plan
from rails.linux.llama import plan as llama_plan


def run(note: str = "gaia terrain", out_dir: Path | None = None, n: int = 32) -> dict:
    out_dir = out_dir or (Path.home() / ".juniorhome" / "gaia_mesh")
    who = spine(note)
    trits = [int(b["trit"]) for b in who.get("bolts") or []]
    blob = i2s_pack(trits or [0])
    svd = svd_tick(48, 30)
    terr = terrain_write(out_dir, note, n) if who.get("ok") else {}
    ue = ue5_write(out_dir, note, n) if who.get("ok") else {"launch": False}
    llama = llama_plan()
    bitn = bitnet_plan()
    return {
        "ok": bool(who.get("ok")),
        "who": who.get("who"),
        "gamma": who.get("gamma"),
        "i2s_bytes": len(blob),
        "i2s_hex": blob.hex(),
        "svd": svd,
        "terrain": {"obj": terr.get("obj"), "faces": terr.get("faces")},
        "ue5": {"json": ue.get("ue5_json"), "launch": False},
        "llama_ready": bool(llama.get("ready")),
        "bitnet_cpp_ready": bool(bitn.get("ready")),
        "mlx": "BitNet-mlx owns Metal",
        "usgs_fetch": False,
        "gemini_macro": False,
        "download": False,
        "gguf": os.environ.get("JUNIOR_GGUF"),
    }
