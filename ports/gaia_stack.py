"""Flagstaff prior → I2_S → SVD reconstruct OBJ → UE5 JSON → honest probes."""
from __future__ import annotations

import os
from pathlib import Path

from junior_bitnet.i2s import pack as i2s_pack
from ports.gaia import spine
from ports.svd_tick import tick as svd_tick
from ports.terrain_spine import OSS, fetch_oss_flagstaff_mesh
from ports.ue5_port import scene
from rails.linux.bitnet_cpp import plan as bitnet_plan
from rails.linux.llama import plan as llama_plan


def _obj(mesh: list[list[float]]) -> str:
    n = len(mesh)
    lines = ["# Gaia Flagstaff SVD reconstruct", f"# n={n}", "o GaiaTerrain"]
    for i in range(n):
        for j in range(n):
            lines.append(f"v {i} {mesh[i][j]:.3f} {j}")
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j + 1
            b, c, d = a + 1, a + n, a + n + 1
            lines.append(f"f {a} {b} {d} {c}")
    return "\n".join(lines) + "\n"


def run(note: str = "gaia they terrain", out_dir: Path | None = None, n: int = 48) -> dict:
    out_dir = out_dir or (Path.home() / ".juniorhome" / "gaia_mesh")
    out_dir.mkdir(parents=True, exist_ok=True)
    who = spine(note)
    raw = fetch_oss_flagstaff_mesh(n)
    svd = svd_tick(n, 30, raw)
    mesh = svd.get("mesh") or raw
    obj_path = out_dir / f"flagstaff_{n}x{n}.obj"
    if who.get("ok"):
        obj_path.write_text(_obj(mesh), encoding="utf-8")
    ue = scene(note, n)
    ue_path = out_dir / "gaia_ue5.json"
    import json

    ue["source"] = str(obj_path)
    ue_path.write_text(json.dumps(ue, indent=2), encoding="utf-8")
    trits = [int(b["trit"]) for b in who.get("bolts") or []]
    blob = i2s_pack(trits or [0])
    llama = llama_plan()
    bitn = bitnet_plan()
    return {
        "ok": bool(who.get("ok")),
        "who": who.get("who"),
        "gamma": who.get("gamma"),
        "i2s_bytes": len(blob),
        "svd": {k: svd[k] for k in ("backend", "n", "k", "energy", "ms") if k in svd},
        "terrain": {"obj": str(obj_path), "faces": (n - 1) ** 2, "z0": mesh[0][0] if mesh else None},
        "ue5": {"json": str(ue_path), "launch": False},
        "llama_ready": bool(llama.get("ready")),
        "bitnet_cpp_ready": bool(bitn.get("ready")),
        "oss": OSS,
        "usgs_fetch": False,
        "gemini_macro": False,
        "download": False,
    }
