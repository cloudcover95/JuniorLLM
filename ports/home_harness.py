"""JuniorLLM side of Home. Live = trit + route. Future = probe only."""
from __future__ import annotations

import os
from pathlib import Path

from junior_bitnet.winsor import pack
from ports.gaia import spine as gaia_spine
from ports.juniorosai_harness import run as osai_run
from ports.xai_gate import status as xai_status

BIND = {"hook": "127.0.0.1:8770", "ui": "127.0.0.1:8771", "i2sd": "127.0.0.1:8767", "portrait": "127.0.0.1:8771/gaia.html"}


def _surface(task: str, watts: float = 45.0) -> dict:
    t = (task or "").lower()
    if any(k in t for k in ("gaia", "companion", "portrait", "goldend")):
        return {"surface": "gaia-portrait", "port": "JuniorGaia", "launch": False}
    if "ue5" in t or "unreal" in t or watts >= 80:
        return {"surface": "ue5", "port": "JuniorAstra", "launch": False}
    if any(k in t for k in ("frameforge", "ff2d", "canvas")):
        return {"surface": "frameforge2d", "port": "JuniorAstra", "launch": False}
    if any(k in t for k in ("flagstaff", "fieldcore", "stonefield")):
        return {"surface": "fieldcore-intent", "port": "JuniorBitNetFieldCore", "launch": False}
    return {"surface": "home-clock", "port": "JuniorAstra", "launch": False}


def harness(task: str = "home clock", watts: float = 45.0) -> dict:
    gguf = os.environ.get("JUNIOR_GGUF", "")
    gguf_ok = bool(gguf) and Path(gguf).is_file()
    q = pack([0.01, -2.0, 0.4, 0.2])
    osai = osai_run(task, 8.0)
    xai = xai_status()
    surf = _surface(task, watts)
    g = gaia_spine(task) if surf["port"] == "JuniorGaia" else None
    return {
        "live": ["trit_tick", "os_route", "T12", "gaia"],
        "bot_next": "T13",
        "winsor_ok": q.get("ok"),
        "gamma": q.get("gamma"),
        "surface": surf,
        "gaia": g,
        "omega_mesh": "stub",
        "ue5_launch": False,
        "ue5_env": os.environ.get("JUNIOR_UE5") == "1",
        "gguf_exists": gguf_ok,
        "llama_ready": gguf_ok,
        "download": False,
        "kernel_patch": False,
        "asahi_probe_only": True,
        "xai_call": xai.get("call", False),
        "bind": BIND,
        "osai_mode": osai.get("mode"),
        "osai_download": osai.get("download"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(harness("gaia they home"), indent=2))
