"""T1: handshake + 32² spine + honest box probe. MLX only if import works."""
from __future__ import annotations

from ports.fieldcore_spine import expand
from ports.gaia_proto import handshake
from ports.layer_mgr import pick_eos
from rails.linux.backend import probe


def run(note: str = "home dash") -> dict:
    box = probe()
    env = handshake(note, job="dash-viewport")
    spine = expand(n=32, k=8)
    port = pick_eos(note, 8)
    accel = box.accel
    return {
        "layer": "T1",
        "ok": bool(env.get("schema_ok")),
        "protocol": env.get("protocol"),
        "port": port.name,
        "area": (env.get("note") or {}).get("area"),
        "gamma": (env.get("note") or {}).get("gamma"),
        "box": {
            "machine": box.machine,
            "system": box.system,
            "accel": accel,
            "kernel": box.kernel,
            "asahi": box.asahi,
            "notes": box.notes,
            "ambiguous": accel == "cpu",
        },
        "spine": {"n": spine.get("n"), "energy": spine.get("energy"), "backend": spine.get("backend"), "path": spine.get("path")},
        "mlx": accel == "mlx",
        "parquet": False,
        "ue5_launch": False,
        "download": False,
    }
