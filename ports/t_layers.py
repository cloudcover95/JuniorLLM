"""Ship ladder. T2 is larger local mesh + trit. Not 1024² MLX parquet."""
from __future__ import annotations

from ports.fieldcore_spine import expand
from ports.gaia_proto import handshake
from ports.svd_tick import tick

LAYERS = {
    "T0": {"n": 0, "job": "dash-viewport", "svd": False, "mlx": False, "parquet": False},
    "T1": {"n": 32, "job": "dash-viewport", "svd": "auto", "mlx": False, "parquet": False},
    "T2": {"n": 48, "job": "gaia-spine", "svd": "power", "mlx": False, "parquet": False},
    "T3": {"n": 0, "job": "agi-capsule", "svd": False, "gguf": "if-on-disk", "mlx": False, "parquet": False},
}


def run(layer: str = "T0", note: str = "home dash") -> dict:
    layer = layer.upper() if layer else "T0"
    spec = LAYERS.get(layer) or LAYERS["T0"]
    env = handshake(note, job=spec["job"])
    extra = {}
    if spec.get("svd"):
        extra["svd"] = tick(int(spec["n"] or 32), 8, mode="power" if spec["svd"] == "power" else None)
    if layer == "T2":
        extra["spine"] = expand(n=48, k=8)
    return {
        "layer": layer,
        "spec": spec,
        "protocol": env.get("protocol"),
        "schema_ok": env.get("schema_ok"),
        "gamma": (env.get("note") or {}).get("gamma"),
        "mlx": False,
        "parquet": False,
        **extra,
    }
