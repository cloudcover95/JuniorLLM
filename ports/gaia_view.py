"""Dash viewport. Portrait/landscape is orientation, not the companion face."""
from __future__ import annotations

from ports.gaia_dash import act

MAX = {"portrait": (1080, 1920), "landscape": (1920, 1080)}


def view(note: str = "home dash", orient: str = "landscape", scale: float = 1.0) -> dict:
    o = "portrait" if str(orient).lower().startswith("port") else "landscape"
    s = min(1.0, max(0.35, float(scale)))
    w, h = MAX[o]
    w, h = int(w * s), int(h * s)
    dash = act(note)
    return {
        "ok": dash.get("ok"),
        "orient": o,
        "scale": s,
        "px": [w, h],
        "max_px": list(MAX[o]),
        "omega": {"job": "dash-viewport", "w": w, "h": h, "mesh": "obj"},
        "agi": {"port": "JuniorGaia", "compute": "JuniorLLM"},
        "area": dash.get("area"),
        "port": dash.get("port"),
        "i2s_hex": dash.get("i2s_hex"),
        "gamma": dash.get("gamma"),
        "optional": True,
        "ue5_launch": False,
    }
