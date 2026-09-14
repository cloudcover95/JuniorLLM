"""JuniorGaia system. One object for Home / Omega / AGI_SDK."""
from __future__ import annotations

from ports.gaia import load_who, spine
from ports.gaia_dash import act
from ports.gaia_view import view
from rails.linux.bitnet_cpp import plan as bitnet_plan
from rails.linux.llama import plan as llama_plan

LAYERS = ("who", "spine", "dash", "view", "walk", "mesh")


def system(note: str = "home dash", *, orient: str = "landscape", scale: float = 1.0) -> dict:
    who = load_who()
    v = view(note, orient, scale)
    d = act(note)
    trit = [int(t) for t in (d.get("trit") or [])]
    return {
        "system": "JuniorGaia",
        "layers": list(LAYERS),
        "who": who,
        "companion_ui": "JuniorHome/ui/gaia.html",
        "dash_ui": "JuniorHome/ui/dash.html",
        "view": {"orient": v.get("orient"), "px": v.get("px"), "scale": v.get("scale"), "optional": True},
        "omega": v.get("omega"),
        "note": {
            "area": d.get("area"),
            "port": d.get("port"),
            "gamma": d.get("gamma"),
            "trit": trit,
            "i2s_hex": d.get("i2s_hex"),
        },
        "field": d.get("field"),
        "probes": {
            "llama_ready": bool(llama_plan().get("ready")),
            "bitnet_cpp_ready": bool(bitnet_plan().get("ready")),
        },
        "ue5_launch": False,
        "download": False,
        "ok": bool(v.get("ok") and d.get("ok")),
    }


def pulse(note: str = "home") -> dict:
    return spine(note)
