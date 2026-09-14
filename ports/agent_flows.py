"""Five Home workflows. Each is Flagstaff → handshake. No random profit."""
from __future__ import annotations

from ports.agent_pipe import step
from ports.gaia_proto import handshake
from ports.t_layers import run as t_run

FLOWS = {
    "dash": {"job": "dash-viewport", "layer": "T0"},
    "spine": {"job": "gaia-spine", "layer": "T1"},
    "cad": {"job": "terrain-obj", "layer": "T1"},
    "vault": {"job": "dash-viewport", "layer": "T0"},
    "pipe": {"job": "agi-capsule", "layer": "T0"},
}


def run(name: str, note: str) -> dict:
    spec = FLOWS.get(name) or FLOWS["dash"]
    hs = handshake(note, job=spec["job"])
    row = step(note, write=name == "pipe" and bool(hs.get("schema_ok")))
    layer = t_run(spec["layer"], note)
    return {
        "flow": name,
        "job": spec["job"],
        "layer": spec["layer"],
        "ok": row.get("ok") and hs.get("schema_ok"),
        "votes": row.get("votes"),
        "profit": row.get("profit"),
        "schema_ok": hs.get("schema_ok"),
        "layer_ok": layer.get("schema_ok"),
        "c_vs_cuda_slide": False,
        "download": False,
    }


def all_flows(note: str = "home dash") -> dict:
    rows = {k: run(k, note) for k in FLOWS}
    return {"n": 5, "ok": all(r.get("ok") for r in rows.values()), "flows": rows}
