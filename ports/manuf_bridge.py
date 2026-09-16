"""CAD/tendon notes → Omega job. No MuJoCo fetch."""
from __future__ import annotations

from ports.agent_flows import run


def bridge(note: str) -> dict:
    row = run("cad", note or "dxf title block")
    return {
        **row,
        "mujoco": False,
        "tendon_sim": False,
        "next": "OBJ/JSON in Omega; physics is operator MuJoCo",
        "download": False,
    }
