"""Gaia handshake + noise/lie sidecar."""
from __future__ import annotations

from ports.gaia_proto import handshake
from ports.mem_lake import remember
from ports.qutrit_noise import thread
from ports.su3_lie import lie_proofs


def train(note: str = "home dash qutrit") -> dict:
    env = handshake(note, job="dash-viewport")
    wrote = remember(note, "gaia-qutrit") if env.get("schema_ok") else None
    return {
        "schema_ok": env.get("schema_ok"),
        "lie": lie_proofs(),
        "noise": thread([0, 1, 2, 1], p=0.0, kind="flip"),
        "mem": bool(wrote),
        "svd": False,
    }
