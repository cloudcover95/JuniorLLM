"""One autonomous tick: handshake spine + receipt + tree zoom."""
from __future__ import annotations

from ports.gaia_proto import handshake
from ports.tree_zoom import tick


def auto(note: str = "gaia spine") -> dict:
    hs = handshake(note, job="gaia-spine")
    tree = tick(note) if hs.get("schema_ok") else {"ok": False, "skipped": True}
    return {
        "schema_ok": hs.get("schema_ok"),
        "job": "gaia-spine",
        "protocol": hs.get("protocol"),
        "scale": tree.get("scale"),
        "nodes": tree.get("nodes"),
        "near": tree.get("near"),
        "ok": bool(hs.get("schema_ok") and tree.get("ok")),
        "ue5": False,
        "download": False,
    }
