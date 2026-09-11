"""Checks before a note becomes a node. Area = Home domain, not only rock."""
from __future__ import annotations

import math

from ports.layer_mgr import pick_eos
from ports.terraform import terraform

AREAS = {
    "home",
    "vault",
    "stock",
    "cad",
    "omega",
    "van",
    "os",
    "llm",
    "flagstaff",
    "climbs",
    "stonefield",
    "xanadu",
    "golden",
    "boulder",
}


def check(note: str, *, area: str = "home", consent: bool = True, private: bool = False) -> dict:
    tf = terraform(note)
    port = pick_eos(note or area, 8).name
    votes = {
        "terraform_ok": bool(tf.get("ok") and tf.get("text")),
        "covenant": not (private and not consent),
        "area": area.lower() in AREAS,
        "junior_port": port.startswith("Junior"),
        "finite_y": math.isfinite(float(tf.get("fusion_y") or 0)),
        "no_bind": "0.0.0.0" not in (tf.get("text") or ""),
    }
    return {"ok": all(votes.values()), "votes": votes, "tf": tf, "port": port, "area": area.lower()}
