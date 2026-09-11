"""Checks and balances before a Flagstaff note becomes a node."""
from __future__ import annotations

import math

from ports.layer_mgr import pick_eos
from ports.terraform import terraform

AREAS = {"flagstaff", "xanadu", "golden", "boulder", "mt xanadu"}


def check(note: str, *, area: str = "flagstaff", consent: bool = True, private: bool = False) -> dict:
    tf = terraform(note)
    votes = {
        "terraform_ok": bool(tf.get("ok") and tf.get("text")),
        "covenant": not (private and not consent),
        "area": area.lower() in AREAS or area.lower() in (tf.get("text") or ""),
        "fieldcore": pick_eos(note or "flagstaff", 8).name == "JuniorBitNetFieldCore",
        "finite_y": math.isfinite(float(tf.get("fusion_y") or 0)),
        "no_bind": "0.0.0.0" not in (tf.get("text") or ""),
    }
    return {"ok": all(votes.values()), "votes": votes, "tf": tf}
