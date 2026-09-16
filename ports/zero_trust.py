"""Zero-trust net. Dilithium/ML-DSA is operator, not the packer."""
from __future__ import annotations

from ports.flagstaff_balance import check
from ports.gaia_proto import handshake

BOUNDS = {
    "bind": "127.0.0.1",
    "download": False,
    "rpc": False,
    "ml_kem": False,
    "ml_dsa": False,
    "dilithium": False,
    "trit_is_sig": False,
}


def metric(note: str = "home dash") -> dict:
    gate = check(note)
    hs = handshake(note, job="dash-viewport")
    net = {
        "no_bind": gate.get("votes", {}).get("no_bind"),
        "schema_ok": hs.get("schema_ok"),
        "loopback": True,
        "votes_ok": gate.get("ok"),
    }
    return {
        "ok": all(net.values()) and gate.get("ok"),
        "net": net,
        "bounds": BOUNDS,
        "dilithium_bpw": None,
        "use": "ML-DSA on operator box; trit for notes",
        "fips": {"204": "ML-DSA", "203": "ML-KEM", "205": "SLH-DSA"},
    }
