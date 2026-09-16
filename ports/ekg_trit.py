"""Guess: sparsity as quiet-neuron fraction. Not medical EKG."""
from __future__ import annotations

from junior_bitnet.winsor import pack
from ports.gaia_proto import handshake


def ekg(note: str) -> dict:
    hs = handshake(note, job="dash-viewport")
    xs = [float((ord(c) % 13) - 6) for c in (note or "x")]
    q = pack(xs)
    sp = float(q.get("sparsity") or 0)
    return {
        "protocol": hs.get("protocol"),
        "gamma": q.get("gamma"),
        "quiet": sp,
        "firing": round(1.0 - sp, 4),
        "medical": False,
        "connectomics": False,
        "download": False,
    }
