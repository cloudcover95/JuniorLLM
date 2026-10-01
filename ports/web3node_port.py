"""web3node port. AbsMean on the note. Not a chain address."""
from __future__ import annotations

import hashlib


def absmean(xs: list[float]) -> tuple[list[int], float]:
    if not xs:
        return [], 1.0
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return out, gamma


def card(note: str = "web3node") -> dict:
    xs = [((ord(c) % 17) - 8) / 8.0 for c in note[:32]] or [0.1]
    wq, gamma = absmean(xs)
    return {
        "name": "web3node",
        "gamma": gamma,
        "zeros": wq.count(0),
        "sha3": hashlib.sha3_256(bytes(t + 1 for t in wq)).hexdigest()[:16],
        "chain": False,
        "model_pull": False,
    }
