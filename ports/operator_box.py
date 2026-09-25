"""Later-layer collab box. Software emu only unless marker file exists."""
from __future__ import annotations

import math
from pathlib import Path

from ports.qutrit_rail import to_balanced, to_unbalanced
from ports.trit_gates import clip, kleene_and, status as gate_st

MARK = Path.home() / ".juniorhome" / "operator" / "qutrit.ok"


def present() -> bool:
    return MARK.is_file()


def cmos_level(trit: int) -> str:
    return {-1: "-V", 0: "0", 1: "+V"}.get(clip(trit), "0")


def qutrit_state(u: int) -> tuple[float, float, float]:
    """Computational basis |0>,|1>,|2> as a real 3-vector. Not unitary hardware."""
    u = int(u) % 3
    return tuple(1.0 if i == u else 0.0 for i in range(3))


def qutrit_rotate(amp: tuple[float, float, float], k: int = 1) -> tuple[float, float, float]:
    k %= 3
    return (amp[-k:] + amp[:-k]) if k else amp


def collab(notes: list[str]) -> dict:
    votes = []
    for n in notes:
        s = 1 if "ok" in n.lower() else (-1 if "fail" in n.lower() else 0)
        votes.append(s)
    acc = votes[0] if votes else 0
    for v in votes[1:]:
        acc = kleene_and(acc, v)
    return {
        "n": len(notes),
        "cons": acc,
        "cmos": [cmos_level(v) for v in votes],
        "operator": present(),
        "qutrit_hw": False,
        "fab": False,
        "gates": gate_st(),
    }


def emu_tick(u: int = 1) -> dict:
    st = qutrit_state(u)
    rot = qutrit_rotate(st, 1)
    return {
        "in": u,
        "amp": st,
        "rot": rot,
        "balanced": to_balanced(u),
        "unbalanced": to_unbalanced(to_balanced(u)),
        "norm2": round(sum(a * a for a in st), 6),
        "marker": str(MARK),
        "live_hw": present(),
        "math": "cyclic basis, not SU(3)",
    }
