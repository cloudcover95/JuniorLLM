"""Interpolate missing elevation from miscellaneous notes vs core profile.

Never silent. Output is a hypothesis scored by ternary drift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from adaptations.astra_reason.rigid_iq import drift, embed, run_iq

NUM = re.compile(r"(?<![A-Z])(\d+(?:\.\d+)?)\s*(MM|IN|INCH)?", re.I)


@dataclass
class Interp:
    height: float | None
    source: str
    agreement: float
    hypothesis: bool
    notes: list[str]


def _nums(text: str) -> list[float]:
    out = []
    for m in NUM.finditer(text or ""):
        v = float(m.group(1))
        if 0.1 < v < 5000:
            out.append(v)
    return out


def interpolate(profile: str, misc: str, sidecar: str = "") -> Interp:
    notes: list[str] = []
    core = _nums(profile + "\n" + sidecar)
    extra = _nums(misc)
    iq = run_iq((profile or "") + " | " + (misc or "") + " | " + sidecar, loops=3)
    agree = max(0.0, min(1.0, 1.0 - drift(embed(profile or "p"), embed(misc or sidecar or "m"))))
    height = None
    source = "none"
    m = re.search(r"HEIGHT\s*[:#]?\s*([0-9.]+)", sidecar + "\n" + misc, re.I)
    if m:
        height = float(m.group(1))
        source = "explicit"
    elif extra:
        # misc numbers that are small vs longest profile dim → likely thickness
        span = max(core) if core else None
        cand = [x for x in extra if span is None or x < span * 0.8]
        if cand:
            cand.sort()
            height = cand[len(cand) // 2]
            source = "misc_median"
            notes.append(f"median misc {height} vs profile span {span}")
    if height is None and core:
        height = min(core) * 0.25
        source = "profile_quarter"
        notes.append("no misc thickness; used 0.25*min profile dim")
    hyp = source != "explicit"
    if iq.recommendation == "low_confidence":
        notes.append("iq low — do not treat as elev")
        if hyp:
            height = None
            source = "blocked"
    return Interp(height, source, round(agree, 4), hyp, notes)
