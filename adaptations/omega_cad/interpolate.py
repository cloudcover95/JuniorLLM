"""Interpolate missing elevation from miscellaneous notes vs core profile."""
from __future__ import annotations

import re
from dataclasses import dataclass

from adaptations.astra_reason.rigid_iq import drift, embed, run_iq
from adaptations.omega_cad.quant import absmean

NUM = re.compile(r"(?<![A-Z])(\d+(?:\.\d+)?)\s*(MM|IN|INCH)?", re.I)


@dataclass
class Interp:
    height: float | None
    source: str
    agreement: float
    hypothesis: bool
    notes: list[str]
    trit_agree: float = 0.0


def _nums(text: str) -> list[float]:
    out = []
    for m in NUM.finditer(text or ""):
        v = float(m.group(1))
        if 0.1 < v < 5000:
            out.append(v)
    return out


def interpolate(profile: str, misc: str, sidecar: str = "") -> Interp:
    notes: list[str] = []
    blob = sidecar + "\n" + misc
    m = re.search(r"HEIGHT\s*[:#]?\s*([0-9.]+)", blob, re.I)
    core, extra = _nums(profile + "\n" + sidecar), _nums(misc)
    ta = 0.0
    if core and extra:
        tp, _ = absmean(core)
        tm, _ = absmean(extra + [0] * max(0, len(tp) - len(extra)))
        n = min(len(tp), len(tm))
        ta = sum(1 for i in range(n) if tp[i] == tm[i]) / n if n else 0.0
    if m:
        return Interp(float(m.group(1)), "explicit", 1.0, False, ["explicit height; skipped IQ"], ta)
    agree = max(0.0, min(1.0, 1.0 - drift(embed(profile or "p"), embed(misc or sidecar or "m"))))
    iq = run_iq((profile or "") + " | " + (misc or "") + " | " + sidecar, loops=3)
    height = None
    source = "none"
    if extra:
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
    if iq.recommendation == "low_confidence" and source != "explicit":
        notes.append("iq low — do not treat as elev")
        height, source = None, "blocked"
    return Interp(height, source, round(agree, 4), True, notes, ta)
