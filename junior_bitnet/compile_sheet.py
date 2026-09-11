"""Compile all inputs into an action list before the next pipeline step."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Action:
    kind: str
    field: str
    detail: str
    blocking: bool


@dataclass
class Compile:
    ready: bool
    height: float | None
    actions: list[Action] = field(default_factory=list)
    data: dict = field(default_factory=dict)


def compile_sheet(
    sidecar: str,
    profile: str = "",
    misc: str = "",
    fixes: list[dict] | None = None,
) -> Compile:
    text = sidecar or ""
    actions: list[Action] = []
    hm = re.search(r"HEIGHT\s*[:#]?\s*([0-9.]+)", text, re.I)
    height = float(hm.group(1)) if hm else None
    signed = False
    for fx in fixes or []:
        if fx.get("field") == "height" and fx.get("by"):
            height = float(fx["value"])
            signed = True
    if height is None:
        actions.append(Action("fix", "height", "No height. POST /fix before IQ or 3D.", True))
    elif not signed and "elev" not in text.lower() and not hm:
        actions.append(Action("fix", "height", "Height is a guess. Sign it.", True))
    if not re.search(r"TITLE\s*[:#]", text, re.I):
        actions.append(Action("fix", "title", "Add TITLE:", True))
    if not re.search(r"REV", text, re.I):
        actions.append(Action("fix", "revision", "Add REV", True))
    data = {
        "sidecar": text,
        "profile": profile,
        "misc": misc,
        "fixes": fixes or [],
        "height": height,
        "run_iq": height is not None and (signed or "elev" in text.lower() or hm is not None),
    }
    ready = not any(a.blocking for a in actions)
    return Compile(ready, height, actions, data)
