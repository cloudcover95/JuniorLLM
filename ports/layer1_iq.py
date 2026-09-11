"""Layer-1 IQ AGI: walk tiers, lock on success, cycle again without unlocking."""
from __future__ import annotations

import json
from pathlib import Path

from ports.terraform import terraform
from rails.linux.backend import probe

TIERS = ("probe", "i2s", "teqp", "ports", "terraform")


def _run(name: str, text: str) -> bool:
    if name == "probe":
        return probe().kernel == "junior_bitnet.absmean"
    if name == "i2s":
        from junior_bitnet.i2s import pack, unpack

        z = [1, 0, -1]
        return unpack(pack(z), 3) == z
    if name == "teqp":
        from junior_bitnet.teqp import props

        return 0.0 <= props([1, 0, -1] * 8).rho <= 1.0
    if name == "ports":
        from ports.layer_mgr import pick_eos

        return pick_eos("cad dxf", 8).name == "JuniorBitNetDraft"
    if name == "terraform":
        return terraform(text)["ok"]
    return False


def cycle(path: Path, text: str = "flagstaff dry beta") -> dict:
    path = Path(path)
    state = {"locked": [], "cycles": 0}
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
    locked = list(state.get("locked") or [])
    for name in TIERS:
        if name in locked:
            continue
        if _run(name, text):
            locked.append(name)
    state = {"locked": locked, "cycles": int(state.get("cycles") or 0) + 1, "tf": terraform(text)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state
