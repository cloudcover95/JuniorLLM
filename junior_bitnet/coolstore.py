"""Cold property tables. CoolProp analog: TTSE-style store, PropsSI front.

Not libCoolProp. Grid is T in {0.5,1,1.5} T_c. Values only. No z.
"""
from __future__ import annotations

import json
from pathlib import Path

from junior_bitnet.refprop import Library
from junior_bitnet.teqp import T_C

GRID = (0.5, 1.0, 1.5)


def build(lib: Library | None = None) -> dict:
    lib = lib or Library()
    fluids = {}
    for name in lib.names():
        pts = []
        for k in GRID:
            T = k * T_C
            pr = lib.flash(name, T)
            pts.append(
                {
                    "T": T,
                    "T_r": pr.T_r,
                    "rho": pr.rho,
                    "A": pr.A,
                    "P": pr.pressure,
                    "phase": pr.phase,
                }
            )
        fluids[name] = {"points": pts, "rho_anchor": pts[1]["rho"]}
    return {"kind": "junior-coolstore-v0", "T_c": T_C, "grid_Tr": list(GRID), "fluids": fluids}


def dump(path: Path, table: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table or build(), indent=2), encoding="utf-8")
    return path


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def props_si(table: dict, out: str, name: str, T: float | None = None) -> float | str:
    name = name.upper()
    T = T_C if T is None else T
    pts = table["fluids"][name]["points"]
    best = min(pts, key=lambda p: abs(p["T"] - T))
    key = {"P": "P", "D": "rho", "A": "A", "PHASE": "phase"}.get(out.upper(), out)
    return best[key]
