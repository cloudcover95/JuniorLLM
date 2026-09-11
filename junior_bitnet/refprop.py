"""JuniorRefprop — property library for trit fluids (REFPROP-shaped, not NIST).

PropsSI-like: props_si(out, name, T=...)
Catalog is sealed in Palace when MemSys/SIS is used; table export is public.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from junior_bitnet.math import absmean
from junior_bitnet.palace import Palace
from junior_bitnet.teqp import T_C, Props, props


def _night() -> list[int]:
    from bitnet_night.dream_mesh import DIM, tick

    st = [1, -1, 0] * (DIM // 3) + [0] * (DIM % 3)
    return tick(st, 0)


def _field() -> list[int]:
    try:
        from adaptations.astra_reason.rigid_iq import embed

        return embed("field beta dry open V4")
    except Exception:
        return [1, 0, -1] * 10 + [0, 0]


def _absmean() -> list[int]:
    z, _ = absmean([40.0, 20.0, 8.0, 6.0, 0.2, -3.0] * 8)
    return z


FLUIDS = {
    "NIGHT": _night,
    "FIELD": _field,
    "ABSMEAN": _absmean,
    "SPARSE": lambda: [0] * 28 + [1, -1, 1, -1],
    "DENSE": lambda: [1, -1] * 16,
    "ZK": lambda: [-1, 0, 1, 1, 0, -1] * 5 + [0, 0],
}


@dataclass
class Row:
    name: str
    T: float
    props: dict
    sealed: bool


class Library:
    def __init__(self) -> None:
        self.palace = Palace()
        self.rows: dict[str, Row] = {}
        for name, factory in FLUIDS.items():
            z = factory()
            self.palace.seal(name.lower(), z)
            pr = props(z, T_C)
            self.rows[name] = Row(name, T_C, asdict(pr), True)

    def names(self) -> list[str]:
        return sorted(self.rows)

    def props_si(self, out: str, name: str, T: float | None = None) -> float | str:
        name = name.upper()
        z = self.palace.pull(name.lower())
        pr = props(z, T if T is not None else T_C)
        key = {"P": "pressure", "D": "rho", "A": "A", "PHASE": "phase", "TR": "T_r", "DR": "rho_r"}.get(out.upper(), out)
        val = getattr(pr, key) if hasattr(pr, key) else pr.__dict__.get(key)
        if val is None:
            raise KeyError(out)
        return val

    def flash(self, name: str, T: float) -> Props:
        return props(self.palace.pull(name.lower()), T)

    def table(self) -> list[dict]:
        return [{"name": r.name, **r.props, "sealed": r.sealed} for r in self.rows.values()]

    def markdown(self) -> str:
        lines = [
            "---",
            "title: JuniorTeqp library",
            "tags: [juniorteqp, refprop, bitnet]",
            "---",
            "",
            "# JuniorTeqp property table",
            "",
            "Not NIST REFPROP. Trit fluids sealed in Palace.",
            "",
            "| name | rho | mag | A | P | phase | sealed |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.table():
            lines.append(
                f"| {r['name']} | {r['rho']} | {r['mag']} | {r['A']} | {r['pressure']} | {r['phase']} | {r['sealed']} |"
            )
        return "\n".join(lines) + "\n"
