"""State-vector palace. Pulls are copies. ZK commit is sealed once.

If JuniorMemSys-Suite is on path, SIS uses its msis_commit.
Teqp / FieldCore / IQ never write back into the sealed slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from junior_bitnet.teqp import props
from lattice_zk.ternary_arg import prove, verify

try:
    from junior_memsys_suite.lattice.msis_commit import commit as _msis
except Exception:
    from lattice_zk.msis_local import commit as _msis


@dataclass
class Slot:
    z: list[int]
    proof: object
    c0: int


@dataclass
class Palace:
    slots: dict[str, Slot] = field(default_factory=dict)
    backend: str = "msis_local"

    def __post_init__(self) -> None:
        try:
            import junior_memsys_suite  # noqa: F401

            self.backend = "junior_memsys_suite"
        except Exception:
            self.backend = "msis_local"

    def seal(self, name: str, z: list[int]) -> Slot:
        zz = [int(max(-1, min(1, v))) for v in z]
        pr = prove(zz)
        slot = Slot(list(zz), pr, pr.c0)
        self.slots[name] = slot
        return slot

    def pull(self, name: str) -> list[int]:
        """Copy. Caller may run teqp on this; must not put it back."""
        return list(self.slots[name].z)

    def observe(self, name: str) -> dict:
        z = self.pull(name)
        p = props(z)
        still = verify(self.slots[name].proof)
        same = _msis(z).c[0] == self.slots[name].c0
        return {
            "name": name,
            "backend": self.backend,
            "verify": still,
            "commit_unchanged": same,
            "rho": p.rho,
            "phase": p.phase,
            "n": p.n,
        }
