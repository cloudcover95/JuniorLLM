"""Step constraints: next state must equal apply(prev, instr); all trits."""
from __future__ import annotations

from lattice_zk.zkvm.isa import Instr, apply, trit


def closed(z: list[int]) -> bool:
    return all(x in (-1, 0, 1) for x in z)


def step_ok(prev: list[int], ins: Instr, nxt: list[int]) -> bool:
    if not closed(prev) or not closed(nxt):
        return False
    expect = apply(prev, ins)
    return expect == nxt
