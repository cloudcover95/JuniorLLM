"""Execute a program, emit a trace of register files."""
from __future__ import annotations

from dataclasses import dataclass

from lattice_zk.zkvm.isa import WIDTH, Instr, apply, trit


@dataclass
class Trace:
    start: list[int]
    program: list[Instr]
    states: list[list[int]]  # len = 1 + len(program)
    cycles: int


def normalize(z: list[int]) -> list[int]:
    out = [trit(x) for x in z[:WIDTH]]
    return out + [0] * (WIDTH - len(out))


def execute(start: list[int], program: list[Instr]) -> Trace:
    cur = normalize(start)
    states = [list(cur)]
    for ins in program:
        cur = apply(cur, ins)
        states.append(list(cur))
    return Trace(states[0], list(program), states, len(program))
