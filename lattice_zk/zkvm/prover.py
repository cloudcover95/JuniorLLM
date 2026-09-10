"""Commit every state + check AIR + SIS on first/last."""
from __future__ import annotations

from dataclasses import dataclass

from lattice_zk.merkle import root
from lattice_zk.msis_local import commit as msis_commit
from lattice_zk.pack import pack
from lattice_zk.zkvm.air import step_ok
from lattice_zk.zkvm.exec import Trace, execute
from lattice_zk.zkvm.isa import Instr


@dataclass
class VmProof:
    start_packed: bytes
    last_packed: bytes
    trace_root: bytes
    start_c0: int
    last_c0: int
    cycles: int
    program: tuple[tuple[int, int, int], ...]
    # sampled packed states for AIR (all of them — statement is small)
    packed_states: tuple[bytes, ...]


def _prog_tuple(program: list[Instr]) -> tuple[tuple[int, int, int], ...]:
    return tuple((i.op, i.a, i.b) for i in program)


def prove_program(start: list[int], program: list[Instr]) -> VmProof:
    tr = execute(start, program)
    packed = tuple(pack(s) for s in tr.states)
    return VmProof(
        packed[0],
        packed[-1],
        root(list(packed)),
        msis_commit(tr.states[0]).c[0],
        msis_commit(tr.states[-1]).c[0],
        tr.cycles,
        _prog_tuple(program),
        packed,
    )


def verify_program(proof: VmProof) -> bool:
    from lattice_zk.pack import unpack
    from lattice_zk.zkvm.isa import WIDTH

    if root(list(proof.packed_states)) != proof.trace_root:
        return False
    if proof.packed_states[0] != proof.start_packed or proof.packed_states[-1] != proof.last_packed:
        return False
    if len(proof.packed_states) != proof.cycles + 1:
        return False
    states = [unpack(p, WIDTH) for p in proof.packed_states]
    if msis_commit(states[0]).c[0] != proof.start_c0:
        return False
    if msis_commit(states[-1]).c[0] != proof.last_c0:
        return False
    program = [Instr(op, a, b) for op, a, b in proof.program]
    if len(program) != proof.cycles:
        return False
    for i, ins in enumerate(program):
        if not step_ok(states[i], ins, states[i + 1]):
            return False
    return True
