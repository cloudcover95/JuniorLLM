"""zkML verify for a BitNet quant layer: program is a MAC/SPARSE schedule."""
from __future__ import annotations

from dataclasses import dataclass

from bitnet_pq.arg import PqProof, prove, verify
from bitnet_pq.params import Params
from lattice_zk.zkvm.isa import CLAMP, MAC, SPARSE, Instr


def layer_program(width: int = 32, depth: int = 4) -> list[Instr]:
    prog: list[Instr] = []
    for d in range(depth):
        for i in range(width):
            prog.append(Instr(MAC, i, (i + 1 + d) % width))
        if d % 2 == 1:
            prog.append(Instr(SPARSE, d % width))
        prog.append(Instr(CLAMP, d % width))
    return prog


@dataclass
class ZkmlReceipt:
    proof: PqProof
    layers: int
    ok: bool


def prove_forward(x: list[int], layers: int = 4, params: Params | None = None) -> ZkmlReceipt:
    pr = prove(x, layer_program(32, layers), params)
    return ZkmlReceipt(pr, layers, verify(pr))


def verify_forward(rec: ZkmlReceipt) -> bool:
    return rec.ok and verify(rec.proof)
