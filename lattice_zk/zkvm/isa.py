"""BitNet-quant ISA. Registers are trits. Not RV64."""
from __future__ import annotations

from dataclasses import dataclass

WIDTH = 32

MAC = 1
SPARSE = 2
FLIP = 3
SHIFT = 4
CLAMP = 5
COMMIT = 6

OP_NAME = {MAC: "MAC", SPARSE: "SPARSE", FLIP: "FLIP", SHIFT: "SHIFT", CLAMP: "CLAMP", COMMIT: "COMMIT"}


@dataclass(frozen=True)
class Instr:
    op: int
    a: int
    b: int = 0


def trit(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def apply(reg: list[int], ins: Instr) -> list[int]:
    z = [trit(x) for x in reg[:WIDTH]]
    z += [0] * (WIDTH - len(z))
    i, j = ins.a % WIDTH, ins.b % WIDTH
    if ins.op == MAC:
        z[i] = trit(z[i] * z[j] + z[i])
    elif ins.op == SPARSE:
        z[i] = 0
    elif ins.op == FLIP:
        z[i] = -z[i]
    elif ins.op == SHIFT:
        z = z[1:] + z[:1]
    elif ins.op == CLAMP:
        z[i] = trit(z[i])
    elif ins.op == COMMIT:
        pass
    else:
        raise ValueError(ins.op)
    return z
