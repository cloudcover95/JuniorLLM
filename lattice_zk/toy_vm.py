"""Tiny ternary ISA. Not RISC-V. Trace root is a Merkle of packed states."""
from __future__ import annotations

from lattice_zk.merkle import root
from lattice_zk.pack import pack

OPS = {"MAC": 0, "FLIP": 1, "SPARSE": 2}


def step(z: list[int], op: str, i: int) -> list[int]:
    out = list(z)
    n = len(out)
    if op == "MAC":
        j = (i + 1) % n
        s = out[i] * out[j]
        out[i] = 0 if s == 0 else (1 if s > 0 else -1)
    elif op == "FLIP":
        out[i % n] = -out[i % n]
    else:
        out[i % n] = 0
    return out


def run(z: list[int], program: list[tuple[str, int]]) -> tuple[list[int], bytes, int]:
    cur = list(z)
    leaves = [pack(cur)]
    for op, i in program:
        cur = step(cur, op, i)
        leaves.append(pack(cur))
    return cur, root(leaves), len(program)
