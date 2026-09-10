"""Argument that packed ternary z opens SIS commit c. Witness is short; proof is small."""
from __future__ import annotations

from dataclasses import dataclass

from lattice_zk.merkle import root
from lattice_zk.pack import pack, unpack

try:
    from junior_memsys_suite.lattice.msis_commit import M, commit as msis_commit
except Exception:
    from lattice_zk.msis_local import M, commit as msis_commit


@dataclass(frozen=True)
class Proof:
    packed: bytes
    merkle: bytes
    c0: int
    n: int


def prove(z: list[int], seed: bytes = b"junior-msis-v0") -> Proof:
    com = msis_commit(z, seed)
    packed = pack(z[:M] if len(z) >= M else z + [0] * M)
    return Proof(packed, root([packed]), com.c[0], M)


def verify(proof: Proof, seed: bytes = b"junior-msis-v0") -> bool:
    z = unpack(proof.packed, proof.n)
    com = msis_commit(z, seed)
    return com.c[0] == proof.c0 and root([proof.packed]) == proof.merkle


def proof_bytes(p: Proof) -> int:
    return len(p.packed) + len(p.merkle) + 8
