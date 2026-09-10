"""128-bit Fiat-Shamir argument over a BitNet zkVM proof.

Binds the zkVM proof with λ-bit challenges. Not a knowledge-sound SNARK.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from bitnet_pq.params import Params
from lattice_zk.zkvm.isa import Instr
from lattice_zk.zkvm.prover import VmProof, prove_program, verify_program


def _fs(*parts: bytes, n: int) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()[:n]


@dataclass
class PqProof:
    vm: VmProof
    challenge: bytes
    params: Params

    @property
    def secure(self) -> bool:
        return self.params.secure


def prove(start: list[int], program: list[Instr], params: Params | None = None) -> PqProof:
    p = params or Params()
    vm = prove_program(start, program)
    ch = _fs(vm.trace_root, vm.start_packed, vm.last_packed, p.scheme.encode(), n=p.challenge_bytes)
    if len(ch) * 8 < p.lambda_bits:
        raise ValueError("challenge shorter than lambda")
    return PqProof(vm, ch, p)


def verify(proof: PqProof) -> bool:
    p = proof.params
    if len(proof.challenge) * 8 != p.lambda_bits:
        return False
    expect = _fs(
        proof.vm.trace_root,
        proof.vm.start_packed,
        proof.vm.last_packed,
        p.scheme.encode(),
        n=p.challenge_bytes,
    )
    if expect != proof.challenge:
        return False
    return verify_program(proof.vm)
