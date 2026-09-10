"""Chain VM — accounts hold ternary notes; blocks carry a PqProof."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from bitnet_pq.arg import PqProof, prove, verify
from bitnet_pq.params import Params
from lattice_zk.zkvm.isa import COMMIT, MAC, SPARSE, Instr


@dataclass
class Account:
    name: str
    note: list[int]
    nonce: int = 0


@dataclass
class Block:
    height: int
    prev: str
    proof: PqProof
    hdr: str


@dataclass
class Chain:
    params: Params = field(default_factory=Params)
    accounts: dict[str, Account] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)

    def genesis(self, name: str, note: list[int]) -> None:
        self.accounts[name] = Account(name, list(note), 0)
        if not self.blocks:
            self.blocks.append(self._block(note, [], "0" * 64))

    def _block(self, start: list[int], program: list[Instr], prev: str) -> Block:
        proof = prove(start, program or [Instr(COMMIT, 0)], self.params)
        hdr = hashlib.sha256(proof.challenge + proof.vm.trace_root + prev.encode()).hexdigest()
        return Block(len(self.blocks), prev, proof, hdr)

    def tick(self, name: str) -> Block:
        acc = self.accounts[name]
        program = [
            Instr(MAC, acc.nonce % 32, (acc.nonce + 1) % 32),
            Instr(SPARSE, (acc.nonce + 3) % 32),
            Instr(COMMIT, 0),
        ]
        prev = self.blocks[-1].hdr if self.blocks else "0" * 64
        blk = self._block(acc.note, program, prev)
        if not verify(blk.proof):
            raise RuntimeError("invalid pq proof")
        if self.params.secure:
            raise RuntimeError("refusing mainnet: Params.secure is a hard gate and still False")
        from lattice_zk.pack import unpack
        from lattice_zk.zkvm.isa import WIDTH

        acc.note = unpack(blk.proof.vm.last_packed, WIDTH)
        acc.nonce += 1
        self.blocks.append(blk)
        return blk

    def tip(self) -> dict:
        b = self.blocks[-1]
        return {
            "height": b.height,
            "hdr": b.hdr,
            "cycles": b.proof.vm.cycles,
            "lambda_bits": self.params.lambda_bits,
            "secure": self.params.secure,
            "challenge_hex": b.proof.challenge.hex(),
        }
