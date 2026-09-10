"""zkVM bench. Cycles here are BitNet-ISA steps, not RV64."""
from __future__ import annotations

import json
import time

from lattice_zk.zkvm.isa import CLAMP, MAC, SPARSE, Instr
from lattice_zk.zkvm.prover import prove_program, verify_program


def _prog(n: int) -> list[Instr]:
    out = []
    for i in range(n):
        out.append(Instr(MAC, i % 32, (i + 3) % 32))
        if i % 4 == 0:
            out.append(Instr(SPARSE, i % 32))
        if i % 8 == 0:
            out.append(Instr(CLAMP, i % 32))
    return out


def main(cycles_target: int = 64) -> dict:
    z = [1, 0, -1, 1] * 8
    program = _prog(cycles_target)
    t0 = time.perf_counter()
    proof = prove_program(z, program)
    prove_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    ok = verify_program(proof)
    verify_s = time.perf_counter() - t1
    size = (
        len(proof.start_packed)
        + len(proof.last_packed)
        + len(proof.trace_root)
        + sum(len(x) for x in proof.packed_states)
        + 16
    )
    cps = proof.cycles / prove_s if prove_s else 0
    return {
        "ok": ok,
        "isa": "JuniorBitNet-quant-6op",
        "cycles": proof.cycles,
        "proof_bytes": size,
        "prove_ms": round(prove_s * 1000, 3),
        "verify_ms": round(verify_s * 1000, 3),
        "bitnet_isa_cycles_per_s": int(cps),
        "jolt_rv_cycles_s_cpu_claimed": 2_000_000,
        "comparable_to_jolt_rv": False,
        "note": "Our cycle is one ternary op on 32 trits. Jolt cycle is RV64IMAC.",
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
