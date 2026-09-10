"""Measure our statement. Print Jolt numbers as reference only."""
from __future__ import annotations

import json
import time

from lattice_zk.mldsa_toy import keygen, sign, verify as sig_ok
from lattice_zk.ternary_arg import proof_bytes, prove, verify
from lattice_zk.toy_vm import run

JOLT = {
    "proof_bytes_claimed": 100_000,
    "rv_cycles_s_cpu_claimed": 2_000_000,
    "rv_cycles_s_apple_claimed": 10_000_000,
    "speedup_vs_curve_claimed": "2-3x",
    "statement": "RISC-V execution (zkVM)",
}


def _time(fn, n: int = 200) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> dict:
    z = [1, 0, -1, 1] * 8
    p = prove(z)
    assert verify(p)
    prove_s = _time(lambda: prove(z))
    verify_s = _time(lambda: verify(p))
    _zf, tr, steps = run(z, [("MAC", i % 32) for i in range(64)])
    k = keygen()
    sig = sign(k, b"palace-note")
    out = {
        "ours": {
            "statement": "Az=c over 32 trits (MemSys/BitNet bind)",
            "proof_bytes": proof_bytes(p),
            "prove_us": round(prove_s * 1e6, 2),
            "verify_us": round(verify_s * 1e6, 2),
            "proves_per_s": int(1 / prove_s) if prove_s else 0,
            "toy_vm_steps": steps,
            "toy_vm_trace_root_hex": tr.hex()[:16],
            "toy_sig_ok": sig_ok(k, b"palace-note", sig),
            "toy_sig_secure": k.secure,
        },
        "jolt_reference_not_measured_here": JOLT,
        "comparable": False,
        "note": "Smaller proof because the statement is 32 MACs, not millions of RV64 cycles.",
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
