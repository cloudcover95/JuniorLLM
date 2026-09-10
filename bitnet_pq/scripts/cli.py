#!/usr/bin/env python3
"""Production scripts for the BitNet PQ chain VM (testnet only)."""
from __future__ import annotations

import json
import sys
import time

from bitnet_pq.arg import prove, verify
from bitnet_pq.chain import Chain
from bitnet_pq.params import Params
from lattice_zk.zkvm.isa import MAC, Instr

SEED = [1, 0, -1, 1] * 8


def genesis() -> dict:
    c = Chain()
    c.genesis("field", SEED)
    return c.tip()


def tick(n: int = 3) -> dict:
    c = Chain()
    c.genesis("field", SEED)
    for _ in range(n):
        c.tick("field")
    return {"tip": c.tip(), "blocks": len(c.blocks)}


def bench() -> dict:
    t0 = time.perf_counter()
    p = prove(SEED, [Instr(MAC, 0, 1)] * 8)
    prove_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    ok = verify(p)
    verify_s = time.perf_counter() - t1
    return {
        "ok": ok,
        "lambda_bits": p.params.lambda_bits,
        "challenge_bits": len(p.challenge) * 8,
        "secure": p.secure,
        "prove_ms": round(prove_s * 1000, 3),
        "verify_ms": round(verify_s * 1000, 3),
        "scheme": p.params.scheme,
    }


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "bench"
    if cmd == "genesis":
        print(json.dumps(genesis(), indent=2))
        return 0
    if cmd == "tick":
        print(json.dumps(tick(), indent=2))
        return 0
    if cmd == "bench":
        print(json.dumps(bench(), indent=2))
        return 0
    if cmd == "params":
        print(json.dumps(Params().__dict__, indent=2))
        return 0
    print("usage: python -m bitnet_pq.scripts.cli [genesis|tick|bench|params]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
