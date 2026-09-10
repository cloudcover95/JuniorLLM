#!/usr/bin/env python3
"""One-shot production check for the BitNet-quant stack."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run() -> dict:
    from bitnet_net.node import Node
    from bitnet_pq.params import Params
    from bitnet_pq.pipeline import run as lake_run
    from bitnet_pq.scripts.cli import bench as pq_bench
    from lattice_zk.zkvm.bench import main as zkvm_bench

    tmp = Path(tempfile.mkdtemp(prefix="junior-quant-"))
    lake = lake_run(tmp / "lake")
    node = Node(tmp / "net")
    node.mint("treasury", 8)
    node.seal()
    node.transfer("treasury", "member", 2)
    node.seal()
    return {
        "params": Params().__dict__,
        "pq_bench": pq_bench(),
        "zkvm": {k: zkvm_bench(8)[k] for k in ("ok", "cycles", "proof_bytes", "comparable_to_jolt_rv")},
        "lake": lake,
        "net": {"balances": node.balances, "height": len(node.blocks)},
        "bind": "127.0.0.1",
        "secure": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
