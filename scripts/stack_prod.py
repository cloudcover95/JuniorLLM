#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.bitnet_cloud import rows
from ports.enduser_llm import build
from ports.inject import write_vault
from ports.layer1_iq import cycle

NOTES = (
    "dxf missing height",
    "triton cuda fused dot",
    "asahi mlx overlay",
    "aarch64 pi i2sd",
    "home local llm",
    "flagstaff dry V4 crimp",
    "custom node book",
    "victron mppt transit",
    "junioros llama plan",
    "palace seal memsys",
    "engrtools title block redline",
    "theorycu zk receipt",
)


def main(argv: list[str]) -> int:
    vault = Path(argv[1]) if len(argv) > 1 else Path("/tmp/juniorhome_vault")
    cycle(vault / "layer1_lock.json", "home local llm")
    card = build(vault, "home local llm")
    out = []
    for n in NOTES:
        row = write_vault(n, vault)
        out.append({"note": n, "area": row.get("area"), "ok": bool(row.get("ok"))})
    print(json.dumps({"llm": card["name"], "runtime": card["runtime"], "domains": out, "bitnetcloud": len(rows(vault))}, indent=2))
    return 0 if all(x["ok"] for x in out) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
