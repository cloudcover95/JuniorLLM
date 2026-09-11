#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.omega_in import ingest
from rails.linux.backend import report
from rails.linux.llama import plan
from scripts.stack_prod import main as stack


def main(argv: list[str]) -> int:
    vault = Path(argv[1]) if len(argv) > 1 else Path("/tmp/juniorhome_vault")
    rc = stack(["stack", str(vault)])
    om = ingest(vault)
    extra = {
        "probe": report(),
        "llama": {"ready": plan()["ready"]},
        "omega": {
            "port": om["port"],
            "guess": om["guess"],
            "tp_match": om["tp_match"],
            "inject_ok": om["inject"]["ok"],
        },
    }
    print(json.dumps(extra, indent=2))
    return rc if om["inject"]["ok"] and om["tp_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
