"""T61 CLI dispatch for skill-pin thirteenth-cousins. Loopback only."""
from __future__ import annotations

import json


def run_t61(argv: list[str], ns: dict) -> int:
    hdr = argv[3] if len(argv) > 3 else None
    dest = argv[4] if len(argv) > 4 else None
    report = ns["skill_pin_thirteenth_cousins"](hdr, dest)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1
