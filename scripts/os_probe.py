#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from junior_bitnet.prove import prove
from rails.linux.backend import report


def main() -> int:
    r = report()
    p = prove()
    print(json.dumps({"probe": r, "prove": p["ok"]}, indent=2))
    return 0 if r["alphabet"] and p["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
