#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from junior_bitnet.coolstore import build
from junior_bitnet.prove import prove
from ports.layer_mgr import report


def main() -> int:
    tab = build()
    p = prove()
    print(json.dumps({"prove": p["ok"], "layers": report(tab, 16)}, indent=2))
    return 0 if p["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
