#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from junior_bitnet.tp import tp_bitlinear
from scripts.stack_prod import NOTES


def main() -> int:
    rows = []
    ok = True
    for n in NOTES:
        xs = [float(ord(c) % 97) for c in n[:16]]
        r = tp_bitlinear(xs, xs[::-1] or [1.0], 2)
        rows.append({"note": n, "match": r["match"], "y": r["y"]})
        ok = ok and r["match"]
    print(json.dumps({"ok": ok, "workers": 2, "rows": rows}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
