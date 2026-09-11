#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from junior_bitnet.math import absmean
from junior_bitnet.prove import prove
from junior_bitnet.teqp import props


def main() -> int:
    t, _ = absmean([40, 20, 8, 6, 0.2, -3] * 8)
    print(json.dumps({"absmean": props(t).__dict__, "prove": prove()["ok"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
