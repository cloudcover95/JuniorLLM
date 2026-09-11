#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from junior_bitnet.prove import prove

if __name__ == "__main__":
    out = prove()
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["ok"] else 1)
