#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from junior_bitnet.prove import prove
from ports.layer1_iq import cycle


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/layer1_lock.json")
    st = cycle(dest)
    print(json.dumps({"layer1": st, "prove": prove()["ok"]}, indent=2))
    return 0 if st["locked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
