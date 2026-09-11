#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from junior_bitnet.prove import prove
from junior_bitnet.refprop import Library
from junior_bitnet.vault import write_vault


def main() -> int:
    lib = Library()
    dest = Path("/tmp/junior_teqp_vault")
    note = write_vault(dest)
    print(json.dumps({"names": lib.names(), "note": str(note), "prove": prove()["ok"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
