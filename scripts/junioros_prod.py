#!/usr/bin/env python3
"""One shot: probe + layer1 lock + Flagstaff I2_S packs + optional overlay stage."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ports.flagstaff import assemble
from ports.layer1_iq import cycle
from rails.linux.backend import report


def main(argv: list[str]) -> int:
    vault = Path(argv[1]) if len(argv) > 1 else Path("/tmp/juniorhome_vault")
    vault.mkdir(parents=True, exist_ok=True)
    lock = vault / "layer1_lock.json"
    cycle(lock, "flagstaff dry beta")
    pack = assemble("flagstaff dry beta", lock)
    (vault / "flagstaff_ctx.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    if "--overlay" in argv:
        subprocess.run(
            ["sh", str(ROOT / "rails/linux/debian/stage_overlay.sh"), str(vault / "overlay")],
            check=False,
        )
    out = {"probe": report(), "flagstaff": {"port": pack["port"], "total": pack["total"], "fills": {k: v["fill"] for k, v in pack["layers"].items()}}, "vault": str(vault)}
    print(json.dumps(out, indent=2))
    return 0 if pack["port"] == "JuniorBitNetFieldCore" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
