#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.fusion import run, triton_available
from ports.flagstaff import assemble
from ports.layer1_iq import cycle
from rails.linux.backend import report
from rails.linux.llama import plan as llama_plan


def main(argv: list[str]) -> int:
    vault = Path(argv[1]) if len(argv) > 1 else Path("/tmp/juniorhome_vault")
    vault.mkdir(parents=True, exist_ok=True)
    lock = vault / "layer1_lock.json"
    cycle(lock, "flagstaff dry beta")
    pack = assemble("flagstaff dry beta", lock)
    (vault / "flagstaff_ctx.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    fus = run([0.5, -0.2, 0.1], [1.2, -0.4, 0.05])
    infer = {
        "llama": llama_plan(),
        "fusion": {"backend": fus["backend"], "y": fus["y"], "triton": triton_available()},
        "flagstaff": {"port": pack["port"], "total": pack["total"]},
        "probe": report(),
    }
    (vault / "infer_status.json").write_text(json.dumps(infer, indent=2), encoding="utf-8")
    if "--overlay" in argv:
        subprocess.run(
            ["sh", str(ROOT / "rails/linux/debian/stage_overlay.sh"), str(vault / "overlay")],
            check=False,
        )
    print(json.dumps({"vault": str(vault), "infer": {"llama_ready": infer["llama"]["ready"], "triton": infer["fusion"]["triton"], "port": pack["port"]}}, indent=2))
    return 0 if pack["port"] == "JuniorBitNetFieldCore" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
