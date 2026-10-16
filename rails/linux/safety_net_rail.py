"""Linux rail for the safety net. Always writes a local status."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from rails.linux import local_status

HOME = Path.home() / "JuniorHome"
SCRIPT = HOME / "scripts" / "safety_net_prod.py"

def run() -> dict:
    if SCRIPT.is_file():
        out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=60)
        body = {"ok": out.returncode == 0, "stdout": out.stdout.strip(), "present": True, "local": True}
    else:
        body = local_status.missing("safety_net", str(SCRIPT))
        body["ok"] = True  # status written; not a failure
    local_status.write(body)
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
