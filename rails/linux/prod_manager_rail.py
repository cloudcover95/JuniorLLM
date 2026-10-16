"""Linux rail for the production manager check. Local only."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

HOME = Path.home() / "JuniorHome"
SCRIPT = HOME / "scripts" / "prod_manager_prod.py"

def run() -> dict:
    if SCRIPT.is_file():
        out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=60)
        return {"ok": out.returncode == 0, "stdout": out.stdout.strip(), "local": True}
    return {"ok": False, "missing": str(SCRIPT), "local": True}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
