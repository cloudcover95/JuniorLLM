#!/usr/bin/env python3
"""Team report one-shot. Calls Home if present, else writes a local stub."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

HOME = Path.home() / "JuniorHome"
SCRIPT = HOME / "scripts" / "team_report_prod.py"

def main():
    team = sys.argv[1] if len(sys.argv) > 1 else "llc"
    if SCRIPT.is_file():
        out = subprocess.run([sys.executable, str(SCRIPT), team], capture_output=True, text=True, timeout=60)
        print(out.stdout or out.stderr)
        raise SystemExit(out.returncode)
    stub = {"team": team, "ok": False, "missing": "JuniorHome/scripts/team_report_prod.py", "local": True}
    print(json.dumps(stub, indent=2))
    raise SystemExit(0)

if __name__ == "__main__":
    main()
