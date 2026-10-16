"""Linux rail for the dual-team report. Local only."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run(team: str = "llc") -> dict:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "team_report_prod.py"), team],
        capture_output=True, text=True, timeout=60,
    )
    return {"team": team, "ok": out.returncode == 0, "stdout": out.stdout.strip()}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
