"""Linux rail: OSai stack check, then the dual-team report. Loopback only.

Runs rails.linux.osai_stack_rail (T0/T1/T2, no bench), then hands off to
rails.linux.team_report_rail for the LLC team. If the team report script is not
present on this checkout, the stack result still stands and the report is marked
as missing rather than failing the rail. No bench. No LUT. No GGUF. No model pull.
Usage: python3 -m rails.linux.osai_team_rail [team]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rails.linux import osai_stack_rail, team_report_rail  # noqa: E402

REPORT_SCRIPT = ROOT / "scripts" / "team_report_prod.py"


def run(team: str = "llc") -> dict:
    stack = osai_stack_rail.run()
    if REPORT_SCRIPT.is_file():
        report = team_report_rail.run(team)
    else:
        report = {"team": team, "ok": False, "missing": str(REPORT_SCRIPT.relative_to(ROOT))}
    return {
        "rail": "linux",
        "team": team,
        "stack": {
            "ok": stack["ok"],
            "native_so": stack["native_so"],
            "tiers": [{k: r[k] for k in ("m", "k", "planned", "layer", "ok")} for r in stack["tiers"]],
        },
        "report": report,
        "ok": stack["ok"],
        "bind": "127.0.0.1",
        "bench": False,
        "lut": False,
        "model_pull": False,
    }


if __name__ == "__main__":
    out = run(sys.argv[1] if len(sys.argv) > 1 else "llc")
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["ok"] else 1)
