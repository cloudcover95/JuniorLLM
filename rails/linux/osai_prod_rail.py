"""Linux rail: OSai stack + team report, then the prod manager check. Loopback only.

Chains rails.linux.osai_team_rail (stack tiers + team report) with
rails.linux.prod_manager_rail. The prod manager runs only from a local
JuniorHome checkout; when it is absent the rail reports it as missing and the
stack result still decides ok. No bench. No LUT. No GGUF. No model pull.
Usage: python3 -m rails.linux.osai_prod_rail [team]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rails.linux import osai_team_rail, prod_manager_rail  # noqa: E402


def run(team: str = "llc") -> dict:
    team_out = osai_team_rail.run(team)
    manager = prod_manager_rail.run()
    return {
        "rail": "linux",
        "team": team,
        "stack": team_out["stack"],
        "report": team_out["report"],
        "prod_manager": manager,
        "ok": team_out["ok"],
        "bind": "127.0.0.1",
        "bench": False,
        "lut": False,
        "model_pull": False,
    }


if __name__ == "__main__":
    out = run(sys.argv[1] if len(sys.argv) > 1 else "llc")
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["ok"] else 1)
