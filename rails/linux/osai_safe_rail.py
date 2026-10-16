"""Linux rail: safety net first, then the OSai prod chain. Loopback only.

Runs rails.linux.safety_net_rail before rails.linux.osai_prod_rail (stack tiers,
team report, prod manager). The safety net runs only from a local JuniorHome
checkout; when it is absent the rail reports it as missing and the stack result
still decides ok. If the safety net is present and fails, the rail fails.
No bench. No LUT. No GGUF. No model pull.
Usage: python3 -m rails.linux.osai_safe_rail [team]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rails.linux import osai_prod_rail, safety_net_rail  # noqa: E402


def run(team: str = "llc") -> dict:
    net = safety_net_rail.run()
    net_present = "missing" not in net
    prod = osai_prod_rail.run(team)
    ok = prod["ok"] and (net["ok"] or not net_present)
    return {
        "rail": "linux",
        "team": team,
        "safety_net": net,
        "stack": prod["stack"],
        "report": prod["report"],
        "prod_manager": prod["prod_manager"],
        "ok": ok,
        "bind": "127.0.0.1",
        "bench": False,
        "lut": False,
        "model_pull": False,
    }


if __name__ == "__main__":
    out = run(sys.argv[1] if len(sys.argv) > 1 else "llc")
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["ok"] else 1)
