"""Wrap every Linux rail so a core report never goes missing. Loopback only.

Runs the stack, team report, prod manager and safety net rails. Each one gets
its own status file under ~/.juniorhome/os/status/<rail>.json, even when its
JuniorHome script is absent or the rail raises. A summary goes to
local_status.json via rails.linux.local_status.
No bench. No LUT. No GGUF. No model pull.
Usage: python3 -m rails.linux.status_rails [team]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rails.linux import (  # noqa: E402
    local_status,
    osai_stack_rail,
    prod_manager_rail,
    safety_net_rail,
    team_report_rail,
)

STATUS_DIR = local_status.STATUS.parent / "status"


def _write(name: str, body: dict[str, Any]) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    p = STATUS_DIR / f"{name}.json"
    p.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return p


def _guard(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        out = fn()
        present = "missing" not in out and out.get("present", True) is not False
        body = {"rail": name, "present": present, "ok": bool(out.get("ok")), "out": out}
    except Exception as e:  # never let a rail vanish from the report
        body = {"rail": name, "present": True, "ok": False, "error": f"{type(e).__name__}: {e}"}
    body["ts"] = datetime.now(timezone.utc).isoformat()
    body["local"] = True
    body["path"] = str(_write(name, body))
    return body


def run(team: str = "llc") -> dict[str, Any]:
    rails = {
        "osai_stack": osai_stack_rail.run,
        "team_report": lambda: team_report_rail.run(team),
        "prod_manager": prod_manager_rail.run,
        "safety_net": safety_net_rail.run,
    }
    rows = [_guard(n, f) for n, f in rails.items()]
    summary = {
        "what": "linux_rails",
        "team": team,
        "rails": {r["rail"]: {"present": r["present"], "ok": r["ok"], "path": r["path"]} for r in rows},
        "ok": all(r["ok"] or not r["present"] for r in rows),
        "bind": "127.0.0.1",
        "bench": False,
        "lut": False,
        "model_pull": False,
        "local": True,
    }
    # safety_net_rail writes local_status.json itself; write the summary last so it wins.
    local_status.write(summary)
    return summary


if __name__ == "__main__":
    out = run(sys.argv[1] if len(sys.argv) > 1 else "llc")
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["ok"] else 1)
