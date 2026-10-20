"""Ternary efficiency summary for the Home tile. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.ternary_efficiency import check

def summary() -> dict:
    eff = check(n=500)
    body = {
        "trit_ms": eff["trit_ms"],
        "baseline_ms": eff["baseline_ms"],
        "cheaper": eff["cheaper"],
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "efficiency_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
