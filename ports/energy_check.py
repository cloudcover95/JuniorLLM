"""Self-check for manager energy. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.manager_with_energy import run

def check(note: str = "energy check") -> dict:
    result = run(note)
    ok = "energy" in result and isinstance(result["energy"], (int, float))
    body = {"ok": ok, "energy": result.get("energy"), "keep": result.get("keep"), "local": True}
    out = Path.home() / ".juniorhome" / "os" / "energy_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
