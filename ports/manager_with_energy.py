"""Production manager with web3 energy. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def run(note: str = "manager energy") -> dict:
    from ports.production_manager import run as manager
    from ports.manager_energy import check
    result = manager(note)
    energy = check(note)
    body = {"keep": result["keep"], "synced": result["synced"], "energy": energy["energy"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "manager_with_energy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
