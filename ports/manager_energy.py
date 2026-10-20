"""Web3 energy check for the production manager. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def check(note: str = "manager") -> dict:
    from web3node.energy import pack_energy
    energy = pack_energy(note)
    body = {"energy": energy, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "manager_energy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
