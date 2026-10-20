"""Agent update from final status. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.agent_mgr import update

FINAL = Path.home() / ".juniorhome" / "os" / "final_status.json"

def sync() -> dict:
    body = {"updated": False, "local": True}
    if FINAL.exists():
        try:
            data = json.loads(FINAL.read_text(encoding="utf-8"))
            check = data.get("energy_check") or {}
            ok = check.get("ok", False)
            energy = check.get("energy", 0.0)
            env = {"layer": "T0" if ok else "T1", "max_tokens": 512}
            update("final", env, energy, "" if ok else "repack")
            body = {"updated": True, "ok": ok, "energy": energy, "local": True}
        except Exception:
            pass
    out = Path.home() / ".juniorhome" / "os" / "final_agent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(sync(), indent=2))
