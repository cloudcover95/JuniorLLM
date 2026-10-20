"""JuniorOS agent entry. Runs one RLVR cycle and writes status. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.rlvr_trit import cycle_with_trit
from ports.hardware_probe import probe

def run(note: str = "os tick", next_agent: str = "home") -> dict:
    hw = probe()["hardware"]
    result = cycle_with_trit(note, hardware=hw, next_agent=next_agent)
    out = Path.home() / ".juniorhome" / "os" / "junioros_agent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
