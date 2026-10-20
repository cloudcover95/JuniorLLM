"""Full stack sandbox: absmean, labelled path, Home tile. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.web3_absmean import feed
from ports.goldend_labelled import run as goldend_run

def stack(note: str = "full stack") -> dict:
    energy = feed(note)
    goldend = goldend_run(note)
    body = {"energy": energy, "goldend": goldend, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "full_stack.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(stack(), indent=2))
