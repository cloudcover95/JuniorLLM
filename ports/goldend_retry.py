"""Goldend path that retries the web3 cycle. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.cycle_retry import retry
from ports.goldend_web3 import run as web3_run

def run(note: str = "goldend retry") -> dict:
    web3 = web3_run(note)
    result = retry(note)
    body = {"web3": web3, "retry": result, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "goldend_retry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
