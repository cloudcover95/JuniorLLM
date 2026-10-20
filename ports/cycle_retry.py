"""Retry the web3 cycle if the check fails. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.cycle_web3_check import check

def retry(note: str = "retry", attempts: int = 2) -> dict:
    last = None
    for _ in range(attempts):
        last = check(note)
        if last.get("ok"):
            break
    body = {"ok": last.get("ok") if last else False, "attempts": attempts, "last": last, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "cycle_retry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(retry(), indent=2))
