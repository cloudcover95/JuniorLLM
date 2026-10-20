"""Full path that can use the web3 adapter. Local only."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ports.web3_adapter import score as web3_score
from ports.osai_terraform import decide
from ports.inference_engine import record

def path(note: str, use_web3: bool = True) -> dict:
    t0 = time.perf_counter()
    layers = web3_score(note) if use_web3 else {}
    decision = decide(note)
    rec = record(note, source="full_path_web3" if use_web3 else "full_path")
    ms = (time.perf_counter() - t0) * 1000
    body = {
        "layers": layers,
        "decision": decision,
        "record": rec,
        "ms": round(ms, 4),
        "web3": use_web3,
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "full_path_web3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(path("os tick"), indent=2))
