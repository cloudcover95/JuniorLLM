"""Hardware stack probe for envelope handoff. Local only."""
from __future__ import annotations
import platform
import json
from pathlib import Path

def probe() -> dict:
    system = platform.system().lower()
    machine = platform.machine().lower()
    hw = "cpu"
    if "arm" in machine or "aarch64" in machine:
        hw = "asahi" if system == "linux" else "metal"
    elif system == "darwin":
        hw = "metal"
    elif system == "linux":
        hw = "cuda" if Path("/dev/nvidia0").exists() else "cpu"
    body = {"hardware": hw, "system": system, "machine": machine, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "hardware_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
