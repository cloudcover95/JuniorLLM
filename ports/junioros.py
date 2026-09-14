"""JuniorOS = overlay + route + trit. Not a vmlinuz we compile here."""
from __future__ import annotations

from pathlib import Path

from ports.gaia_proto import handshake
from rails.linux.backend import probe

ROOT = Path(__file__).resolve().parents[1] / "rails" / "linux"


def kernel_contract() -> dict:
    box = probe()
    return {
        "name": "JuniorOS",
        "kind": "userspace-overlay",
        "vmlinuz": False,
        "iso": False,
        "kconfig": (ROOT / "kconfig.junior").is_file(),
        "os_release": (ROOT / "os-release.junior").is_file(),
        "i2sd": (ROOT / "i2sd.py").is_file(),
        "asahi_probe": box.asahi,
        "accel": box.accel,
        "bind": "127.0.0.1",
        "trit": "winsor-p95",
        "protocol": "goldend-osai-omega/1",
    }


def boot(note: str = "home dash") -> dict:
    c = kernel_contract()
    env = handshake(note, job="dash-viewport")
    return {**c, "schema_ok": env.get("schema_ok"), "who": (env.get("who") or {}).get("name"), "ue5_launch": False}
