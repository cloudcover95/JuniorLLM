"""One harness: Python winsor + C if .so + shader-equiv + compat + FieldCore + OSai."""
from __future__ import annotations

import json
from pathlib import Path

from ports.fieldcore_spine import expand
from ports.gaia_proto import handshake
from ports.osai_goldens import run as golden_run
from rails.linux.absmean_ffi import pack as cpack
from rails.linux.asahi.vk_exec import exec_shader

ROOT = Path(__file__).resolve().parents[1]
COMPAT = ROOT / "rails" / "linux" / "asahi" / "COMPAT.md"
STAGES = ROOT / "junior_osai" / "goldens" / "vk_stages.json"


def ship(note: str = "home dash") -> dict:
    hs = handshake(note, job="dash-viewport")
    xs = [float((ord(c) % 13) - 6) for c in note] or [0.2]
    c = cpack(xs)
    vk = exec_shader(xs)
    fc = expand(n=32, k=8)
    stages = json.loads(STAGES.read_text(encoding="utf-8")) if STAGES.is_file() else {}
    return {
        "protocol": hs.get("protocol"),
        "schema_ok": hs.get("schema_ok"),
        "gamma": (hs.get("note") or {}).get("gamma"),
        "c_backend": c.get("backend"),
        "pipeline": ["TOP_OF_PIPE", "COMPUTE_SHADER", "HOST"],
        "vk": {"spirv": vk.get("spirv"), "vk_submit": False, "ms": vk.get("ms")},
        "compat": COMPAT.is_file(),
        "gpu_warranted": bool(stages.get("gpu_warranted")),
        "fetch_driver": bool(stages.get("fetch_driver")),
        "fieldcore": {"n": fc.get("n"), "energy": fc.get("energy")},
        "osai_vk_stages": STAGES.name if STAGES.is_file() else None,
        "download": False,
    }
