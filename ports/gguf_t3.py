"""T3: read a local GGUF. No download. No llama.plan."""
from __future__ import annotations

import os
import struct
from pathlib import Path

MAGIC = b"GGUF"


def find() -> Path | None:
    env = os.environ.get("JUNIOR_GGUF")
    if env and Path(env).is_file():
        return Path(env)
    roots = [
        Path.home() / ".juniorhome" / "models",
        Path.home() / "models",
        Path("/opt/junior/models"),
    ]
    for r in roots:
        if not r.is_dir():
            continue
        hits = sorted(r.glob("*.gguf"))
        if hits:
            return hits[0]
    return None


def header(path: Path) -> dict:
    data = path.read_bytes()[:24]
    if data[:4] != MAGIC:
        return {"ok": False, "reason": "not-gguf", "path": str(path)}
    version = struct.unpack_from("<I", data, 4)[0] if len(data) >= 8 else 0
    return {
        "ok": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "version": version,
        "download": False,
    }


def run(note: str = "t3") -> dict:
    p = find()
    if not p:
        return {"layer": "T3", "ok": False, "reason": "no-local-gguf", "download": False, "hint": "JUNIOR_GGUF or ~/.juniorhome/models/*.gguf"}
    h = header(p)
    h["layer"] = "T3"
    h["note"] = note
    h["infer"] = False
    return h
