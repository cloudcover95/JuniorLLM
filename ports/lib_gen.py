"""Emit a host adapter JSON for an OSS lib. No C++ thread pool in-tree."""
from __future__ import annotations

from junior_bitnet.winsor import pack


def generate(name: str, note: str = "host") -> dict:
    q = pack([float((ord(c) % 13) - 6) for c in (name + note)])
    return {
        "lib": name,
        "lang": "python-host",
        "cpp_pool": False,
        "ternary": {"gamma": q.get("gamma"), "sparsity": q.get("sparsity")},
        "pull": False,
        "example": "mjbatch" if "mujoco" in name.lower() or "mj" in name.lower() else name,
    }
