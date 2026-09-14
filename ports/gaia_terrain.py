"""Procedural Gaia terrain from 1.58 trit. Real faces, not a stub capsule."""
from __future__ import annotations

import math
from pathlib import Path

from junior_bitnet.winsor import pack
from ports.gaia import spine


def _hash(s: str) -> int:
    h = 2166136261
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h or 1


def heightfield(note: str, n: int = 32) -> list[list[float]]:
    q = pack([ord(c) % 13 - 6 for c in (note or "gaia")[:48]] or [0.2, -1, 0.4])
    trit = q.get("trit") or [0]
    seed = _hash(note or "gaia")
    grid = []
    for j in range(n):
        row = []
        for i in range(n):
            u, v = i / max(1, n - 1), j / max(1, n - 1)
            t = trit[(i + j) % len(trit)]
            e = math.sin((u * 7 + seed % 9) * 1.7) * 0.35
            e += math.cos((v * 5 + (seed >> 3) % 7) * 1.3) * 0.25
            e += 0.18 * int(t)
            row.append(e)
        grid.append(row)
    return grid


def obj_terrain(note: str, n: int = 32) -> str:
    g = heightfield(note, n)
    lines = ["# JuniorGaia terrain", f"# n={n}", "o GaiaTerrain"]
    for j in range(n):
        for i in range(n):
            lines.append(f"v {i} {g[j][i]:.4f} {j}")
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i + 1
            b = a + 1
            c = a + n
            d = c + 1
            lines.append(f"f {a} {b} {d} {c}")
    return "\n".join(lines) + "\n"


def write(out_dir: Path, note: str = "gaia terrain", n: int = 32) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / f"gaia_terrain_{n}.obj"
    text = obj_terrain(note, n)
    obj.write_text(text, encoding="utf-8")
    s = spine(note)
    faces = (n - 1) * (n - 1)
    return {
        "ok": s.get("ok"),
        "obj": str(obj),
        "n": n,
        "verts": n * n,
        "faces": faces,
        "bytes": obj.stat().st_size,
        "omega_mesh": "obj",
        "ue5_launch": False,
        "download": False,
    }
