"""Turn Gaia trit bolts into an OBJ + Blender py. No bpy import. No GLB pull."""
from __future__ import annotations

from pathlib import Path

from ports.gaia import spine


def _pts(bolts: list[dict]) -> list[tuple[float, float, float]]:
    pts = [(0.0, 1.6, 0.0)]
    y = 1.4
    for b in bolts:
        x = 0.12 * int(b.get("trit") or 0)
        z = 0.08 * (1 if b.get("lit") else 0)
        pts.append((x, y, z))
        y -= 0.18
    pts.append((0.0, y, 0.0))
    return pts


def obj_text(note: str = "gaia home", who: dict | None = None) -> str:
    s = spine(note, who)
    pts = _pts(s.get("bolts") or [])
    lines = [
        "# JuniorGaia goldend spine — not a Halo asset",
        f"# name={s['who'].get('name')} pronouns={s['who'].get('pronouns')}",
        f"# gamma={s.get('gamma')}",
        "o GaiaSpine",
    ]
    for x, y, z in pts:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    for i in range(1, len(pts)):
        lines.append(f"l {i} {i + 1}")
    # tiny head ellipse as triangle fan
    lines += ["v 0.0000 1.7200 0.0000", "v 0.1200 1.6000 0.0000", "v -0.1200 1.6000 0.0000", "f -3 -2 -1"]
    return "\n".join(lines) + "\n"


def blender_py(note: str = "gaia home") -> str:
    return (
        "# Run in Blender: blender --background --python this_file.py\n"
        "# Does not ship a .blend. JuniorOmega may replace this later.\n"
        "try:\n"
        "    import bpy\n"
        "    bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        "    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.18, location=(0, 1.6, 0))\n"
        "    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=1.4, location=(0, 0.7, 0))\n"
        "    print('gaia mesh stub ok')\n"
        "except Exception as e:\n"
        "    print('no bpy', type(e).__name__)\n"
    )


def write(out_dir: Path, note: str = "gaia home", who: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / "gaia_spine.obj"
    py = out_dir / "gaia_blender.py"
    obj.write_text(obj_text(note, who), encoding="utf-8")
    py.write_text(blender_py(note), encoding="utf-8")
    s = spine(note, who)
    return {
        "ok": s.get("ok"),
        "obj": str(obj),
        "blender_py": str(py),
        "omega_mesh": "stub",
        "ue5_launch": False,
        "download": False,
        "who": s.get("who"),
        "n_bolts": len(s.get("bolts") or []),
    }
