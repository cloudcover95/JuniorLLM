"""Map UE5 JSON actors → bpy script. Does not run blender. Does not spawn Unreal."""
from __future__ import annotations

from pathlib import Path

from ports.ue5_port import scene


def bpy_from_scene(spec: dict) -> str:
    lines = [
        "# generated — operator may: blender --background --python this.py",
        "import bpy",
        "bpy.ops.wm.read_factory_settings(use_empty=True)",
    ]
    for i, a in enumerate(spec.get("actors") or []):
        name = a.get("name", f"a{i}")
        klass = a.get("class", "")
        if klass == "StaticMeshActor" or a.get("mesh"):
            lines += [
                "bpy.ops.mesh.primitive_plane_add(size=8)",
                f"bpy.context.object.name = {name!r}",
            ]
        elif klass == "PointLight":
            lines += [
                "bpy.ops.object.light_add(type='POINT')",
                f"bpy.context.object.name = {name!r}",
            ]
        elif klass == "CameraActor":
            lines += [
                "bpy.ops.object.camera_add()",
                f"bpy.context.object.name = {name!r}",
            ]
        else:
            lines += [
                "bpy.ops.mesh.primitive_cube_add()",
                f"bpy.context.object.name = {name!r}",
            ]
    lines.append("# export obj from the operator box; Home will not os.system blender")
    return "\n".join(lines) + "\n"


def write(out_dir: Path | None = None, note: str = "gaia dash", surfaces: tuple[str, ...] = ("gaia", "terrain", "markets")) -> dict:
    out_dir = out_dir or (Path.home() / ".juniorhome" / "gaia_mesh")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = scene(note, 32, surfaces)
    py = out_dir / "ue5_actors.py"
    py.write_text(bpy_from_scene(spec), encoding="utf-8")
    return {"py": str(py), "actors": len(spec.get("actors") or []), "launch": False, "blender_cli": False, "protocol": "goldend-osai-omega/1"}
