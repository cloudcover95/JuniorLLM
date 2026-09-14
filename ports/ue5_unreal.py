"""Text an operator can paste into UE5 Python. Home never imports unreal."""
from __future__ import annotations

from pathlib import Path

SNIPPET = '''# PASTE IN UE5 PYTHON CONSOLE. Not executed by Home.
# import unreal
# world = unreal.EditorLevelLibrary.get_editor_world()
# unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0,0,0))
'''


def write(out: Path | None = None) -> dict:
    p = out or (Path.home() / ".juniorhome" / "gaia_mesh" / "ue5_unreal_paste.py")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(SNIPPET, encoding="utf-8")
    return {"path": str(p), "launch": False, "home_imports_unreal": False}
