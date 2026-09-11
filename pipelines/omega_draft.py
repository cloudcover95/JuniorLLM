"""CAD sidecar → JuniorBitNetDraft → optional Omega project."""
from __future__ import annotations

from pathlib import Path

from adaptations.omega_cad.draft import interpret


def run(sidecar: str, sheet: Path | None = None, dest: Path | None = None) -> dict:
    call = interpret(sidecar)
    out = {
        "port": call.port,
        "rec": call.rec,
        "rigidity": call.rigidity,
        "height": call.height,
        "allow_extrude": call.allow_extrude,
        "proposals": call.proposals,
        "project": None,
    }
    if sheet and dest and call.allow_extrude:
        try:
            from cad.legacy.project import export

            out["project"] = export(Path(sheet), sidecar, Path(dest)).get("files")
        except Exception as exc:
            out["project_error"] = str(exc)
    return out
