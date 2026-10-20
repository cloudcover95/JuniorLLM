"""Full path with provenance labels and event emit. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from ports.full_path import path
from ports.provenance import label
from ports.events import emit

def labelled(note: str) -> dict:
    result = path(note)
    trit = label(result["layers"]["trit_score"], "trit")
    combined = label(result["layers"]["combined"], "combined")
    emit("full_path", {"layer": result["layers"]["layer"], "trit": trit, "combined": combined})
    body = {"result": result, "trit": trit, "combined": combined, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "labelled_path.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(labelled("os tick"), indent=2))
