"""Read the Home OS mesh ticket. Does not pull a model."""
from pathlib import Path
import json
MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "os_mesh.jsonl"
def last():
    if not MESH.exists():
        return {"ok": False, "reason": "no os_mesh.jsonl", "model_pull": False}
    line = MESH.read_text(encoding="utf-8").strip().splitlines()[-1]
    row = json.loads(line)
    row["ok"] = True
    row["model_pull"] = False
    row["bind"] = "127.0.0.1"
    return row
