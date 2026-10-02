"""Read Home registry.json. Does not write."""
from pathlib import Path
import json
REG = Path.home() / ".juniorhome" / "os" / "registry.json"
def read():
    if not REG.exists():
        return {"ok": False, "reason": "no registry.json", "model_pull": False}
    row = json.loads(REG.read_text(encoding="utf-8"))
    row["ok"] = True
    row["model_pull"] = False
    return row
