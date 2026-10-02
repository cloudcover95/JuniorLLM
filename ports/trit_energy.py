"""Read Home trit-energy rating. Does not write."""
from pathlib import Path
import json
OUT = Path.home() / ".juniorhome" / "os" / "trit_energy.json"
def read():
    if not OUT.exists():
        return {"ok": False, "reason": "no trit_energy.json", "model_pull": False}
    row = json.loads(OUT.read_text(encoding="utf-8"))
    row["model_pull"] = False
    return row
