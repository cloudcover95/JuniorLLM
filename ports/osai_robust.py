"""Read the OSai robust receipt. Does not write."""
from pathlib import Path
import json
OUT = Path.home() / ".juniorhome" / "gaia_mesh" / "osai_robust.jsonl"
def last():
    if not OUT.exists():
        return {"ok": False, "reason": "no osai_robust.jsonl", "model_pull": False}
    row = json.loads(OUT.read_text(encoding="utf-8").strip().splitlines()[-1])
    row["model_pull"] = False
    return row
