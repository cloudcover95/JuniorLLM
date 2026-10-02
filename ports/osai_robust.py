"""Read the OSai gate receipt. Missing file denies."""
from pathlib import Path
import json
OUT = Path.home() / ".juniorhome" / "gaia_mesh" / "osai_robust.jsonl"
def last():
    if not OUT.exists():
        return {"ok": False, "allow_push": False, "reason": "no osai_robust.jsonl", "model_pull": False}
    row = json.loads(OUT.read_text(encoding="utf-8").strip().splitlines()[-1])
    row["allow_push"] = bool(row.get("ok"))
    row["model_pull"] = False
    return row
