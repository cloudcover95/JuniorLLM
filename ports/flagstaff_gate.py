"""Flagstaff must deny. A missing coolstore is not a pack."""
from ports.flagstaff import assemble
def closed(ask="flagstaff"):
    row = assemble(ask)
    return {"ok": bool(row.get("allow")), "allow": bool(row.get("allow")),
            "cache": row.get("cache"), "error": row.get("error"), "model_pull": False}
