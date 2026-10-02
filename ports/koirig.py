"""Koirig route. Does not terraform a crossed name."""
LAND = ("note", "step", "gamma", "filter", "clock", "mix")
SEA = ("tick", "gate", "envelope")
def route(kind, name):
    ok = (kind == "land" and name in LAND) or (kind == "sea" and name in SEA)
    return {"ok": ok, "engine": "koirig", "side": kind if ok else "deny", "model_pull": False}
