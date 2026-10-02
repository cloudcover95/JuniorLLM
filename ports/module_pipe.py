"""Read a Home module identity. Does not clone weights."""
MODULES = {"gaia": "helper", "deck": "controls"}
def route(name):
    if name not in MODULES:
        return {"ok": False, "agent": "deny", "model_pull": False}
    return {"ok": True, "name": name, "role": MODULES[name], "clone_of": None, "model_pull": False}
