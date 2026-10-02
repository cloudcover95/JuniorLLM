"""OSai verbs by envelope. Empty until the device is present."""
CAPS = {"gaia": {"t4": ["ask", "label"], "t0": ["ask", "label", "note"]},
        "deck": {"t4": ["step", "note"], "t0": ["step", "note", "mix"]}}
def expand(name, layer, present):
    verbs = CAPS.get(name, {}).get(layer, [])
    return {"name": name, "layer": layer, "verbs": verbs if present else [], "admit": bool(present and verbs), "model_pull": False}
