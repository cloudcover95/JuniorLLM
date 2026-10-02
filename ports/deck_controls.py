"""Deck control ticket. Does not open a device."""
PORTS = ("JuniorLLM", "AGI_SDK", "JuniorOSai", "JuniorOS", "web3node", "obsidian")
def control(name="note"):
    return {"ok": True, "who": "deck", "control": name, "ports": list(PORTS),
            "synced": False, "admit": False, "model_pull": False}
