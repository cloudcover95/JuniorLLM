"""Read a second-brain note shape. Does not sync a vault."""
PORTS = ("JuniorLLM", "AGI_SDK", "JuniorOSai", "JuniorOS", "web3node", "obsidian")
def note(who="gaia"):
    return {"who": who, "ports": list(PORTS), "synced": False, "admit": False, "model_pull": False}
