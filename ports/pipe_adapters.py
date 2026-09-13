"""JuniorLLM adapter names for the Home pipeline."""
STAGES = ("collect", "reduce", "memory", "route", "train")
FRAMEWORK = "JuniorOSai-adapters"

def bind(stage: str) -> str:
    s = (stage or "route").lower()
    if s == "collect":
        return "T4-ticket"
    if s == "reduce":
        return "T0-mean"
    if s == "memory":
        return "obsidian-vault"
    if s == "train":
        return "spark-latch"
    return "JuniorOSai"

if __name__ == "__main__":
    print({s: bind(s) for s in STAGES})
