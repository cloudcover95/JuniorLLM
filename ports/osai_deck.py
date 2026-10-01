"""OSai port for JuniorDeck. Reads the Home join. No weight pull."""
from __future__ import annotations

from ports.deck_receipt import receipt
from ports.registry import pick


def run(task: str = "deck audio") -> dict:
    port = pick(task, 0.0)
    row = receipt()
    return {
        "port": port.name,
        "quant": port.quant,
        "download_gb": port.max_download_gb,
        "appended": row["appended"],
        "flagstaff": row["row"].get("flagstaff"),
        "model_pull": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
