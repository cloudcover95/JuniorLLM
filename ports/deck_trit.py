"""Read JuniorHome osai join. Ticket only. T43 untouched."""
from __future__ import annotations

import json
from pathlib import Path

JOIN = Path.home() / ".juniorhome" / "deck" / "osai_join.json"


def load() -> dict:
    if not JOIN.exists():
        return {"ok": False, "reason": "no_join", "model_pull": False}
    try:
        row = json.loads(JOIN.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "reason": "bad_json", "model_pull": False}
    trit = row.get("trit") or {}
    votes = [
        row.get("protocol") == "goldend-osai-omega/1",
        row.get("bind") == "127.0.0.1",
        row.get("model_pull") is False,
        bool(trit.get("sha3")),
        "pack5" in trit,
        isinstance(row.get("objectives"), list),
    ]
    return {
        "ok": all(votes),
        "flagstaff": all(votes),
        "sha3": trit.get("sha3"),
        "objectives": row.get("objectives") or [],
        "audio_n": row.get("audio_n", 0),
        "model_pull": False,
        "ue5_launch": False,
    }


if __name__ == "__main__":
    print(json.dumps(load(), indent=2))
