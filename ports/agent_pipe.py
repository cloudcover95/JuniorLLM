"""LLM agent: Flagstaff 6-vote then optional local jsonl. No remote train."""
from __future__ import annotations

import json
from pathlib import Path

from ports.flagstaff_balance import check
from ports.gaia_proto import handshake
from ports.xyz_check import xyz

PIPE = Path.home() / ".juniorhome" / "gaia_mesh" / "agent_pipe.jsonl"

VOTES = ("terraform_ok", "covenant", "area", "junior_port", "finite_y", "no_bind")


def step(note: str, *, write: bool = True) -> dict:
    gate = check(note)
    hs = handshake(note, job="dash-viewport")
    x = xyz(note)
    row = {
        "ok": gate.get("ok"),
        "votes": gate.get("votes"),
        "needed": list(VOTES),
        "protocol": hs.get("protocol"),
        "profit": (x.get("X_junior") or {}).get("profit"),
        "web3node": "pointer-repo, runtime_active=false",
        "train": False,
        "infer": True,
        "download": False,
    }
    if write and gate.get("ok"):
        PIPE.parent.mkdir(parents=True, exist_ok=True)
        PIPE.open("a", encoding="utf-8").write(json.dumps(row) + "\n")
        row["jsonl"] = str(PIPE)
    return row
