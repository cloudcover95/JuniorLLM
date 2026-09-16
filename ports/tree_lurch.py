"""Grab a tree node, recompute, write same id. Hamming is the ANN."""
from __future__ import annotations

import json
from pathlib import Path

from ports.fieldcore_spine import expand
from ports.gaia_proto import handshake
from ports.tree_dense import TREE, add, kind_of
from ports.trit_cache import hit, put

LURCH = Path.home() / ".juniorhome" / "gaia_mesh" / "tree_lurch.jsonl"


def _load() -> list[dict]:
    if not TREE.is_file():
        return []
    rows = []
    for line in TREE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def search(note: str, max_d: int = 2) -> dict:
    put(note)
    return {"ann": "i2s-hamming", "faiss": False, **hit(note, max_d=max_d)}


def lurch(note: str) -> dict:
    found = search(note)
    nodes = _load()
    target = next((n for n in reversed(nodes) if n.get("note") == note), None)
    if target is None:
        target = add(note)
    if not target.get("ok"):
        return {**target, "lurch": False, "faiss": False}
    hs = handshake(note, job="gaia-spine")
    fc = expand(n=32, k=8)
    gen = int(target.get("gen") or 0) + 1
    row = {
        **target,
        "gen": gen,
        "kind": kind_of(note),
        "gamma": fc.get("gamma"),
        "sparsity": fc.get("sparsity"),
        "schema_ok": hs.get("schema_ok"),
        "near": found.get("near"),
        "lurch": True,
        "faiss": False,
        "ann": "i2s-hamming",
    }
    LURCH.parent.mkdir(parents=True, exist_ok=True)
    LURCH.open("a", encoding="utf-8").write(json.dumps(row) + "\n")
    TREE.open("a", encoding="utf-8").write(json.dumps(row) + "\n")
    return row
