"""BitnetCloud: second brain with a cloud name and a local disk."""
from __future__ import annotations

import json
from pathlib import Path

from junior_bitnet.edgepack import edgepack
from ports.obsidian import write as md_write


def root(vault: Path) -> Path:
    p = Path(vault) / "bitnetCloud"
    p.mkdir(parents=True, exist_ok=True)
    return p


def put(vault: Path, text: str, area: str = "home") -> dict:
    row = {
        "area": area,
        "text": text,
        "edgepack": edgepack(text),
        "cloud": False,
        "local": True,
    }
    idx = root(vault) / "index.jsonl"
    with idx.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    (root(vault) / "README.md").write_text(
        "# BitnetCloud\n\nNamed cloud. Lives in this vault. Open notes/ in Obsidian.\n",
        encoding="utf-8",
    )
    row["note"] = str(md_write(vault, text, area))
    return row


def rows(vault: Path) -> list[dict]:
    idx = root(vault) / "index.jsonl"
    if not idx.is_file():
        return []
    out = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
