"""Obsidian-shaped notes. Frontmatter is local edgepack, not a sync service."""
from __future__ import annotations

from pathlib import Path

from junior_bitnet.edgepack import edgepack


def render(text: str, area: str) -> str:
    ep = edgepack(text)
    slug = "".join(c if c.isalnum() else "-" for c in text.lower())[:40].strip("-")
    return (
        f"---\narea: {area}\nlocal: true\ncloud: false\ni2s: {ep['i2s']}\n---\n\n# {text}\n\n#{area} #bitnet\n"
    ), slug or "note"


def write(vault: Path, text: str, area: str) -> Path:
    body, slug = render(text, area)
    d = Path(vault) / "bitnetCloud" / "notes"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{area}-{slug}.md"
    p.write_text(body, encoding="utf-8")
    return p
