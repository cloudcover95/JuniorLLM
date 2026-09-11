"""Write JuniorTeqp notes a vault can ingest. Stdlib only."""
from __future__ import annotations

from pathlib import Path

from junior_bitnet.refprop import Library


def write_vault(root: Path) -> Path:
    root = Path(root)
    dest = root / "JuniorTeqp"
    dest.mkdir(parents=True, exist_ok=True)
    lib = Library()
    note = dest / "property_table.md"
    note.write_text(lib.markdown(), encoding="utf-8")
    (dest / "fluids.json").write_text(
        __import__("json").dumps(lib.table(), indent=2),
        encoding="utf-8",
    )
    return note
