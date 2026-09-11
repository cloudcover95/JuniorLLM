#!/usr/bin/env python3
"""One shot: prove + vault + ledger + cold table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junior_bitnet.coolstore import dump
from junior_bitnet.ledger import append
from junior_bitnet.prove import prove
from junior_bitnet.vault import write_vault


def main(argv: list[str]) -> int:
    dest = Path(argv[1]) if len(argv) > 1 else Path("/tmp/juniorhome_vault")
    note = write_vault(dest)
    led = append(dest / "JuniorTeqp" / "observations.jsonl")
    cold = dump(dest / "JuniorTeqp" / "coolstore.json")
    out = prove()
    print(
        json.dumps(
            {
                "prove": out["ok"],
                "note": str(note),
                "ledger": str(led),
                "coolstore": str(cold),
                "fluids": out.get("catalog"),
            },
            indent=2,
        )
    )
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
