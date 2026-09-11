"""Public observation ledger. Never writes z."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from junior_bitnet.refprop import Library


def append(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lib = Library()
    now = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as fh:
        for name in lib.names():
            obs = lib.palace.observe(name.lower())
            fh.write(
                json.dumps(
                    {
                        "at": now,
                        "fluid": name,
                        "rho": obs["rho"],
                        "phase": obs["phase"],
                        "commit_unchanged": obs["commit_unchanged"],
                        "backend": obs["backend"],
                    }
                )
                + "\n"
            )
    return path
