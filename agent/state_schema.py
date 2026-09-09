"""A9 — night STATE + receipt JSON (TEST_ROADMAP T3)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED = ("at", "slice", "port", "status", "next")


def receipt(
    slice_id: str,
    port: str,
    status: str,
    nxt: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "at": datetime.now(timezone.utc).isoformat(),
        "slice": slice_id,
        "port": port,
        "status": status,
        "next": nxt,
    }
    if extra:
        body.update(extra)
    return body


def validate(body: dict[str, Any]) -> list[str]:
    return [k for k in REQUIRED if not body.get(k)]


def write_pair(root: Path, body: dict[str, Any]) -> None:
    miss = validate(body)
    if miss:
        raise ValueError(f"missing {miss}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "STATE.md").write_text(
        f"# STATE\n\n- at: {body['at']}\n- slice: {body['slice']}\n- port: {body['port']}\n- status: {body['status']}\n- next: {body['next']}\n",
        encoding="utf-8",
    )
    (root / "grok_bot" / "LAST_RECEIPT.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "grok_bot" / "LAST_RECEIPT.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
