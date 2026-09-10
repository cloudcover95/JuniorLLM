"""Local content-addressed data lake (files on disk). No vendor warehouse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class Lake:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.obj = self.root / "obj"
        self.obj.mkdir(parents=True, exist_ok=True)

    def put(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True).encode()
        hid = hashlib.sha256(raw).hexdigest()
        (self.obj / hid).write_bytes(raw)
        return hid

    def get(self, hid: str) -> dict[str, Any]:
        return json.loads((self.obj / hid).read_text(encoding="utf-8"))
