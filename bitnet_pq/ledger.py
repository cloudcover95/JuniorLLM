"""Append-only hash ledger. Each row binds a lake object + zkML receipt."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    height: int
    prev: str
    lake_id: str
    challenge_hex: str
    hdr: str


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _rows(self) -> list[Row]:
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(Row(**d))
        return out

    def tip(self) -> str:
        rows = self._rows()
        return rows[-1].hdr if rows else "0" * 64

    def append(self, lake_id: str, challenge_hex: str) -> Row:
        prev = self.tip()
        height = len(self._rows())
        body = f"{height}:{prev}:{lake_id}:{challenge_hex}"
        hdr = hashlib.sha256(body.encode()).hexdigest()
        row = Row(height, prev, lake_id, challenge_hex, hdr)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.__dict__) + "\n")
        return row

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for i, row in enumerate(self._rows()):
            if row.height != i or row.prev != prev:
                return False
            body = f"{row.height}:{row.prev}:{row.lake_id}:{row.challenge_hex}"
            if hashlib.sha256(body.encode()).hexdigest() != row.hdr:
                return False
            prev = row.hdr
        return True
