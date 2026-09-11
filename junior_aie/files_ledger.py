"""T4 — JuniorFileLedger. Local create/read under a root. Never fetch."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

CAP_BYTES = 8 * 1024 * 1024 * 1024
DENY_FRAGMENTS = (
    "docker.sock",
    "device-login",
    "github_pat_",
    "gpt-6",
    "gpt6-astra",
)
ZERO = "0" * 64


class LedgerDenied(Exception):
    """Path, size, or name refused by the Files ledger."""


@dataclass(frozen=True)
class FileRow:
    height: int
    prev: str
    op: str
    rel: str
    sha256: str
    size: int
    at: str
    hdr: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def too_big(n: int) -> bool:
    return n > CAP_BYTES


class FilesLedger:
    """Append-only create/read log bound to files under ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "FILES.jsonl"
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _rows(self) -> list[FileRow]:
        out: list[FileRow] = []
        raw = self.path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            if line.strip():
                out.append(FileRow(**json.loads(line)))
        return out

    def tip(self) -> str:
        rows = self._rows()
        return rows[-1].hdr if rows else ZERO

    def _resolve(self, rel: str) -> Path:
        text = (rel or "").strip()
        if not text:
            raise LedgerDenied("empty_path")
        low = text.replace("\\", "/").lower()
        if text.startswith("/") or text.startswith("~"):
            raise LedgerDenied("absolute_path")
        if ".." in Path(text).parts or ".." in low:
            raise LedgerDenied("path_escape")
        if any(frag in low for frag in DENY_FRAGMENTS):
            raise LedgerDenied("denied_name")
        target = (self.root / text).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise LedgerDenied("path_escape") from exc
        return target

    def _append(self, op: str, rel: str, digest: str, size: int) -> FileRow:
        prev = self.tip()
        height = len(self._rows())
        at = _now()
        body = f"{height}:{prev}:{op}:{rel}:{digest}:{size}:{at}"
        hdr = hashlib.sha256(body.encode()).hexdigest()
        row = FileRow(height, prev, op, rel, digest, size, at, hdr)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(row)) + "\n")
        return row

    def create(self, rel: str, data: bytes) -> FileRow:
        if too_big(len(data)):
            raise LedgerDenied("cap_8gb")
        target = self._resolve(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return self._append("create", rel, sha256_bytes(data), len(data))

    def read(self, rel: str) -> tuple[bytes, FileRow]:
        target = self._resolve(rel)
        if not target.is_file():
            raise LedgerDenied("missing")
        data = target.read_bytes()
        if too_big(len(data)):
            raise LedgerDenied("cap_8gb")
        row = self._append("read", rel, sha256_bytes(data), len(data))
        return data, row

    def verify_chain(self) -> bool:
        prev = ZERO
        for i, row in enumerate(self._rows()):
            if row.height != i or row.prev != prev:
                return False
            body = f"{row.height}:{row.prev}:{row.op}:{row.rel}:{row.sha256}:{row.size}:{row.at}"
            if hashlib.sha256(body.encode()).hexdigest() != row.hdr:
                return False
            prev = row.hdr
        return True
