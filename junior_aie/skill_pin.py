"""T5 — load SKILL.md under a root and pin sha256. Never fetch. Never exec."""
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
DENY_BODY = ("eval(", "exec(", "0.0.0.0")
ZERO = "0" * 64
SKILL_NAME = "SKILL.md"


class SkillDenied(Exception):
    """Path, size, name, pin mismatch, or banned body refused."""


@dataclass(frozen=True)
class SkillRow:
    height: int
    prev: str
    op: str
    name: str
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


def parse_frontmatter(text: str) -> tuple[str, str]:
    """Return (name, description) from YAML-ish frontmatter. No eval."""
    name = ""
    desc = ""
    if not text.startswith("---"):
        return name, desc
    rest = text[3:]
    end = rest.find("\n---")
    if end < 0:
        return name, desc
    block = rest[:end]
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("\"'")
        elif line.startswith("description:"):
            desc = line.split(":", 1)[1].strip().strip("\"'")
    return name, desc


class SkillPins:
    """Append-only pin log for SKILL.md files under ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "SKILL_PINS.jsonl"
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _rows(self) -> list[SkillRow]:
        out: list[SkillRow] = []
        raw = self.path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            if line.strip():
                out.append(SkillRow(**json.loads(line)))
        return out

    def tip(self) -> str:
        rows = self._rows()
        return rows[-1].hdr if rows else ZERO

    def _resolve(self, rel: str) -> Path:
        text = (rel or "").strip()
        if not text:
            raise SkillDenied("empty_path")
        low = text.replace("\\", "/").lower()
        if text.startswith("/") or text.startswith("~"):
            raise SkillDenied("absolute_path")
        if ".." in Path(text).parts or ".." in low:
            raise SkillDenied("path_escape")
        if any(frag in low for frag in DENY_FRAGMENTS):
            raise SkillDenied("denied_name")
        target = (self.root / text).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise SkillDenied("path_escape") from exc
        return target

    def _append(self, op: str, name: str, rel: str, digest: str, size: int) -> SkillRow:
        prev = self.tip()
        height = len(self._rows())
        at = _now()
        body = f"{height}:{prev}:{op}:{name}:{rel}:{digest}:{size}:{at}"
        hdr = hashlib.sha256(body.encode()).hexdigest()
        row = SkillRow(height, prev, op, name, rel, digest, size, at, hdr)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(row)) + "\n")
        return row

    def _scan_body(self, text: str) -> None:
        low = text.lower()
        for needle in DENY_BODY:
            if needle in low:
                raise SkillDenied("banned_body")

    def discover(self, skills_dir: str = "skills") -> list[str]:
        base = self._resolve(skills_dir)
        if not base.is_dir():
            return []
        found: list[str] = []
        for child in sorted(base.iterdir()):
            skill = child / SKILL_NAME
            if child.is_dir() and skill.is_file():
                found.append(f"{skills_dir}/{child.name}/{SKILL_NAME}")
        return found

    def pin(self, rel: str) -> SkillRow:
        target = self._resolve(rel)
        if not target.is_file() or target.name != SKILL_NAME:
            raise SkillDenied("not_skill_md")
        data = target.read_bytes()
        if too_big(len(data)):
            raise SkillDenied("cap_8gb")
        text = data.decode("utf-8")
        self._scan_body(text)
        name, _desc = parse_frontmatter(text)
        if not name:
            name = target.parent.name
        return self._append("pin", name, rel, sha256_bytes(data), len(data))

    def latest_pin(self, name: str) -> SkillRow | None:
        hit = None
        for row in self._rows():
            if row.op == "pin" and (row.name == name or row.rel == name):
                hit = row
        return hit

    def load(self, rel: str) -> tuple[str, SkillRow]:
        target = self._resolve(rel)
        if not target.is_file() or target.name != SKILL_NAME:
            raise SkillDenied("not_skill_md")
        data = target.read_bytes()
        if too_big(len(data)):
            raise SkillDenied("cap_8gb")
        text = data.decode("utf-8")
        self._scan_body(text)
        name, _desc = parse_frontmatter(text)
        if not name:
            name = target.parent.name
        digest = sha256_bytes(data)
        pinned = self.latest_pin(name) or self.latest_pin(rel)
        if pinned is None:
            raise SkillDenied("unpinned")
        if pinned.sha256 != digest:
            raise SkillDenied("pin_mismatch")
        row = self._append("load", name, rel, digest, len(data))
        return text, row

    def verify_chain(self) -> bool:
        prev = ZERO
        for i, row in enumerate(self._rows()):
            if row.height != i or row.prev != prev:
                return False
            body = (
                f"{row.height}:{row.prev}:{row.op}:{row.name}:"
                f"{row.rel}:{row.sha256}:{row.size}:{row.at}"
            )
            if hashlib.sha256(body.encode()).hexdigest() != row.hdr:
                return False
            prev = row.hdr
        return True
