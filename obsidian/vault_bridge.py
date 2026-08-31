"""JuniorSkillVault — Obsidian-compatible markdown notes + SKILL.md pins."""
from __future__ import annotations

from pathlib import Path

from agent.guardrails import hash_text, scan_skill_source


def write_note(vault: Path, rel: str, body: str) -> Path:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def pin_skill(vault: Path, name: str, src: str) -> dict:
    rail = scan_skill_source(src)
    dest = vault / "skills" / name / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src, encoding="utf-8")
    meta = vault / "skills" / name / "HASH.txt"
    meta.write_text(rail.skill_hash or hash_text(src), encoding="utf-8")
    return {"ok": rail.ok, "reasons": rail.reasons, "path": str(dest)}
