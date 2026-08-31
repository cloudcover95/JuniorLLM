"""JuniorGuardrailPipeline — skill/plugin/scan + covenant + no-exec rails."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, asdict
from typing import Any

BANNED_CALLS = {"eval", "exec", "compile", "__import__"}
BANNED_MODULES = {"ctypes", "subprocess", "socket", "http.client"}


@dataclass
class RailResult:
    ok: bool
    reasons: list[str]
    skill_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def scan_skill_source(src: str) -> RailResult:
    reasons: list[str] = []
    try:
        tree = ast.parse(src or "")
    except SyntaxError as exc:
        return RailResult(False, [f"syntax:{exc}"], hash_text(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                reasons.append(f"banned_call:{node.func.id}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BANNED_MODULES:
                    reasons.append(f"banned_import:{top}")
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in BANNED_MODULES:
                reasons.append(f"banned_import:{top}")
    return RailResult(not reasons, reasons, hash_text(src))


def scan_prompt(prompt: str) -> RailResult:
    p = (prompt or "").lower()
    reasons = []
    needles = [
        "ignore previous instructions",
        "disable guardrail",
        "exfiltrate",
        "drop table",
    ]
    for n in needles:
        if n in p:
            reasons.append(f"prompt_injection:{n}")
    return RailResult(not reasons, reasons, hash_text(prompt))


def memory_allowed(topic: str, include_sensitive: bool) -> bool:
    sensitive = {"health", "beliefs", "medical", "religion", "politics"}
    if topic.lower() in sensitive and not include_sensitive:
        return False
    return True
