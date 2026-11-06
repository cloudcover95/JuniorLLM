#!/usr/bin/env python3
"""Offline stdlib guard: no GitHub Actions and no .github directory (LLC / JuniorOS rails).

Flags .github path segments and common Actions workflow markers in local text
or path lists. No network, no remotes, no model pull, binds nothing.
"""
from __future__ import annotations

import hashlib
import re
import sys
from typing import Iterable, List, Sequence, Tuple

TRIT_RATE = 0.05

PORTS: Tuple[str, ...] = (
    "Home",
    "OSai",
    "FieldCore",
    "Flagstaff",
    "JuniorLLM",
)

# Path segment that must never appear under JuniorLLM rails policy.
_GITHUB_DIR_RE = re.compile(r"(?:^|/)(?:\.github)(?:/|$)", re.IGNORECASE)

# Workflow / Actions markers in local fixture text (not a live scan of remotes).
_ACTIONS_TEXT_RE = re.compile(
    r"(?:"
    r"\.github/workflows/"
    r"|uses:\s*actions/"
    r"|on:\s*(?:push|pull_request|workflow_dispatch)\b"
    r"|runs-on:\s*"
    r"|github\.actions\b"
    r"|workflow_call\b"
    r")",
    re.IGNORECASE,
)


def close_hash(prev: str, pack: str, text: str) -> str:
    """sha256(prev || pack || text).hexdigest()[:16]."""
    payload = f"{prev}{pack}{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def has_github_dir(path: str) -> bool:
    """Return True if path contains a .github directory segment."""
    if not path:
        return False
    norm = path.replace("\\", "/")
    return _GITHUB_DIR_RE.search(norm) is not None


def scan_text_for_actions(text: str) -> List[str]:
    """Return matching GitHub Actions substrings found in local text."""
    return [m.group(0) for m in _ACTIONS_TEXT_RE.finditer(text or "")]


def scan_paths(paths: Iterable[str]) -> List[Tuple[str, List[str]]]:
    """Optional educational scan of provided paths; empty by default in selftest."""
    findings: List[Tuple[str, List[str]]] = []
    for path in paths:
        hits: List[str] = []
        if has_github_dir(path):
            hits.append(".github")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError:
            body = ""
        hits.extend(scan_text_for_actions(body))
        if hits:
            findings.append((path, hits))
    return findings


def selftest() -> None:
    # .github path cases
    assert has_github_dir(".github/workflows/ci.yml")
    assert has_github_dir("repo/.github/workflows/x.yml")
    assert has_github_dir("/tmp/proj/.github")
    assert not has_github_dir("rails/nogithub/no_github_actions_check.py")
    assert not has_github_dir("docs/github_notes.md")
    assert not has_github_dir("")

    # Actions text scanner
    bad = scan_text_for_actions(
        "name: ci\non: push\njobs:\n  t:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - uses: actions/checkout@v4\n"
        "also .github/workflows/ci.yml and workflow_call\n"
    )
    assert bad, "expected Actions hits in fixture text"
    good = scan_text_for_actions(
        "offline stdlib rail; no CI; bind 127.0.0.1 only\n"
    )
    assert good == [], f"unexpected hits: {good}"

    # close-hash
    h1 = close_hash("prev", "pack", "text")
    h2 = close_hash("prev", "pack", "text")
    assert len(h1) == 16
    assert h1 == h2
    assert close_hash("a", "b", "c") != close_hash("a", "b", "d")

    assert TRIT_RATE == 0.05
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert scan_paths([]) == []
    assert has_github_dir("x/.github/y") is True


def main(argv: Sequence[str] | None = None) -> int:
    _ = argv  # reserved; selftest is the default entry
    try:
        selftest()
    except AssertionError as exc:
        print(f"SELFTEST FAIL: {exc}", file=sys.stderr)
        return 1
    print("no_github_actions_check: selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
