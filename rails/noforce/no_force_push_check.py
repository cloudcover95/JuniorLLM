#!/usr/bin/env python3
"""Offline stdlib guard: never force-push (LLC / JuniorOS rails).

Flags force-push patterns in argv lists or local text fixtures.
No network, no remotes, no filesystem walk required for selftest.
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

# Patterns that indicate a forced push when scanning command / example text.
_FORCE_TEXT_RE = re.compile(
    r"(?:"
    r"git\s+push\s+[^\n]*--force(?:-with-lease)?\b"
    r"|git\s+push\s+[^\n]*\s-f\b"
    r"|--force-with-lease\b"
    r"|\+refs/"
    r")",
    re.IGNORECASE,
)


def close_hash(prev: str, pack: str, text: str) -> str:
    """sha256(prev || pack || text).hexdigest()[:16]."""
    payload = f"{prev}{pack}{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def is_force_push_argv(argv: List[str]) -> bool:
    """Return True if argv looks like a git force-push invocation."""
    if not argv:
        return False
    tokens = [t for t in argv if t]
    lower = [t.lower() for t in tokens]

    # Must involve push somewhere after git (or bare push as first meaningful).
    try:
        push_i = lower.index("push")
    except ValueError:
        # Also catch `git push` split across typical argv starting with git.
        if "git" in lower and "push" not in lower:
            return False
        return False

    # Force flags after push (or anywhere if clearly force-with-lease / -f).
    after = lower[push_i + 1 :]
    full_after = tokens[push_i + 1 :]

    for t, raw in zip(after, full_after):
        if t in ("--force", "-f", "--force-with-lease"):
            return True
        if t.startswith("--force=") or t.startswith("--force-with-lease="):
            return True
        # Forced refspec: +refs/heads/main:refs/heads/main
        if raw.startswith("+refs/") or (raw.startswith("+") and ":" in raw):
            return True

    # `git push -f` style where -f may appear before other remotes/branches
    # already covered; also scan whole argv for explicit force flags with push.
    if "push" in lower:
        for t in lower:
            if t in ("--force", "--force-with-lease") or t == "-f":
                # Avoid flagging unrelated -f on non-push; require push present.
                return True
    return False


def scan_text_for_force_push(text: str) -> List[str]:
    """Return matching force-push substrings found in local text."""
    return [m.group(0) for m in _FORCE_TEXT_RE.finditer(text or "")]


def scan_paths(paths: Iterable[str]) -> List[Tuple[str, List[str]]]:
    """Optional educational scan of provided paths; empty by default in selftest."""
    findings: List[Tuple[str, List[str]]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError:
            continue
        hits = scan_text_for_force_push(body)
        if hits:
            findings.append((path, hits))
    return findings


def selftest() -> None:
    # Force argv cases
    assert is_force_push_argv(["git", "push", "--force", "origin", "main"])
    assert is_force_push_argv(["git", "push", "-f", "origin", "HEAD"])
    assert is_force_push_argv(["git", "push", "--force-with-lease", "origin", "main"])
    assert is_force_push_argv(
        ["git", "push", "origin", "+refs/heads/main:refs/heads/main"]
    )
    assert is_force_push_argv(["git", "push", "origin", "+main:main"])

    # Safe argv cases
    assert not is_force_push_argv(["git", "push", "origin", "main"])
    assert not is_force_push_argv(["git", "push", "--set-upstream", "origin", "main"])
    assert not is_force_push_argv(["git", "status"])
    assert not is_force_push_argv([])
    assert not is_force_push_argv(["echo", "-f"])

    # Text scanner
    bad = scan_text_for_force_push(
        "example: git push --force origin main\nalso +refs/heads/x:refs/heads/x\n"
    )
    assert bad, "expected force-push hits in fixture text"
    good = scan_text_for_force_push("git push origin main\ngit status\n")
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


def main(argv: Sequence[str] | None = None) -> int:
    _ = argv  # reserved; selftest is the default entry
    try:
        selftest()
    except AssertionError as exc:
        print(f"SELFTEST FAIL: {exc}", file=sys.stderr)
        return 1
    print("no_force_push_check: selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
