#!/usr/bin/env python3
"""Offline stdlib guard: one repo per run (LLC / JuniorOS rails).

A builder run may name and clone exactly one target repository, and that
target must be a lane repo (never JuniorHome). No network, no remotes,
no model pull, binds nothing.
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

LANE_REPOS: Tuple[str, ...] = (
    "cloudcover95/JuniorLLM",
    "cloudcover95/JuniorOS",
    "cloudcover95/JuniorPiThon",
    "cloudcover95/JuniorPiPython",
)

FORBIDDEN_REPOS: Tuple[str, ...] = (
    "cloudcover95/JuniorHome",
)

# owner/name in a URL, scp-style remote, or bare slug.
_REPO_RE = re.compile(
    r"(?:(?:https?://|git@)[^/\s:]+[:/])?"
    r"(cloudcover95/(?:JuniorLLM|JuniorOS|JuniorPiThon|JuniorPiPython|JuniorHome))"
    r"(?:\.git)?",
    re.IGNORECASE,
)

_CLONE_RE = re.compile(r"\bgit\s+clone\b", re.IGNORECASE)


def close_hash(prev: str, pack: str, text: str) -> str:
    """sha256(prev || pack || text).hexdigest()[:16]."""
    payload = f"{prev}{pack}{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def normalize_repo(value: str) -> str:
    """Return owner/name if value names a known lane or forbidden repo."""
    if not value:
        return ""
    text = value.strip().replace("\\", "/")
    match = _REPO_RE.search(text)
    if not match:
        return ""
    owner, name = match.group(1).split("/", 1)
    return f"{owner.lower()}/{name}"


def repos_named(values: Iterable[str]) -> List[str]:
    """Return unique normalized repos, preserving first-seen order."""
    seen = set()
    out: List[str] = []
    for value in values:
        repo = normalize_repo(value)
        if repo and repo not in seen:
            seen.add(repo)
            out.append(repo)
    return out


def scan_text_for_repos(text: str) -> List[str]:
    """Return unique normalized repos found in local fixture text."""
    if not text:
        return []
    found = [m.group(1) for m in _REPO_RE.finditer(text)]
    return repos_named(found)


def clone_count(text: str) -> int:
    """Count git clone invocations in local fixture text."""
    if not text:
        return 0
    return len(_CLONE_RE.findall(text))


def one_repo_ok(repos: Sequence[str]) -> bool:
    """True when exactly one distinct lane repo is named and none are forbidden."""
    unique = repos_named(repos)
    if len(unique) != 1:
        return False
    target = unique[0]
    if target in {normalize_repo(item) for item in FORBIDDEN_REPOS}:
        return False
    return target in {normalize_repo(item) for item in LANE_REPOS}


def run_plan_ok(text: str) -> bool:
    """True when fixture text names one lane repo and at most one clone."""
    repos = scan_text_for_repos(text)
    if not one_repo_ok(repos):
        return False
    return clone_count(text) <= 1


def scan_paths(paths: Iterable[str]) -> List[Tuple[str, List[str]]]:
    """Optional educational scan of provided paths; empty by default in selftest."""
    findings: List[Tuple[str, List[str]]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError:
            continue
        repos = scan_text_for_repos(body)
        issues: List[str] = []
        if not one_repo_ok(repos):
            issues.extend(repos or ["<none>"])
        if clone_count(body) > 1:
            issues.append("multiple-clone")
        if issues:
            findings.append((path, issues))
    return findings


def selftest() -> None:
    assert normalize_repo("cloudcover95/JuniorLLM") == "cloudcover95/JuniorLLM"
    assert (
        normalize_repo("https://github.com/cloudcover95/JuniorLLM.git")
        == "cloudcover95/JuniorLLM"
    )
    assert (
        normalize_repo("git@github.com:cloudcover95/JuniorHome.git")
        == "cloudcover95/JuniorHome"
    )
    assert normalize_repo("unrelated/repo") == ""
    assert normalize_repo("") == ""

    assert one_repo_ok(["cloudcover95/JuniorLLM"])
    assert one_repo_ok(
        ["https://github.com/cloudcover95/JuniorLLM.git", "cloudcover95/JuniorLLM"]
    )
    assert one_repo_ok(["cloudcover95/JuniorOS"])
    assert not one_repo_ok([])
    assert not one_repo_ok(["cloudcover95/JuniorLLM", "cloudcover95/JuniorOS"])
    assert not one_repo_ok(["cloudcover95/JuniorHome"])
    assert not one_repo_ok(["cloudcover95/JuniorLLM", "cloudcover95/JuniorHome"])

    plan = (
        "target cloudcover95/JuniorLLM\n"
        "git clone --depth 1 https://github.com/cloudcover95/JuniorLLM.git\n"
    )
    assert run_plan_ok(plan)
    assert clone_count(plan) == 1
    assert scan_text_for_repos(plan) == ["cloudcover95/JuniorLLM"]

    two_repos = "clone cloudcover95/JuniorLLM and cloudcover95/JuniorPiThon\n"
    assert not run_plan_ok(two_repos)
    two_clones = (
        "git clone --depth 1 https://github.com/cloudcover95/JuniorLLM.git\n"
        "git clone --depth 1 https://github.com/cloudcover95/JuniorLLM.git\n"
    )
    assert not run_plan_ok(two_clones)
    home = "read nothing in cloudcover95/JuniorHome except docs/LANES.md\n"
    assert not run_plan_ok(home)

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
    print("one_repo_check: selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
