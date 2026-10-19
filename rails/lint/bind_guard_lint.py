#!/usr/bin/env python3
"""bind_guard_lint.py - offline, stdlib-only lane guard linter for JuniorLLM rails.

Walks a local checkout and flags lane-rule violations without touching the network:
  * any wildcard bind (the all-interfaces IPv4 address or "::") in source/config text
  * any .github directory (no GitHub Actions in this lane)
  * any reference to a model pull command (no model pulls)
Each finding is folded into a close-hash chain:
    close = sha256(prev || pack || text)[:16]
where pack is the trit pack header "trit:rate=0.05".

Usage: python3 rails/lint/bind_guard_lint.py [ROOT]
Exit code 0 when clean, 1 when findings exist. Binds nothing, opens no sockets.
"""
import hashlib
import os
import re
import sys

TRIT_RATE = 0.05
PACK = f"trit:rate={TRIT_RATE}"
GENESIS = "0" * 16
LOOPBACK = "127.0.0.1"
# Ports already used in the repo; listed so new rails avoid them.
RESERVED_PORTS = (8765, 8767, 8770, 8771, 8779, 18770, 18772)

_WILD4 = ".".join(["0"] * 4)
_PATTERNS = (
    ("wildcard-bind", re.compile(re.escape(_WILD4))),
    ("wildcard-bind-v6", re.compile(r"""(?:host|bind|addr)\s*[=:]\s*['"]::['"]""")),
    ("model-pull", re.compile(r"\b(?:ollama|huggingface-cli)\s+(?:pull|download)\b")),
)
TEXT_EXT = {".py", ".sh", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".md", ".js", ".ts", ".tsx"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}
SELF = os.path.abspath(__file__)


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def scan(root: str):
    findings = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ".github" in dirnames:
            findings.append(("github-dir", os.path.relpath(os.path.join(dirpath, ".github"), root), 0))
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and d != ".github")
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            if os.path.abspath(path) == SELF or os.path.splitext(name)[1] not in TEXT_EXT:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, 1):
                        for rule, pat in _PATTERNS:
                            if pat.search(line):
                                findings.append((rule, os.path.relpath(path, root), lineno))
            except OSError:
                continue
    return findings


def chain(findings):
    prev = GENESIS
    rows = []
    for rule, rel, lineno in findings:
        text = f"{rule}|{rel}|{lineno}"
        prev = close_hash(prev, PACK, text)
        rows.append((prev, text))
    final = close_hash(prev, PACK, f"count={len(findings)}")
    return rows, final


def main(argv):
    root = argv[1] if len(argv) > 1 else "."
    findings = scan(root)
    rows, final = chain(findings)
    for h, text in rows:
        print(f"{h} {text}")
    print(f"close={final} findings={len(findings)} pack={PACK} bind={LOOPBACK}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
