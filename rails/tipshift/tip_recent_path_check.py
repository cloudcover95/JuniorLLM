#!/usr/bin/env python3
"""Offline stdlib guard: when the repo tip is recent, the new file must sit in a
directory different from the directories touched by the latest commits.
No network, no git calls. Input: lists of paths. Trit rate 0.05.
Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM."""
import hashlib, os, sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(new_path: str, recent_paths: list) -> bool:
    d = os.path.dirname(new_path)
    if not d or ".github" in new_path.split("/"):
        return False
    return d not in {os.path.dirname(p) for p in recent_paths}


def selftest() -> int:
    assert check("rails/tipshift/a.py", ["rails/loopback/x.py"])
    assert not check("rails/loopback/b.py", ["rails/loopback/x.py"])
    assert not check(".github/w.yml", [])
    assert close_hash("", "trit", "x") == close_hash("", "trit", "x")
    assert len(close_hash("a", "b", "c")) == 16
    print("selftest ok rate=%s close=%s" % (RATE, close_hash("", "trit0.05", ",".join(PORTS))))
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if len(sys.argv) < 2 or sys.argv[1] == "selftest" else 2)
