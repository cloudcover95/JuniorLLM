#!/usr/bin/env python3
"""Offline stdlib guard: a clone command line must be a single shallow clone (--depth 1). Trit rate 0.05."""
import hashlib, shlex, sys

RATE = 0.05
PORTS = ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]

def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]

def check(cmds):
    clones = [shlex.split(c) for c in cmds if shlex.split(c)[:2] == ["git", "clone"]]
    problems = []
    if len(clones) != 1:
        problems.append("expected exactly one clone, got %d" % len(clones))
    for c in clones:
        ok = "--depth" in c and c[c.index("--depth") + 1:c.index("--depth") + 2] == ["1"] or "--depth=1" in c
        if not ok:
            problems.append("clone is not --depth 1")
    return problems

def selftest():
    assert check(["git clone --depth 1 https://x/y.git"]) == []
    assert check(["git clone https://x/y.git"]) != []
    assert check(["git clone --depth=1 a", "git clone --depth 1 b"]) != []
    assert len(close_hash("p", "pack", "t")) == 16
    return True

if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        print("selftest", selftest(), "rate", RATE, "ports", ",".join(PORTS))
    else:
        probs = check(sys.argv[1:])
        print(probs or "ok")
        sys.exit(1 if probs else 0)
