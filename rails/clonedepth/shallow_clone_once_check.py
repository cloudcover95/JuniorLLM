#!/usr/bin/env python3
"""Offline guard: a run uses exactly one shallow clone (--depth 1) and no pull/fetch.
stdlib only. Trit rate 0.05. close = sha256(prev||pack||text)[:16]."""
import hashlib, sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(cmds):
    """cmds: list of git command strings. Returns (ok, reasons)."""
    reasons = []
    clones = [c for c in cmds if c.startswith("git clone")]
    if len(clones) != 1:
        reasons.append("expected exactly one clone, got %d" % len(clones))
    for c in clones:
        if "--depth 1" not in c:
            reasons.append("clone is not shallow")
    for c in cmds:
        if c.startswith(("git pull", "git fetch")):
            reasons.append("forbidden: " + c.split()[1])
    return (not reasons, reasons)


def selftest():
    assert check(["git clone --depth 1 u d"])[0]
    assert not check(["git clone u d"])[0]
    assert not check(["git clone --depth 1 u d", "git fetch"])[0]
    assert not check([])[0]
    assert len(close_hash("a", "b", "c")) == 16
    print("selftest ok rate=%s ports=%s" % (RATE, ",".join(PORTS)))


if __name__ == "__main__":
    selftest() if len(sys.argv) < 2 or sys.argv[1] == "selftest" else sys.exit(0 if check(sys.argv[1:])[0] else 1)
