#!/usr/bin/env python3
"""Offline guard: a run uses exactly one shallow clone (--depth 1), no pull/fetch.
Stdlib only. Trit rate 0.05. Close hash = sha256(prev||pack||text)[:16]."""
import hashlib, re, sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(cmds):
    clones = [c for c in cmds if re.search(r"\bgit\s+clone\b", c)]
    bad = [c for c in cmds if re.search(r"\bgit\s+(pull|fetch)\b", c)]
    shallow = all("--depth 1" in c or "--depth=1" in c for c in clones)
    return len(clones) == 1 and shallow and not bad


def selftest():
    assert check(["git clone --depth 1 u d", "git push"])
    assert not check(["git clone u d"])
    assert not check(["git clone --depth 1 u d", "git fetch"])
    assert not check(["git clone --depth 1 a", "git clone --depth 1 b"])
    assert len(close_hash("a", "b", "c")) == 16
    assert RATE == 0.05
    print("ok", close_hash("", "trit@%s" % RATE, ",".join(PORTS)))


if __name__ == "__main__":
    selftest() if len(sys.argv) < 2 else print(check(sys.argv[1:]))
