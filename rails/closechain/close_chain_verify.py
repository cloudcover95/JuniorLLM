#!/usr/bin/env python3
"""Offline stdlib verifier for the close-hash chain.

close = sha256(prev || pack || text)[:16]. Trit pack rate 0.05.
Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM. No network, no model pull,
127.0.0.1 only (this tool binds nothing).
"""
import hashlib
import sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def verify(entries, genesis="0" * 16, pack="trit"):
    """entries: list of (text, claimed_close). Returns index of first bad link or -1."""
    prev = genesis
    for i, (text, claimed) in enumerate(entries):
        cur = close(prev, pack, text)
        if cur != claimed:
            return i
        prev = cur
    return -1


def selftest():
    g = "0" * 16
    a = close(g, "trit", "one")
    b = close(a, "trit", "two")
    assert verify([("one", a), ("two", b)]) == -1
    assert verify([("one", a), ("two", "bad")]) == 1
    assert len(a) == 16 and RATE == 0.05 and len(PORTS) == 5
    print("selftest ok")


if __name__ == "__main__":
    selftest() if len(sys.argv) > 1 and sys.argv[1] == "selftest" else print(__doc__)
