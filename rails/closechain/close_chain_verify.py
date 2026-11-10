#!/usr/bin/env python3
"""Offline stdlib verifier for a close-hash chain (JuniorLLM rail).

close = sha256(prev || pack || text)[:16]; trit pack rate 0.05.
Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM. No network, no binds (127.0.0.1 only if ever served).
"""
import hashlib
import sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def verify(entries, genesis="0" * 16):
    """entries: list of (pack, text, claimed_close). Returns (ok, bad_index)."""
    prev = genesis
    for i, (pack, text, claimed) in enumerate(entries):
        calc = close_hash(prev, pack, text)
        if calc != claimed:
            return False, i
        prev = calc
    return True, -1


def selftest():
    prev = "0" * 16
    entries = []
    for t in ("a", "b", "c"):
        c = close_hash(prev, "trit", t)
        entries.append(("trit", t, c))
        prev = c
    assert verify(entries) == (True, -1)
    entries[1] = ("trit", "tampered", entries[1][2])
    assert verify(entries) == (False, 1)
    assert RATE == 0.05 and len(PORTS) == 5
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        selftest()
