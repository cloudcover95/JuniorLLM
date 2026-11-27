#!/usr/bin/env python3
"""Offline tick-gap guard: checks a list of tick timestamps (seconds) is strictly increasing
with no gap above a limit. Stdlib only, no network. Trit rate 0.05. Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM."""
import hashlib, sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(ticks, max_gap):
    """Return (trit, reason): 1 ok, 0 gap too large, -1 non-monotonic."""
    for a, b in zip(ticks, ticks[1:]):
        if b <= a:
            return -1, "non-monotonic"
        if b - a > max_gap:
            return 0, "gap"
    return 1, "ok"


def selftest():
    assert check([1, 2, 3], 2) == (1, "ok")
    assert check([1, 2, 2], 2)[0] == -1
    assert check([1, 10], 2)[0] == 0
    assert len(close_hash("", "p", "t")) == 16
    assert close_hash("a", "p", "t") != close_hash("b", "p", "t")
    assert RATE == 0.05 and len(PORTS) == 5
    print("selftest ok")


if __name__ == "__main__":
    selftest() if len(sys.argv) < 2 or sys.argv[1] == "selftest" else None
