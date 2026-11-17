#!/usr/bin/env python3
"""Offline stdlib guard: a repo with a trit pack must use rate 0.05.
Local only, no network, no model pull, no remote scripts. Lane: JuniorLLM rails.
Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
close = sha256(prev || pack || text)[:16]
"""
import hashlib
import sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check_rate(has_pack: bool, rate: float) -> bool:
    """True when the rate is acceptable for the repo."""
    return (not has_pack) or abs(rate - RATE) < 1e-12


def selftest() -> int:
    assert check_rate(True, 0.05)
    assert not check_rate(True, 0.1)
    assert check_rate(False, 0.1)
    h = close_hash("0" * 16, "trit", "rate-pack")
    assert len(h) == 16 and h == close_hash("0" * 16, "trit", "rate-pack")
    print("selftest ok close=" + h + " ports=" + ",".join(PORTS))
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
