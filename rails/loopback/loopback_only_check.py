#!/usr/bin/env python3
"""Offline stdlib guard: every bind address in a config/text must be 127.0.0.1 (never 0.0.0.0).
JuniorCloud LLC rail. Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM. No network, no model pull."""
import hashlib, re, sys

TRIT_RATE = 0.05
PORTS = ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]
BAD = re.compile(r"(0\.0\.0\.0|::\s*\]|host\s*[=:]\s*['\"]?\*)")

def scan(text):
    return [i + 1 for i, l in enumerate(text.splitlines()) if BAD.search(l)]

def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]

def selftest():
    assert scan("bind=127.0.0.1:8080") == []
    assert scan("a\nhost = 0.0.0.0") == [2]
    assert len(close_hash("p", "trit", "t")) == 16
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        print("ok" if selftest() else "fail"); sys.exit(0)
    data = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    hits = scan(data)
    print("rate", TRIT_RATE, "ports", ",".join(PORTS), "violations", hits,
          "close", close_hash("prev", "trit0.05", data))
    sys.exit(1 if hits else 0)
