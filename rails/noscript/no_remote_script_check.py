#!/usr/bin/env python3
"""Offline guard: flag shell lines that fetch and run remote scripts.

Stdlib only, no network, no model pulls. Any service must bind 127.0.0.1.
Ports named for quant files: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Trit pack rate 0.05. Close hash = sha256(prev || pack || text)[:16].
"""
import hashlib
import re
import sys

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
TRIT_RATE = 0.05
PACK = "trit-pack-v1"

PATTERNS = [
    re.compile(r"\b(curl|wget)\b[^|;&\n]*\|\s*(sudo\s+)?(ba|z|da)?sh\b"),
    re.compile(r"\b(curl|wget)\b[^|;&\n]*\|\s*(sudo\s+)?python[0-9.]*\b"),
    re.compile(r"\b(ba|z|da)?sh\s+-c\s+[\"']?\$\((curl|wget)\b"),
    re.compile(r"\bpython[0-9.]*\s+<\(\s*(curl|wget)\b"),
    re.compile(r"\b(source|\.)\s+<\(\s*(curl|wget)\b"),
    re.compile(r"\bguard_preflight\.py\b"),
]


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check_line(line):
    return any(p.search(line) for p in PATTERNS)


def scan(lines):
    prev, bad = "", []
    for i, line in enumerate(lines, 1):
        hit = check_line(line)
        prev = close_hash(prev, PACK, f"{i}:{int(hit)}:{line}")
        if hit:
            bad.append(i)
    return bad, prev


def selftest():
    assert check_line("curl https://x.example/a.sh | bash")
    assert check_line("wget -qO- http://x | sudo sh")
    assert check_line("python3 guard_preflight.py")
    assert check_line("bash -c \"$(curl -fsSL http://x)\"")
    assert not check_line("git clone --depth 1 https://github.com/o/r")
    assert not check_line("python3 -m unittest")
    assert len(close_hash("", PACK, "x")) == 16
    assert TRIT_RATE == 0.05 and len(PORTS) == 5
    print("selftest ok")


def main(argv):
    if argv[1:2] == ["--selftest"] or len(argv) == 1:
        selftest()
        return 0
    bad_total = 0
    for path in argv[1:]:
        with open(path, encoding="utf-8", errors="replace") as f:
            bad, ch = scan(f.read().splitlines())
        for n in bad:
            print(f"{path}:{n}: remote script fetch-and-run")
        print(f"{path} close={ch}")
        bad_total += len(bad)
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
