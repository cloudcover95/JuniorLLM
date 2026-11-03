#!/usr/bin/env python3
"""rails/quantports/quant_ports_check.py

Offline, stdlib-only guard: every quant file must name all five ports
(Home, OSai, FieldCore, Flagstaff, JuniorLLM). A file counts as a quant
file when its path or text mentions "quant". Non-quant files pass.

Also extends the close-hash chain: close = sha256(prev || pack || text)[:16],
with trit pack rate 0.05. No network, no model pull, binds nothing.

Usage:
  python3 quant_ports_check.py [paths...]   # scan files/dirs (default: .)
  python3 quant_ports_check.py --selftest
"""
import hashlib
import os
import re
import sys

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
TRIT_RATE = 0.05
PACK = "trit:rate=%.2f" % TRIT_RATE
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
EXTS = (".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt")


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def is_quant(path, text):
    return "quant" in path.lower() or re.search(r"\bquant", text, re.I) is not None


def missing_ports(text):
    return [p for p in PORTS if re.search(r"\b%s\b" % re.escape(p), text) is None]


def check_text(path, text):
    if not is_quant(path, text):
        return []
    return missing_ports(text)


def iter_files(paths):
    for root in paths:
        if os.path.isfile(root):
            yield root
            continue
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                if f.endswith(EXTS):
                    yield os.path.join(d, f)


def scan(paths):
    prev = "0" * 16
    bad = 0
    for p in sorted(iter_files(paths)):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        miss = check_text(p, text)
        prev = close_hash(prev, PACK, p + "|" + ",".join(miss))
        if miss:
            bad += 1
            print("FAIL %s missing ports: %s" % (p, ", ".join(miss)))
    print("close=%s rate=%.2f failures=%d" % (prev, TRIT_RATE, bad))
    return 1 if bad else 0


def selftest():
    full = "quant desk ports: Home OSai FieldCore Flagstaff JuniorLLM"
    assert check_text("x/quant_a.py", full) == []
    assert check_text("x/quant_b.py", "quant: Home OSai") == ["FieldCore", "Flagstaff", "JuniorLLM"]
    assert check_text("x/notes.md", "nothing here") == []
    assert check_text("x/q.py", "Quantile? no ports") != []  # conservative: \bquant matches
    assert check_text("x/q.py", "OSaix only quant") == ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]
    c1 = close_hash("0" * 16, PACK, "a")
    assert len(c1) == 16 and c1 == close_hash("0" * 16, PACK, "a")
    assert c1 != close_hash(c1, PACK, "a")
    assert TRIT_RATE == 0.05
    print("selftest ok close=%s" % c1)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]:
        sys.exit(selftest())
    sys.exit(scan(args or ["."]))
