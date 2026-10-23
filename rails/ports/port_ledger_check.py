#!/usr/bin/env python3
"""rails/ports/port_ledger_check.py - offline stdlib loopback port registry check.

Validates a small registry of named JuniorOS ports against the ports already in
use in this repo, enforces 127.0.0.1-only binds, and emits a close-hash chain.

Named ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Trit pack rate: 0.05.
Close rule: close = sha256(prev || pack || text)[:16].

No network, no model pull, no subprocess. Run: python3 port_ledger_check.py
"""
import hashlib
import json
import sys

TRIT_RATE = 0.05
LOOPBACK = "127.0.0.1"
NAMED_PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
# Ports already used in the repo (bitnetd 8765, i2sd 8767, others, health 8779).
RESERVED = (8765, 8767, 8770, 8771, 8779, 18770, 18772)

REGISTRY = [
    {"name": "Home", "host": LOOPBACK, "port": 8781},
    {"name": "OSai", "host": LOOPBACK, "port": 8782},
    {"name": "FieldCore", "host": LOOPBACK, "port": 8783},
    {"name": "Flagstaff", "host": LOOPBACK, "port": 8784},
    {"name": "JuniorLLM", "host": LOOPBACK, "port": 8785},
]


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def pack_of(rate=TRIT_RATE):
    return "trit:rate=%.2f" % rate


def check(registry):
    errors = []
    seen = set()
    names = [e["name"] for e in registry]
    for n in NAMED_PORTS:
        if n not in names:
            errors.append("missing named port %s" % n)
    for e in registry:
        if e["host"] != LOOPBACK:
            errors.append("%s binds %s, must be %s" % (e["name"], e["host"], LOOPBACK))
        p = e["port"]
        if not (1024 <= p <= 65535):
            errors.append("%s port %d out of range" % (e["name"], p))
        if p in RESERVED:
            errors.append("%s port %d collides with reserved" % (e["name"], p))
        if p in seen:
            errors.append("%s port %d duplicated" % (e["name"], p))
        seen.add(p)
    return errors


def chain(registry, prev="0" * 16):
    pack = pack_of()
    out = []
    for e in registry:
        text = json.dumps(e, sort_keys=True)
        prev = close_hash(prev, pack, text)
        out.append({"entry": e, "close": prev})
    return out


def _selftest():
    assert check(REGISTRY) == []
    bad = [dict(e) for e in REGISTRY]
    bad[0]["host"] = "0.0." + "0.0"
    bad[1]["port"] = 8765
    bad[2]["port"] = bad[3]["port"]
    errs = check(bad)
    assert any("must be" in x for x in errs)
    assert any("reserved" in x for x in errs)
    assert any("duplicated" in x for x in errs)
    assert any("missing" in x for x in check(REGISTRY[1:]))
    c1, c2 = chain(REGISTRY), chain(REGISTRY)
    assert c1 == c2 and len(c1[-1]["close"]) == 16
    assert close_hash("a", "b", "c") == hashlib.sha256(b"abc").hexdigest()[:16]
    return True


if __name__ == "__main__":
    _selftest()
    errs = check(REGISTRY)
    print(json.dumps({"ok": not errs, "errors": errs, "rate": TRIT_RATE,
                      "chain": chain(REGISTRY)}, indent=2))
    sys.exit(1 if errs else 0)
