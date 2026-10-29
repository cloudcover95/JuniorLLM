#!/usr/bin/env python3
"""Loopback bind check for JuniorOS / JuniorLLM rails.

Offline, stdlib only. Scans text (config, scripts, docs) for host bind
values and passes only if every bind is loopback (127.0.0.1 / localhost).
Any 0.0.0.0 or :: wildcard bind fails. Nothing is launched, nothing is
bound; this only reads text.

Close rule: close = sha256(prev || pack || text)[:16].
Trit pack rate: 0.05.

Usage:
  python3 loopback_bind_check.py FILE [FILE ...]
  python3 loopback_bind_check.py --selftest
"""
import hashlib
import re
import sys

TRIT_RATE = 0.05
PACK = "rails/bind:loopback:v1"
LOOPBACK = {"127.0.0.1", "localhost"}
WILDCARD = {"0.0.0.0", "::", "[::]"}

_BIND_RE = re.compile(
    r"""(?ix)
    (?:host|bind|listen|address|addr)\s*[=:]\s*["']?
    (?P<h>\[::\]|::|[0-9]{1,3}(?:\.[0-9]{1,3}){3}|localhost)
    """
)


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def scan(text):
    """Return list of (line_no, host, ok)."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in _BIND_RE.finditer(line):
            h = m.group("h").lower()
            out.append((i, h, h in LOOPBACK))
        for w in ("0.0.0.0",):
            if w in line and not any(r[0] == i and r[1] == w for r in out):
                out.append((i, w, False))
    return out


def check(name, text, prev="0" * 16):
    hits = scan(text)
    bad = [h for h in hits if not h[2]]
    close = close_hash(prev, PACK, text)
    return {
        "file": name,
        "binds": len(hits),
        "bad": [(ln, h) for ln, h, _ in bad],
        "ok": not bad,
        "trit_rate": TRIT_RATE,
        "close": close,
    }


def selftest():
    good = 'host = "127.0.0.1"\nport = 8785\nlisten: localhost\n'
    bad = 'bind = "0.0.0.0"\n'
    bad6 = "address: ::\n"
    r1 = check("good", good)
    assert r1["ok"] and r1["binds"] == 2, r1
    r2 = check("bad", bad)
    assert not r2["ok"] and r2["bad"] == [(1, "0.0.0.0")], r2
    r3 = check("bad6", bad6)
    assert not r3["ok"], r3
    r4 = check("bare", "server.run(0.0.0.0)\n")
    assert not r4["ok"], r4
    assert TRIT_RATE == 0.05
    c1 = close_hash("0" * 16, PACK, good)
    assert len(c1) == 16 and c1 == r1["close"]
    c2 = close_hash(c1, PACK, bad)
    assert c2 != c1 and len(c2) == 16
    assert check("bad", bad, prev=c1)["close"] == c2
    print("selftest ok close=%s next=%s" % (c1, c2))
    return 0


def main(argv):
    if not argv or argv[0] == "--selftest":
        return selftest()
    prev = "0" * 16
    failed = False
    for path in argv:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        r = check(path, text, prev)
        prev = r["close"]
        failed |= not r["ok"]
        print("%s binds=%d ok=%s close=%s bad=%s" % (
            path, r["binds"], r["ok"], r["close"], r["bad"]))
    print("chain close=%s trit_rate=%.2f" % (prev, TRIT_RATE))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
