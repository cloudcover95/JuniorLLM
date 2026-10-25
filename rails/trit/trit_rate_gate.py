#!/usr/bin/env python3
"""JuniorOS rail: trit rate gate (offline, stdlib only).

Checks that every trit-pack record declares the adopted rate 0.05 and that
each record's close hash follows the close rule:

    close = sha256(prev || pack || text)[:16]

No network, no model pull, no services bound. JuniorCloud LLC.
"""
import hashlib
import json
import sys

TRIT_RATE = 0.05
GENESIS = "0" * 16


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def build(records, prev=GENESIS):
    out = []
    for r in records:
        c = close_hash(prev, r["pack"], r["text"])
        out.append(dict(r, rate=r.get("rate", TRIT_RATE), prev=prev, close=c))
        prev = c
    return out


def gate(chain):
    """Return list of problems; empty means the gate passes."""
    problems = []
    prev = GENESIS
    for i, r in enumerate(chain):
        if abs(float(r.get("rate", -1)) - TRIT_RATE) > 1e-12:
            problems.append("record %d: rate %r != %r" % (i, r.get("rate"), TRIT_RATE))
        if r.get("prev") != prev:
            problems.append("record %d: prev link broken" % i)
        want = close_hash(prev, r.get("pack", ""), r.get("text", ""))
        if r.get("close") != want:
            problems.append("record %d: close mismatch" % i)
        prev = r.get("close", "")
    return problems


def _selftest():
    chain = build([{"pack": "trit", "text": "slice-a"}, {"pack": "trit", "text": "slice-b"}])
    assert gate(chain) == [], gate(chain)
    bad = [dict(chain[0], rate=0.1), chain[1]]
    assert any("rate" in p for p in gate(bad))
    tampered = [chain[0], dict(chain[1], text="x")]
    assert any("close" in p for p in gate(tampered))
    return chain[-1]["close"]


def main(argv):
    if len(argv) > 1:
        with open(argv[1], encoding="utf-8") as fh:
            chain = [json.loads(l) for l in fh if l.strip()]
        problems = gate(chain)
        for p in problems:
            print(p)
        print("PASS" if not problems else "FAIL")
        return 0 if not problems else 1
    print("selftest ok, tail close=%s rate=%s" % (_selftest(), TRIT_RATE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
