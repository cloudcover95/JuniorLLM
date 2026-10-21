#!/usr/bin/env python3
"""rails/ledger/close_ledger.py - append-only JSONL close ledger for JuniorLLM rails.

Offline, stdlib only, no network, no model pull, binds nothing.
Each entry chains: close = sha256(prev || pack || text)[:16].
Trit rate: 0.05 (repo carries a trit pack, see docs/TRIT_*.md).

Usage:
  python3 close_ledger.py append LEDGER.jsonl --pack trit --text "note"
  python3 close_ledger.py verify LEDGER.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
import time

TRIT_RATE = 0.05
GENESIS = "0" * 16


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def read_entries(path: str):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append(path: str, pack: str, text: str) -> dict:
    entries = read_entries(path)
    prev = entries[-1]["close"] if entries else GENESIS
    entry = {
        "seq": len(entries),
        "ts": int(time.time()),
        "prev": prev,
        "pack": pack,
        "text": text,
        "rate": TRIT_RATE,
        "close": close_hash(prev, pack, text),
    }
    with open(path, "a", encoding="utf-8") as fh:  # append-only, never rewrite
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def verify(path: str):
    prev = GENESIS
    for i, e in enumerate(read_entries(path)):
        if e.get("seq") != i or e.get("prev") != prev:
            return False, i
        if e.get("rate") != TRIT_RATE:
            return False, i
        if e.get("close") != close_hash(prev, e.get("pack", ""), e.get("text", "")):
            return False, i
        prev = e["close"]
    return True, -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("append")
    a.add_argument("ledger")
    a.add_argument("--pack", default="trit")
    a.add_argument("--text", required=True)
    v = sub.add_parser("verify")
    v.add_argument("ledger")
    args = ap.parse_args(argv)
    if args.cmd == "append":
        print(json.dumps(append(args.ledger, args.pack, args.text), sort_keys=True))
        return 0
    ok, bad = verify(args.ledger)
    print(json.dumps({"ok": ok, "first_bad": bad}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
