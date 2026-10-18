#!/usr/bin/env python3
"""Offline verifier for JuniorOS / JuniorLLM close-hash chains.

Rails slice 2026-10-18 (JuniorCloud LLC). Additive, stdlib only.
No network, no sockets, no model pulls: this file only reads a local
JSONL chain and recomputes each close hash.

Close rule: close = sha256(prev || pack || text)[:16]
  prev : previous close hash (hex string, "" for genesis)
  pack : trit pack string for the step (e.g. "+0-0+")
  text : step payload text

Trit pack rate: 0.05 (the repo carries a trit pack, so the guard rate applies).

Record format, one JSON object per line:
  {"prev": "...", "pack": "...", "text": "...", "close": "..."}

Usage:
  python3 rails/audit/close_chain_verify.py chain.jsonl
  python3 rails/audit/close_chain_verify.py --selftest
"""
from __future__ import annotations

import hashlib
import json
import sys
from typing import Iterable, List, Tuple

TRIT_RATE = 0.05
CLOSE_LEN = 16
TRITS = set("+0-")


def close_hash(prev: str, pack: str, text: str) -> str:
    """Return sha256(prev || pack || text)[:16] as hex."""
    data = prev.encode("utf-8") + pack.encode("utf-8") + text.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:CLOSE_LEN]


def valid_pack(pack: str) -> bool:
    return all(ch in TRITS for ch in pack)


def build_chain(steps: Iterable[Tuple[str, str]]) -> List[dict]:
    """Build chain records from (pack, text) steps, genesis prev = ""."""
    out: List[dict] = []
    prev = ""
    for pack, text in steps:
        c = close_hash(prev, pack, text)
        out.append({"prev": prev, "pack": pack, "text": text, "close": c})
        prev = c
    return out


def verify_chain(records: List[dict]) -> Tuple[bool, List[str]]:
    """Verify linkage and close hashes. Returns (ok, errors)."""
    errors: List[str] = []
    expected_prev = ""
    for i, rec in enumerate(records):
        prev = rec.get("prev", "")
        pack = rec.get("pack", "")
        text = rec.get("text", "")
        close = rec.get("close", "")
        if prev != expected_prev:
            errors.append(f"step {i}: prev {prev!r} != expected {expected_prev!r}")
        if not valid_pack(pack):
            errors.append(f"step {i}: pack has non-trit symbols")
        want = close_hash(prev, pack, text)
        if close != want:
            errors.append(f"step {i}: close {close!r} != recomputed {want!r}")
        expected_prev = close
    return (not errors, errors)


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _selftest() -> int:
    chain = build_chain([("+0-", "tick 1"), ("0+0", "tick 2"), ("--+", "tick 3")])
    ok, errs = verify_chain(chain)
    assert ok, errs
    assert len(chain[0]["close"]) == CLOSE_LEN
    assert chain[0]["close"] == hashlib.sha256(b"+0-tick 1").hexdigest()[:16]
    tampered = [dict(r) for r in chain]
    tampered[1]["text"] = "tick 2 edited"
    ok2, errs2 = verify_chain(tampered)
    assert not ok2 and any("step 1" in e for e in errs2)
    assert any("step 1" in e for e in verify_chain([chain[0], chain[2]])[1])
    assert TRIT_RATE == 0.05
    print(json.dumps({"selftest": "ok", "trit_rate": TRIT_RATE,
                      "tip": chain[-1]["close"]}))
    return 0


def main(argv: List[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[1] == "--selftest":
        return _selftest()
    records = load_jsonl(argv[1])
    ok, errs = verify_chain(records)
    tip = records[-1]["close"] if records else ""
    print(json.dumps({"ok": ok, "steps": len(records), "tip": tip,
                      "trit_rate": TRIT_RATE, "errors": errs}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
