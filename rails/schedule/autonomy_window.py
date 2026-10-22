#!/usr/bin/env python3
"""JuniorOS rail: autonomy window gate (offline, stdlib only).

Answers "may an LLC bot write right now?" for the dual autonomy windows
(America/Denver):
  - LLC bots write      02:00-04:45
  - personal quant team 05:05-07:05 (LLC bots never write here)

No network, no sockets, no model pulls. Each decision is appended to a
close-hash chain: close = sha256(prev || pack || text)[:16], trit rate 0.05.

Usage:
  python3 autonomy_window.py                 # check now
  python3 autonomy_window.py --at 2026-10-22T03:10
  python3 autonomy_window.py --selftest
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Denver")
TRIT_RATE = 0.05
PACK = "juniorllm.rails.schedule.autonomy_window.v1"
GENESIS = "0" * 16

WINDOWS = {
    "llc": (time(2, 0), time(4, 45)),
    "personal": (time(5, 5), time(7, 5)),
}


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def window_for(ts: datetime) -> str:
    local = ts.astimezone(TZ).time()
    for name, (start, end) in WINDOWS.items():
        if start <= local <= end:
            return name
    return "none"


def decide(ts: datetime, prev: str = GENESIS) -> dict:
    local = ts.astimezone(TZ)
    win = window_for(local)
    allowed = win == "llc"
    text = json.dumps(
        {"at": local.isoformat(timespec="minutes"), "window": win,
         "llc_write": allowed, "trit_rate": TRIT_RATE},
        sort_keys=True,
    )
    return {"text": text, "llc_write": allowed, "window": win,
            "close": close_hash(prev, PACK, text), "prev": prev}


def parse_at(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=TZ)


def selftest() -> int:
    cases = [
        ("2026-10-22T02:00", "llc", True),
        ("2026-10-22T04:45", "llc", True),
        ("2026-10-22T04:46", "none", False),
        ("2026-10-22T05:05", "personal", False),
        ("2026-10-22T07:05", "personal", False),
        ("2026-10-22T01:59", "none", False),
        ("2026-10-22T12:00", "none", False),
    ]
    prev = GENESIS
    for at, win, ok in cases:
        r = decide(parse_at(at), prev)
        assert r["window"] == win, (at, r)
        assert r["llc_write"] is ok, (at, r)
        assert r["close"] == close_hash(prev, PACK, r["text"])
        assert len(r["close"]) == 16
        prev = r["close"]
    # chain must be order-sensitive
    a = decide(parse_at(cases[0][0]))["close"]
    b = decide(parse_at(cases[0][0]), prev=a)["close"]
    assert a != b
    print("selftest ok", prev)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--at", help="ISO time (Denver if no offset)")
    ap.add_argument("--prev", default=GENESIS, help="previous close hash")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ts = parse_at(a.at) if a.at else datetime.now(TZ)
    r = decide(ts, a.prev)
    print(json.dumps(r, sort_keys=True))
    return 0 if r["llc_write"] else 3


if __name__ == "__main__":
    sys.exit(main())
