#!/usr/bin/env python3
"""Offline stdlib guard: LLC bots may write only 02:00-04:45 America/Denver.
Local only, no network, no remote scripts. Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Usage: write_window_check.py [HH:MM]   (defaults to current Denver time)
"""
import hashlib, sys
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
TRIT_RATE = 0.05
START, END = 2 * 60, 4 * 60 + 45


def minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def in_window(hhmm):
    return START <= minutes(hhmm) <= END


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def selftest():
    assert in_window("02:00") and in_window("04:45") and in_window("03:30")
    assert not in_window("01:59") and not in_window("04:46") and not in_window("05:05")
    assert close_hash("a", "b", "c") == close_hash("a", "b", "c")
    assert len(close_hash("", "", "")) == 16
    return "selftest ok"


def main(argv):
    print(selftest())
    if len(argv) > 1:
        now = argv[1]
    elif ZoneInfo:
        now = datetime.now(ZoneInfo("America/Denver")).strftime("%H:%M")
    else:
        print("no tz data; pass HH:MM"); return 2
    ok = in_window(now)
    print(f"{now} America/Denver write_allowed={ok} rate={TRIT_RATE} ports={','.join(PORTS)}")
    print("close", close_hash("write_window", "trit0.05", now))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
