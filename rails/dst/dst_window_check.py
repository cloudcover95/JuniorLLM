#!/usr/bin/env python3
"""rails/dst/dst_window_check.py - JuniorCloud LLC rail (offline, stdlib only).

Daylight-saving guard for the dual autonomy windows (America/Denver):
  - LLC bots write 02:00-04:45 local.
  - Personal quant team owns 05:05-07:05 local; LLC bots never write there.

On fall-back day (first Sunday of November) local 01:00-01:59 happens twice
and on spring-forward day 02:00-02:59 does not exist. This rail converts a UTC
instant to Denver wall time with zoneinfo, reports the UTC offset and fold,
and decides whether an LLC write is allowed. Ambiguous or nonexistent local
times are never treated as inside the LLC window by local-string parsing;
decisions are always made from a real UTC instant.

No network, no model pull, no services bound (nothing listens; if a caller
ever wraps this in a service it must bind 127.0.0.1 only).
Trit rate 0.05. Close hash = sha256(prev || pack || text)[:16].
Ports named for the close pack: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
import hashlib
import json
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

DENVER = ZoneInfo("America/Denver")
TRIT_RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
LLC_WINDOW = (time(2, 0), time(4, 45))
PERSONAL_WINDOW = (time(5, 5), time(7, 5))


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def _in(t: time, win) -> bool:
    return win[0] <= t <= win[1]


def is_ambiguous(local_naive: datetime) -> bool:
    a = local_naive.replace(tzinfo=DENVER, fold=0).utcoffset()
    b = local_naive.replace(tzinfo=DENVER, fold=1).utcoffset()
    return a != b and _roundtrips(local_naive, 0) and _roundtrips(local_naive, 1)


def is_nonexistent(local_naive: datetime) -> bool:
    return not _roundtrips(local_naive, 0) and not _roundtrips(local_naive, 1)


def _roundtrips(local_naive: datetime, fold: int) -> bool:
    aware = local_naive.replace(tzinfo=DENVER, fold=fold)
    back = aware.astimezone(timezone.utc).astimezone(DENVER)
    return back.replace(tzinfo=None) == local_naive


def decide(utc_instant: datetime) -> dict:
    if utc_instant.tzinfo is None:
        raise ValueError("utc_instant must be timezone-aware")
    local = utc_instant.astimezone(DENVER)
    t = local.time().replace(tzinfo=None)
    off = local.utcoffset()
    in_llc = _in(t, LLC_WINDOW)
    in_personal = _in(t, PERSONAL_WINDOW)
    return {
        "utc": utc_instant.astimezone(timezone.utc).isoformat(),
        "denver": local.isoformat(),
        "abbr": local.tzname(),
        "offset_hours": off.total_seconds() / 3600,
        "fold": local.fold,
        "ambiguous_local": is_ambiguous(local.replace(tzinfo=None, fold=0)),
        "llc_window": in_llc,
        "personal_window": in_personal,
        "write_allowed": in_llc and not in_personal,
    }


def windows_disjoint() -> bool:
    return LLC_WINDOW[1] < PERSONAL_WINDOW[0]


def pack(decision: dict) -> str:
    body = dict(decision, trit_rate=TRIT_RATE, ports=list(PORTS))
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def selftest() -> int:
    u = timezone.utc
    assert windows_disjoint()
    # Fall-back day 2026-11-01: 08:30Z = 01:30 MDT (fold 0), 09:30Z = 01:30 MST (fold 1)
    d1 = decide(datetime(2026, 11, 1, 7, 30, tzinfo=u))
    d2 = decide(datetime(2026, 11, 1, 8, 30, tzinfo=u))
    assert d1["denver"].startswith("2026-11-01T01:30") and d1["abbr"] == "MDT"
    assert d2["denver"].startswith("2026-11-01T01:30") and d2["abbr"] == "MST"
    assert d1["ambiguous_local"] and d2["ambiguous_local"]
    assert not d1["write_allowed"] and not d2["write_allowed"]
    # 09:51Z on fall-back day = 02:51 MST -> inside LLC window
    d3 = decide(datetime(2026, 11, 1, 9, 51, tzinfo=u))
    assert d3["abbr"] == "MST" and d3["llc_window"] and d3["write_allowed"]
    assert not d3["ambiguous_local"]
    # Same UTC clock a day earlier (MDT) = 03:51 MDT -> inside; 11:10Z = 05:10 MDT -> personal
    assert decide(datetime(2026, 10, 31, 9, 51, tzinfo=u))["write_allowed"]
    d4 = decide(datetime(2026, 10, 31, 11, 10, tzinfo=u))
    assert d4["personal_window"] and not d4["write_allowed"]
    # 04:46 local is outside LLC window
    assert not decide(datetime(2026, 11, 2, 11, 46, tzinfo=u))["write_allowed"]
    # Spring-forward 2027-03-14: local 02:30 does not exist
    assert is_nonexistent(datetime(2027, 3, 14, 2, 30))
    assert not is_nonexistent(datetime(2027, 3, 14, 3, 30))
    # Naive input refused
    try:
        decide(datetime(2026, 11, 1, 2, 51))
        raise AssertionError("naive accepted")
    except ValueError:
        pass
    # Close-hash chain
    prev = "0" * 16
    chain = []
    for d in (d1, d2, d3):
        p = pack(d)
        prev = close_hash(prev, p, "dst_window_check")
        chain.append(prev)
    assert len(set(chain)) == 3 and all(len(c) == 16 for c in chain)
    assert close_hash("a", "b", "c") == hashlib.sha256(b"abc").hexdigest()[:16]
    assert TRIT_RATE == 0.05
    print("selftest ok", chain[-1])
    return 0


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "--selftest":
        return selftest()
    if len(argv) > 1:
        inst = datetime.fromisoformat(argv[1].replace("Z", "+00:00"))
    else:
        inst = datetime.now(timezone.utc)
    d = decide(inst)
    p = pack(d)
    print(json.dumps(dict(d, close=close_hash("0" * 16, p, "dst_window_check")), indent=2))
    return 0 if d["write_allowed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
