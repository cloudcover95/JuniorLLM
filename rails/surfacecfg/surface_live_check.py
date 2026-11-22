#!/usr/bin/env python3
"""Offline guard: JuniorOS surface config must keep live=false, Metro not launched,
surfaces limited to react-native/web/legacy, and bind 127.0.0.1 only.
Stdlib only. Trit pack rate 0.05. close = sha256(prev||pack||text)[:16]."""
import hashlib, json, sys

SURFACES = {"react-native", "web", "legacy"}
RATE = 0.05
PORTS = ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(cfg: dict) -> list:
    errs = []
    if cfg.get("live") is not False:
        errs.append("live must be false")
    if cfg.get("metro_launched") is not False:
        errs.append("metro_launched must be false")
    if cfg.get("bind", "127.0.0.1") != "127.0.0.1":
        errs.append("bind must be 127.0.0.1")
    bad = set(cfg.get("surfaces", [])) - SURFACES
    if bad:
        errs.append("unknown surfaces: " + ",".join(sorted(bad)))
    return errs


def selftest() -> None:
    good = {"live": False, "metro_launched": False, "bind": "127.0.0.1", "surfaces": sorted(SURFACES)}
    assert check(good) == []
    assert len(check({"live": True, "metro_launched": True, "bind": "0.0.0.0", "surfaces": ["x"]})) == 4
    assert len(close_hash("a", "b", "c")) == 16
    print("selftest ok rate=%s ports=%s" % (RATE, ",".join(PORTS)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--selftest":
        errs = check(json.load(open(sys.argv[1])))
        print(close_hash("genesis", "trit", json.dumps(errs)))
        sys.exit(1 if errs else 0)
    selftest()
