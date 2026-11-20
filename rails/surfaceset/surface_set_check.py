#!/usr/bin/env python3
"""Surface set guard: JuniorOS surfaces must be exactly react-native, web, legacy; live must be false. Offline stdlib, no network."""
import hashlib, json, sys

SURFACES = ("react-native", "web", "legacy")
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
RATE = 0.05

def check(cfg):
    problems = []
    if tuple(cfg.get("surfaces", ())) != SURFACES:
        problems.append("surfaces must be %s" % (SURFACES,))
    if cfg.get("live") is not False:
        problems.append("live must be false")
    if cfg.get("metro_launched"):
        problems.append("metro must not be launched")
    if cfg.get("bind", "127.0.0.1") != "127.0.0.1":
        problems.append("bind must be 127.0.0.1")
    return problems

def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]

def selftest():
    good = {"surfaces": list(SURFACES), "live": False, "metro_launched": False, "bind": "127.0.0.1"}
    assert check(good) == []
    assert check(dict(good, live=True))
    assert check(dict(good, surfaces=["web"]))
    assert check(dict(good, bind="0.0.0.0"))
    assert len(close_hash("a", "b", "c")) == 16
    return close_hash("", "trit@%s" % RATE, json.dumps(good, sort_keys=True))

if __name__ == "__main__":
    print("selftest ok close=%s ports=%s" % (selftest(), ",".join(PORTS)))
    sys.exit(0)
