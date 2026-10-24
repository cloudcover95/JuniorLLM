#!/usr/bin/env python3
"""JuniorOS surface matrix rail (offline, stdlib only).

Records the state of the JuniorOS surfaces (react-native, web, legacy)
without launching anything: Metro stays down, live=false, no UE5,
no network. Each row is chained with the close rule:
    close = sha256(prev || pack || text)[:16]
Trit pack rate: 0.05. Any loopback reference is 127.0.0.1 only.
"""
import hashlib
import json
import sys

TRIT_RATE = 0.05
BIND_HOST = "127.0.0.1"
PACK = "trit-pack:rate=%.2f" % TRIT_RATE
GENESIS = "0" * 16

SURFACES = (
    {"surface": "react-native", "core": "ports/surface_spin.py", "metro_launched": False, "live": False},
    {"surface": "web", "core": "ports/surface_spin.py", "metro_launched": False, "live": False},
    {"surface": "legacy", "core": "ports/surface_spin.py", "metro_launched": False, "live": False},
)


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def build(surfaces=SURFACES, prev=GENESIS):
    rows = []
    for s in surfaces:
        if s.get("live") or s.get("metro_launched"):
            raise ValueError("surface %s must stay offline" % s["surface"])
        text = json.dumps(dict(s, bind=BIND_HOST), sort_keys=True)
        prev = close_hash(prev, PACK, text)
        rows.append({"text": text, "close": prev})
    return rows


def verify(rows, prev=GENESIS):
    for r in rows:
        prev2 = close_hash(prev, PACK, r["text"])
        if prev2 != r["close"]:
            return False
        prev = prev2
    return True


def _self_test():
    rows = build()
    assert len(rows) == 3 and verify(rows)
    tampered = [dict(rows[0], text=rows[0]["text"].replace("false", "true"))] + rows[1:]
    assert not verify(tampered)
    try:
        build([{"surface": "web", "live": True}])
        raise AssertionError("live surface accepted")
    except ValueError:
        pass
    assert all(json.loads(r["text"])["bind"] == "127.0.0.1" for r in rows)
    return rows


if __name__ == "__main__":
    rows = _self_test()
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(r["close"], json.loads(r["text"])["surface"])
        print("ok trit_rate=%.2f bind=%s" % (TRIT_RATE, BIND_HOST))
