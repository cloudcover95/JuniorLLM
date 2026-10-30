#!/usr/bin/env python3
"""rails/metro/metro_guard.py - offline stdlib guard for JuniorOS surfaces.

Checks a surface config (JSON) so that:
  * every surface (react-native, web, legacy) has live == false
  * Metro is not launched (metro.launched == false)
  * no command launches UE5 (UnrealEditor / UE5 / UE4Editor)
  * any host binding is 127.0.0.1 or localhost only (never 0.0.0.0 / ::)
Each result is chained with close = sha256(prev || pack || text)[:16].
Trit pack rate 0.05. No network, no subprocess, nothing launched.

Usage: python3 metro_guard.py <config.json>   |   python3 metro_guard.py --selftest
"""
import hashlib
import json
import sys

TRIT_RATE = 0.05
PACK = "trit:0.05"
SURFACES = ("react-native", "web", "legacy")
UE5_MARKERS = ("unrealeditor", "ue5", "ue4editor", "unrealengine")
SAFE_HOSTS = ("127.0.0.1", "localhost")


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def check(cfg):
    problems = []
    surfaces = cfg.get("surfaces", {})
    for name in SURFACES:
        s = surfaces.get(name)
        if s is None:
            problems.append("missing surface %s" % name)
            continue
        if s.get("live", True) is not False:
            problems.append("surface %s live is not false" % name)
        host = s.get("host", "127.0.0.1")
        if host not in SAFE_HOSTS:
            problems.append("surface %s binds %s" % (name, host))
    if cfg.get("metro", {}).get("launched", True) is not False:
        problems.append("metro launched")
    for cmd in cfg.get("commands", []):
        low = str(cmd).lower()
        if any(m in low for m in UE5_MARKERS):
            problems.append("ue5 launch command: %s" % cmd)
        if "0.0.0.0" in low or "[::]" in low:
            problems.append("wildcard bind in command: %s" % cmd)
    return problems


def run(cfg, prev="0" * 16):
    problems = check(cfg)
    text = "ok" if not problems else "; ".join(problems)
    return {"ok": not problems, "problems": problems, "trit_rate": TRIT_RATE,
            "close": close_hash(prev, PACK, text)}


def selftest():
    good = {"surfaces": {n: {"live": False, "host": "127.0.0.1"} for n in SURFACES},
            "metro": {"launched": False},
            "commands": ["python3 ports/os_tick.py --bind 127.0.0.1"]}
    r = run(good)
    assert r["ok"], r
    assert r["close"] == close_hash("0" * 16, PACK, "ok")
    assert len(r["close"]) == 16
    bad = json.loads(json.dumps(good))
    bad["surfaces"]["web"]["live"] = True
    bad["surfaces"]["legacy"]["host"] = "0.0.0.0"
    bad["metro"]["launched"] = True
    bad["commands"].append("UnrealEditor.exe Junior.uproject")
    bad["commands"].append("serve --host 0.0.0.0")
    rb = run(bad, prev=r["close"])
    assert not rb["ok"] and len(rb["problems"]) == 5, rb
    missing = {"surfaces": {}, "metro": {"launched": False}}
    assert len(check(missing)) == 3
    assert TRIT_RATE == 0.05
    print("selftest ok close=%s" % rb["close"])
    return 0


def main(argv):
    if len(argv) > 1 and argv[1] == "--selftest":
        return selftest()
    if len(argv) != 2:
        print(__doc__)
        return 2
    with open(argv[1], "r", encoding="utf-8") as fh:
        res = run(json.load(fh))
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
