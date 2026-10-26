#!/usr/bin/env python3
"""rails/additive/additive_guard.py - offline additive-only diff guard.

Reads `git diff --name-status` style lines (stdin or file) and passes only if
every entry is an addition (A) and no path touches .github/. Emits a receipt
chained with the close rule: close = sha256(prev || pack || text)[:16].
Trit pack rate 0.05. Stdlib only, no network, no model pull, no services.
JuniorCloud LLC.
"""
import hashlib
import json
import sys

TRIT_RATE = 0.05
PACK = "trit"
GENESIS = "0" * 16


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def check(lines):
    problems = []
    added = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        status, paths = parts[0], parts[1:]
        if not paths:
            problems.append("malformed: " + line)
            continue
        for p in paths:
            if p == ".github" or p.startswith(".github/") or "/.github/" in p:
                problems.append("github path: " + p)
        if status != "A":
            problems.append("non-additive %s: %s" % (status, " ".join(paths)))
        else:
            added.append(paths[0])
    return added, problems


def receipt(lines, prev=GENESIS):
    added, problems = check(lines)
    text = json.dumps({"added": added, "problems": problems}, sort_keys=True)
    return {
        "ok": not problems,
        "added": added,
        "problems": problems,
        "rate": TRIT_RATE,
        "prev": prev,
        "close": close_hash(prev, PACK, text),
    }


def selftest():
    good = receipt(["A\trails/additive/additive_guard.py"])
    assert good["ok"] and good["added"] == ["rails/additive/additive_guard.py"]
    assert len(good["close"]) == 16
    assert good["close"] == close_hash(GENESIS, PACK, json.dumps(
        {"added": good["added"], "problems": []}, sort_keys=True))
    assert not receipt(["D\tports/os_tick.py"])["ok"]
    assert not receipt(["M\trails/x.py"])["ok"]
    assert not receipt(["A\t.github/workflows/ci.yml"])["ok"]
    assert not receipt(["R100\ta.py\tb.py"])["ok"]
    chained = receipt(["A\tdocs/x.md"], prev=good["close"])
    assert chained["prev"] == good["close"] and chained["close"] != good["close"]
    assert TRIT_RATE == 0.05
    return True


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("selftest ok" if selftest() else "selftest failed")
        sys.exit(0)
    src = sys.stdin
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        src = open(args[0], encoding="utf-8")
    r = receipt(src.read().splitlines())
    print(json.dumps(r, sort_keys=True))
    sys.exit(0 if r["ok"] else 1)
