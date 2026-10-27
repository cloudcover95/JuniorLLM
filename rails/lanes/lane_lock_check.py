#!/usr/bin/env python3
"""Lane lock check for JuniorOS / JuniorLLM rails (JuniorCloud LLC).

Offline, stdlib only. Given a list of touched paths (repo-relative) and a
target repo slug, passes only if:
  - the repo is inside the LLC builder lane
    (cloudcover95/JuniorLLM, cloudcover95/JuniorPiThon, cloudcover95/JuniorPiPython),
  - the repo is never JuniorHome,
  - exactly one path is touched (one new file per run),
  - the path lives under rails/ or docs/,
  - no path is inside .github/.
Each verdict is chained with close = sha256(prev || pack || text)[:16]
using trit rate 0.05. No network, no sockets, no model pulls.

Usage:
  python3 lane_lock_check.py --repo cloudcover95/JuniorLLM rails/x/y.py
  python3 lane_lock_check.py --selftest
"""
import argparse
import hashlib
import json
import sys

TRIT_RATE = 0.05
LANE = ("cloudcover95/JuniorLLM", "cloudcover95/JuniorPiThon", "cloudcover95/JuniorPiPython")
FORBIDDEN_REPOS = ("cloudcover95/JuniorHome",)
ALLOWED_ROOTS = ("rails/", "docs/")
GENESIS = "0" * 16


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def check(repo, paths):
    problems = []
    if repo in FORBIDDEN_REPOS:
        problems.append("repo is JuniorHome (out of lane)")
    elif repo not in LANE:
        problems.append("repo not in lane: %s" % repo)
    if len(paths) != 1:
        problems.append("expected exactly one new file, got %d" % len(paths))
    for p in paths:
        norm = p.replace("\\", "/").lstrip("./")
        if ".." in norm.split("/"):
            problems.append("path escapes repo: %s" % p)
        if norm.startswith(".github/") or "/.github/" in "/" + norm:
            problems.append(".github path: %s" % p)
        if not norm.startswith(ALLOWED_ROOTS):
            problems.append("path outside rails/ or docs/: %s" % p)
    return problems


def verdict(repo, paths, prev=GENESIS):
    problems = check(repo, paths)
    pack = json.dumps({"trit_rate": TRIT_RATE, "repo": repo}, sort_keys=True)
    text = json.dumps({"paths": sorted(paths), "problems": problems}, sort_keys=True)
    return {
        "ok": not problems,
        "problems": problems,
        "trit_rate": TRIT_RATE,
        "prev": prev,
        "close": close_hash(prev, pack, text),
    }


def selftest():
    good = verdict("cloudcover95/JuniorLLM", ["rails/lanes/lane_lock_check.py"])
    assert good["ok"], good
    assert len(good["close"]) == 16
    assert verdict("cloudcover95/JuniorHome", ["docs/x.md"])["ok"] is False
    assert verdict("someone/Else", ["rails/x.py"])["ok"] is False
    assert verdict("cloudcover95/JuniorLLM", [".github/workflows/a.yml"])["ok"] is False
    assert verdict("cloudcover95/JuniorLLM", ["rails/.github/a.yml"])["ok"] is False
    assert verdict("cloudcover95/JuniorLLM", ["src/a.py"])["ok"] is False
    assert verdict("cloudcover95/JuniorLLM", ["rails/../x.py"])["ok"] is False
    assert verdict("cloudcover95/JuniorLLM", ["rails/a.py", "rails/b.py"])["ok"] is False
    assert verdict("cloudcover95/JuniorPiThon", ["docs/a.md"])["ok"]
    # chain: next close depends on prev
    nxt = verdict("cloudcover95/JuniorLLM", ["docs/a.md"], prev=good["close"])
    alt = verdict("cloudcover95/JuniorLLM", ["docs/a.md"])
    assert nxt["close"] != alt["close"]
    assert TRIT_RATE == 0.05
    print("selftest ok close=%s" % good["close"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lane lock check (offline)")
    ap.add_argument("--repo", default="cloudcover95/JuniorLLM")
    ap.add_argument("--prev", default=GENESIS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    v = verdict(a.repo, a.paths, a.prev)
    print(json.dumps(v, indent=2, sort_keys=True))
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
