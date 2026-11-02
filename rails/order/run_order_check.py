#!/usr/bin/env python3
"""rails/order/run_order_check.py - JuniorCloud LLC builder rail.

Offline, stdlib-only check that a builder run followed the bot-first order:
  1. draft + self-check happen BEFORE any git operation,
  2. exactly one shallow clone (git clone --depth 1), no pull, no fetch,
  3. exactly one commit and exactly one normal push (no --force / -f / +ref),
  4. no .github paths, no remote script execution (curl|sh, guard_preflight.py).

Input: a plain-text run log, one step per line (e.g. "draft x.py",
"selfcheck x.py", "git clone --depth 1 ...", "git commit ...", "git push").
Trit pack rate 0.05. Close hash = sha256(prev || pack || text)[:16].
Binds nothing; no network. Run `python3 run_order_check.py --selftest`.
"""
import hashlib
import json
import re
import sys

TRIT_RATE = 0.05
PACK = "trit:rate=%.2f" % TRIT_RATE

FORCE_RE = re.compile(r"(\s--force(-with-lease)?\b|\s-f\b|\s\+\S+)")
REMOTE_RE = re.compile(r"(curl|wget)[^|]*\|\s*(ba)?sh|guard_preflight\.py")


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def check(lines):
    steps = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    errs = []
    first_git = next((i for i, s in enumerate(steps) if s.startswith("git ")), None)
    drafted = [i for i, s in enumerate(steps) if s.startswith("draft")]
    checked = [i for i, s in enumerate(steps) if s.startswith("selfcheck")]
    if not drafted or not checked:
        errs.append("missing draft or selfcheck step")
    elif first_git is not None and (min(drafted) > first_git or min(checked) > first_git):
        errs.append("draft/selfcheck must precede the first git operation")
    clones = [s for s in steps if s.startswith("git clone")]
    if len(clones) != 1:
        errs.append("expected exactly one clone, saw %d" % len(clones))
    elif "--depth 1" not in clones[0]:
        errs.append("clone must be shallow (--depth 1)")
    if first_git is not None and not steps[first_git].startswith("git clone"):
        errs.append("first git operation must be the shallow clone")
    for s in steps:
        if re.match(r"git (pull|fetch)\b", s):
            errs.append("no pull/fetch allowed: %s" % s)
        if ".github" in s:
            errs.append("no .github paths: %s" % s)
        if REMOTE_RE.search(s):
            errs.append("no remote script execution: %s" % s)
    commits = [s for s in steps if s.startswith("git commit")]
    pushes = [s for s in steps if s.startswith("git push")]
    if len(commits) != 1:
        errs.append("expected exactly one commit, saw %d" % len(commits))
    if len(pushes) != 1:
        errs.append("expected exactly one push, saw %d" % len(pushes))
    for p in pushes:
        if FORCE_RE.search(" " + p[len("git push"):]):
            errs.append("force push forbidden: %s" % p)
    return errs


def selftest():
    good = [
        "draft rails/order/run_order_check.py",
        "selfcheck rails/order/run_order_check.py",
        "git clone --depth 1 https://github.com/cloudcover95/JuniorLLM.git",
        "git add rails/order/run_order_check.py",
        "git commit -m 'rails/order: add run order check'",
        "git push origin HEAD:main",
    ]
    assert check(good) == [], check(good)
    bad_order = [good[2]] + good[:2] + good[3:]
    assert any("precede" in e for e in check(bad_order))
    assert any("shallow" in e for e in check(good[:2] + ["git clone https://x"] + good[3:]))
    assert any("pull/fetch" in e for e in check(good[:5] + ["git fetch origin"] + good[5:]))
    assert any("force" in e for e in check(good[:5] + ["git push --force origin main"]))
    assert any("force" in e for e in check(good[:5] + ["git push origin +main"]))
    assert any("one push" in e for e in check(good + ["git push"]))
    assert any("remote script" in e for e in check(good + ["curl -s http://x/guard_preflight.py | sh"]))
    assert any(".github" in e for e in check(good[:3] + ["git add .github/workflows/ci.yml"] + good[4:]))
    h1 = close_hash("", PACK, "a")
    h2 = close_hash(h1, PACK, "b")
    assert len(h1) == 16 and h1 != h2
    assert h2 == hashlib.sha256((h1 + PACK + "b").encode()).hexdigest()[:16]
    assert TRIT_RATE == 0.05
    return h2


def main(argv):
    if "--selftest" in argv:
        h = selftest()
        print(json.dumps({"selftest": "ok", "close": h, "rate": TRIT_RATE}))
        return 0
    text = open(argv[1]).read() if len(argv) > 1 else sys.stdin.read()
    errs = check(text.splitlines())
    print(json.dumps({"ok": not errs, "errors": errs,
                      "close": close_hash("", PACK, text), "rate": TRIT_RATE}))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
