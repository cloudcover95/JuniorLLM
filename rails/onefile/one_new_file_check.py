#!/usr/bin/env python3
"""JuniorLLM rail: one-new-file guard (JuniorCloud LLC).

Checks that a staged builder commit is exactly one *added* file with no
modifications, deletions, renames, or copies. Input is the output of
`git diff --cached --name-status` (or a file/stdin with the same format).
Offline, stdlib only. Never runs git itself unless --staged is passed,
and even then only the read-only `git diff --cached --name-status`.

Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Trit pack rate: 0.05.
Close rule: close = sha256(prev || pack || text)[:16].
"""
import hashlib
import json
import subprocess
import sys

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
TRIT_RATE = 0.05
GENESIS = "0" * 16


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def trit(ok, warn=False):
    """+1 pass, 0 warn, -1 fail."""
    if not ok:
        return -1
    return 0 if warn else 1


def parse_name_status(text):
    rows = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0].strip()
        paths = [p for p in parts[1:] if p]
        rows.append((status, paths))
    return rows


def check(text, prev=GENESIS):
    rows = parse_name_status(text)
    problems = []
    added = [r for r in rows if r[0] == "A"]
    for status, paths in rows:
        code = status[:1]
        if code != "A":
            problems.append("non-additive change %s: %s" % (status, " -> ".join(paths)))
        for p in paths:
            segs = p.replace("\\", "/").split("/")
            if ".github" in segs:
                problems.append("forbidden .github path: %s" % p)
    if len(added) != 1:
        problems.append("expected exactly 1 added file, found %d" % len(added))
    ok = not problems
    pack = json.dumps({"rate": TRIT_RATE, "trit": trit(ok), "n": len(rows)}, sort_keys=True)
    return {
        "ok": ok,
        "added": [r[1][0] for r in added if r[1]],
        "problems": problems,
        "ports": list(PORTS),
        "pack": pack,
        "close": close_hash(prev, pack, text),
    }


def selftest():
    good = "A\trails/onefile/one_new_file_check.py\n"
    r = check(good)
    assert r["ok"] and r["added"] == ["rails/onefile/one_new_file_check.py"], r
    assert len(r["close"]) == 16
    assert r["close"] == close_hash(GENESIS, r["pack"], good)
    assert check(good, prev=r["close"])["close"] != r["close"]
    assert not check("")["ok"]
    assert not check("A\ta.py\nA\tb.py\n")["ok"]
    assert not check("A\ta.py\nM\tREADME.md\n")["ok"]
    assert not check("D\told.py\n")["ok"]
    assert not check("R100\told.py\tnew.py\n")["ok"]
    assert not check("A\t.github/workflows/ci.yml\n")["ok"]
    assert check("A\tdocs/github_notes.md\n")["ok"]
    assert TRIT_RATE == 0.05 and PORTS[-1] == "JuniorLLM"
    print("selftest ok")
    return 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if "--staged" in argv:
        text = subprocess.run(["git", "diff", "--cached", "--name-status"],
                              capture_output=True, text=True, check=True).stdout
    else:
        args = [a for a in argv if not a.startswith("--")]
        text = open(args[0], encoding="utf-8").read() if args else sys.stdin.read()
    r = check(text)
    print(json.dumps(r, indent=2))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
