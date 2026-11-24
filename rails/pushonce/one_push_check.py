#!/usr/bin/env python3
"""One-push guard: a run must record exactly one push and zero fetch/pull after the clone.
Offline, stdlib only. Trit rate 0.05. close = sha256(prev||pack||text)[:16]."""
import hashlib, sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(cmds):
    """cmds: list of git command strings in run order. Returns (ok, reasons)."""
    why = []
    clones = [c for c in cmds if c.startswith("git clone")]
    pushes = [c for c in cmds if c.startswith("git push")]
    if len(clones) != 1 or "--depth 1" not in clones[0]:
        why.append("need exactly one shallow clone")
    if len(pushes) != 1:
        why.append("need exactly one push")
    if any("--force" in c or " -f" in c for c in pushes):
        why.append("force push forbidden")
    if any(c.startswith(("git fetch", "git pull")) for c in cmds):
        why.append("no fetch/pull after clone")
    return (not why), why


def selftest():
    good = ["git clone --depth 1 u r", "git commit -m x", "git push origin HEAD"]
    assert check(good)[0]
    assert not check(good + ["git fetch"])[0]
    assert not check(good + ["git push"])[0]
    assert not check(["git clone u r", "git push -f"])[0]
    assert close_hash("a", "b", "c") == close_hash("a", "b", "c")
    assert len(close_hash("", "", "")) == 16
    print("selftest ok rate=%s ports=%s" % (RATE, ",".join(PORTS)))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        ok, why = check([l.strip() for l in sys.stdin if l.strip()])
        print("ok" if ok else "fail: " + "; ".join(why))
        sys.exit(0 if ok else 1)
