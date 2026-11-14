#!/usr/bin/env python3
"""Offline stdlib guard: the draft file must live in a scratch dir OUTSIDE the repo checkout.
Close hash = sha256(prev || pack || text)[:16]; trit rate 0.05. Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
No network, no model pull, no GitHub Actions."""
import hashlib, os, sys

TRIT_RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def outside(repo: str, draft: str) -> bool:
    r = os.path.realpath(repo)
    d = os.path.realpath(draft)
    return os.path.commonpath([r, d]) != r


def check(repo: str, draft: str, prev: str = "") -> dict:
    ok = outside(repo, draft)
    return {"ok": ok, "rate": TRIT_RATE, "ports": PORTS,
            "close": close_hash(prev, "trit", f"{repo}|{draft}|{ok}")}


def selftest() -> int:
    assert outside("/tmp/repo", "/tmp/scratch/x.py")
    assert not outside("/tmp/repo", "/tmp/repo/rails/x.py")
    assert len(close_hash("a", "b", "c")) == 16
    print("selftest ok")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    if len(sys.argv) >= 3:
        res = check(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
        print(res)
        sys.exit(0 if res["ok"] else 1)
    print("usage: scratch_first_check.py selftest | <repo> <draft> [prev]")
    sys.exit(2)
