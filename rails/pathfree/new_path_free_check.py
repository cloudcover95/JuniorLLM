#!/usr/bin/env python3
"""Offline stdlib guard: a new-file target path must not already exist (never overwrite).
Trit pack rate 0.05. close = sha256(prev || pack || text)[:16].
Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM. No network, no binds."""
import hashlib, os, sys, tempfile

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
RATE = 0.05


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check_target(repo_root: str, rel_path: str) -> int:
    """Return trit: +1 path is free and safe, -1 exists or unsafe."""
    parts = rel_path.replace("\\", "/").split("/")
    if rel_path.startswith("/") or ".." in parts or ".github" in parts:
        return -1
    return -1 if os.path.exists(os.path.join(repo_root, rel_path)) else 1


def selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.txt"), "w").close()
        assert check_target(d, "a.txt") == -1
        assert check_target(d, "b.txt") == 1
        assert check_target(d, ".github/x.yml") == -1
        assert check_target(d, "../x") == -1
    h = close_hash("0" * 16, "pack@%s" % RATE, "ok")
    assert len(h) == 16 and h == close_hash("0" * 16, "pack@%s" % RATE, "ok")
    print("selftest ok rate=%s ports=%s close=%s" % (RATE, ",".join(PORTS), h))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        t = check_target(sys.argv[1], sys.argv[2])
        print(t)
        sys.exit(0 if t == 1 else 1)
    selftest()
