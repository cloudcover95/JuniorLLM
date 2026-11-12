"""Offline stdlib guard: a planned new file path must not already exist in a checkout.

Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM. Binds nothing; no network.
Trit rate 0.05. Close hash = sha256(prev || pack || text)[:16].
"""
import hashlib
import os
import sys

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
RATE = 0.05


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(root: str, rel: str) -> bool:
    """True only if rel is a new, non-.github path under root."""
    if ".github" in rel.split("/"):
        return False
    return not os.path.lexists(os.path.join(root, rel))


def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.txt"), "w").close()
        assert not check(d, "a.txt")
        assert check(d, "b.txt")
        assert not check(d, ".github/x.yml")
    print("ok", close_hash("0" * 16, "trit-" + str(RATE), ",".join(PORTS)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3:
        sys.exit(0 if check(sys.argv[1], sys.argv[2]) else 1)
    sys.exit(selftest())
