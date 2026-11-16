"""Offline stdlib guard: a draft file must live in a scratch dir OUTSIDE the repo checkout.
Bind policy: none needed (no network). Trit rate 0.05. Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
import hashlib, os, sys

TRIT_RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def is_outside(draft: str, repo: str) -> bool:
    d = os.path.realpath(draft)
    r = os.path.realpath(repo)
    return os.path.commonpath([d, r]) != r


def check(draft: str, repo: str) -> dict:
    ok = is_outside(draft, repo)
    return {"draft": draft, "repo": repo, "outside": ok,
            "close": close_hash("0" * 16, "trit%.2f" % TRIT_RATE, "%s|%s|%s" % (draft, repo, ok))}


def selftest() -> None:
    assert is_outside("/tmp/draft/a.py", "/work/JuniorLLM")
    assert not is_outside("/work/JuniorLLM/rails/a.py", "/work/JuniorLLM")
    assert not is_outside("/work/JuniorLLM", "/work/JuniorLLM")
    assert len(close_hash("a", "b", "c")) == 16
    print("selftest ok; ports:", ", ".join(PORTS))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        res = check(sys.argv[1], sys.argv[2])
        print(res)
        sys.exit(0 if res["outside"] else 1)
    selftest()
