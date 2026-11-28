#!/usr/bin/env python3
"""no-launch guard: offline stdlib check that a surface config keeps live=false,
never launches Metro/UE5 and never binds 0.0.0.0. Trit rate 0.05."""
import hashlib, re, sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
BAD = (r"live\s*[=:]\s*true", r"0\.0\.0\.0", r"\bmetro\b.*\b(start|launch)\b", r"UnrealEditor|UE5.*launch")


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(text: str):
    hits = [p for p in BAD if re.search(p, text, re.I)]
    return (not hits), hits


def selftest() -> int:
    ok1, _ = check("live=false\nbind 127.0.0.1")
    ok2, _ = check("live = true")
    ok3, _ = check("host 0.0.0.0")
    assert ok1 and not ok2 and not ok3
    assert len(close_hash("a", "b", "c")) == 16
    print("selftest ok rate=%s ports=%s close=%s" % (RATE, ",".join(PORTS), close_hash("0", "pack", "live_false")))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "selftest":
        ok, hits = check(open(sys.argv[1]).read())
        print("ok" if ok else "FAIL %s" % hits)
        sys.exit(0 if ok else 1)
    sys.exit(selftest())
