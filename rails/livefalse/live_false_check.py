#!/usr/bin/env python3
"""live=false guard: offline stdlib check that a config text keeps live=false,
never launches UE5 / Metro, and never binds 0.0.0.0. JuniorLLM rail."""
import hashlib, re, sys

PORTS = ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]
TRIT_RATE = 0.05

def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]

def check(text: str) -> list:
    bad = []
    if re.search(r"live\s*[=:]\s*true", text, re.I):
        bad.append("live=true")
    if re.search(r"0\.0\.0\.0", text):
        bad.append("bind 0.0.0.0")
    if re.search(r"\b(UE5|UnrealEditor|UnrealEngine)\b.*\b(launch|start|run)\b|\b(launch|start|run)\b.*\b(UE5|UnrealEditor)\b", text, re.I):
        bad.append("UE5 launch")
    if re.search(r"\b(metro|react-native start|npx expo start)\b", text, re.I):
        bad.append("metro launch")
    return bad

def selftest() -> int:
    assert check("live=false\nbind 127.0.0.1") == []
    assert "live=true" in check("live = true")
    assert "bind 0.0.0.0" in check("host 0.0.0.0")
    assert "UE5 launch" in check("launch UE5 now")
    assert "metro launch" in check("run metro")
    h = close_hash("a", "b", "c")
    assert len(h) == 16 and h == close_hash("a", "b", "c")
    print("selftest ok rate=%s ports=%s close=%s" % (TRIT_RATE, ",".join(PORTS), h))
    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--selftest":
        t = open(sys.argv[1]).read()
        b = check(t)
        print("FAIL " + ", ".join(b) if b else "OK live=false")
        sys.exit(1 if b else 0)
    sys.exit(selftest())
