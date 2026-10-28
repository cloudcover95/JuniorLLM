#!/usr/bin/env python3
"""JuniorOS report-slot SHA check (offline, stdlib only).

JuniorCloud LLC rail. A report slot only checks and reports file SHAs of the
JuniorOS core files under ports/ and commits nothing, unless a core file is
missing, in which case at most one additive file may be pushed.

Core files: ports/os_tick.py, ports/os_mesh.py, ports/surface_spin.py
Surfaces: react-native, web, legacy (Metro not launched, live=false).
Close rule: close = sha256(prev || pack || text)[:16]; trit rate 0.05.
No network, no model pull, no .github, never binds anything (127.0.0.1 only).
"""
import hashlib
import json
import os
import sys

CORE_FILES = ("ports/os_tick.py", "ports/os_mesh.py", "ports/surface_spin.py")
SURFACES = ("react-native", "web", "legacy")
TRIT_RATE = 0.05
LIVE = False
METRO_LAUNCHED = False
BIND_HOST = "127.0.0.1"


def close_hash(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def report(root: str, prev: str = "0" * 16, pack: str = "trit") -> dict:
    entries, missing = [], []
    for rel in CORE_FILES:
        full = os.path.join(root, rel)
        if os.path.isfile(full):
            entries.append({"path": rel, "sha256": file_sha(full)})
        else:
            missing.append(rel)
            entries.append({"path": rel, "sha256": None})
    text = json.dumps(entries, sort_keys=True)
    return {
        "entries": entries,
        "missing": missing,
        "action": "push_one_additive_file" if missing else "report_only",
        "max_new_files": 1 if missing else 0,
        "surfaces": list(SURFACES),
        "live": LIVE,
        "metro_launched": METRO_LAUNCHED,
        "bind": BIND_HOST,
        "trit_rate": TRIT_RATE,
        "close": close_hash(prev, pack, text),
    }


def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "ports"))
        for rel in CORE_FILES:
            with open(os.path.join(tmp, rel), "w") as fh:
                fh.write("# " + rel + "\n")
        r = report(tmp)
        assert r["missing"] == [] and r["action"] == "report_only"
        assert r["max_new_files"] == 0
        assert all(e["sha256"] and len(e["sha256"]) == 64 for e in r["entries"])
        assert len(r["close"]) == 16
        assert r["close"] == report(tmp)["close"], "close must be deterministic"
        os.remove(os.path.join(tmp, CORE_FILES[1]))
        r2 = report(tmp)
        assert r2["missing"] == [CORE_FILES[1]]
        assert r2["action"] == "push_one_additive_file" and r2["max_new_files"] == 1
        assert r2["close"] != r["close"]
    assert close_hash("a", "b", "c") == hashlib.sha256(b"abc").hexdigest()[:16]
    assert TRIT_RATE == 0.05 and LIVE is False and METRO_LAUNCHED is False
    assert BIND_HOST == "127.0.0.1"
    print("selftest ok")
    return 0


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "--selftest":
        return selftest()
    root = argv[1] if len(argv) > 1 else "."
    r = report(root)
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
