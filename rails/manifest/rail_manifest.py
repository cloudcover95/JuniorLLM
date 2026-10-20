#!/usr/bin/env python3
"""rails/manifest/rail_manifest.py -- offline SHA manifest for JuniorLLM rails.

JuniorCloud LLC. Additive rail, stdlib only, no network, no model pull,
never binds a socket. Walks a directory (default: rails/ next to this file's
parent), records sha256 of every regular file, and chains each entry with the
close rule:

    close = sha256(prev || pack || text)[:16]

where prev is the previous close (empty for the first entry), pack is the
trit-pack tag (rate 0.05), and text is "<relpath>:<sha256>".

Skips .git and .github directories entirely (this rail never reads or writes
CI config). Usage:

    python3 rails/manifest/rail_manifest.py [ROOT] [--json]
    python3 rails/manifest/rail_manifest.py --self-check
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

TRIT_RATE = 0.05
PACK = "trit:rate=%.2f" % TRIT_RATE
SKIP_DIRS = {".git", ".github", "__pycache__"}


def close_hash(prev: str, pack: str, text: str) -> str:
    """Close rule: sha256(prev || pack || text)[:16]."""
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(root: str):
    """Yield (relpath, sha256) in a stable sorted order."""
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full) or os.path.islink(full):
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            yield rel, file_sha256(full)


def build_manifest(root: str) -> dict:
    prev = ""
    entries = []
    for rel, sha in walk(root):
        text = "%s:%s" % (rel, sha)
        prev = close_hash(prev, PACK, text)
        entries.append({"path": rel, "sha256": sha, "close": prev})
    return {
        "root": os.path.basename(os.path.abspath(root)) or root,
        "pack": PACK,
        "rate": TRIT_RATE,
        "count": len(entries),
        "entries": entries,
        "final_close": prev,
    }


def verify_manifest(manifest: dict) -> bool:
    prev = ""
    for e in manifest.get("entries", []):
        prev = close_hash(prev, manifest["pack"], "%s:%s" % (e["path"], e["sha256"]))
        if prev != e["close"]:
            return False
    return prev == manifest.get("final_close", "")


def self_check() -> int:
    assert close_hash("", PACK, "a:b") == hashlib.sha256(
        (PACK + "a:b").encode()).hexdigest()[:16]
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "sub"))
        os.makedirs(os.path.join(tmp, ".github"))
        for rel, body in (("a.txt", b"alpha"), ("sub/b.txt", b"beta"),
                          (".github/skip.yml", b"never")):
            with open(os.path.join(tmp, rel), "wb") as fh:
                fh.write(body)
        m = build_manifest(tmp)
        paths = [e["path"] for e in m["entries"]]
        assert paths == ["a.txt", "sub/b.txt"], paths
        assert m["rate"] == 0.05 and m["count"] == 2
        assert verify_manifest(m)
        m["entries"][0]["sha256"] = "0" * 64
        assert not verify_manifest(m)
        again = build_manifest(tmp)
        assert again["final_close"] == build_manifest(tmp)["final_close"]
    print("self-check ok")
    return 0


def main(argv) -> int:
    if "--self-check" in argv:
        return self_check()
    args = [a for a in argv if not a.startswith("--")]
    here = os.path.dirname(os.path.abspath(__file__))
    root = args[0] if args else os.path.dirname(here)
    m = build_manifest(root)
    if "--json" in argv:
        print(json.dumps(m, indent=2))
    else:
        for e in m["entries"]:
            print("%s  %s  %s" % (e["close"], e["sha256"][:12], e["path"]))
        print("final_close=%s count=%d pack=%s" % (m["final_close"], m["count"], m["pack"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
