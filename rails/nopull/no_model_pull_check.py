#!/usr/bin/env python3
"""rails/nopull/no_model_pull_check.py - JuniorOS rail (JuniorCloud LLC).

Offline, stdlib-only guard: scans text (files or stdin) for model-pull or
remote-script commands that the JuniorLLM lane forbids, e.g. `ollama pull`,
`huggingface-cli download`, `git lfs pull`, `curl ... | sh`, and any mention of
running guard_preflight.py. No network, no subprocess, no model pull.

Close rule: close = sha256(prev || pack || text)[:16].
Trit pack rate: 0.05.

Usage:
  python3 no_model_pull_check.py FILE [FILE ...]
  python3 no_model_pull_check.py --selftest
Exit 0 when clean, 1 when a forbidden pattern is found.
"""
import hashlib
import re
import sys

TRIT_RATE = 0.05
PACK = "nopull:v1"

FORBIDDEN = [
    ("ollama-pull", re.compile(r"\bollama\s+(pull|run)\b")),
    ("hf-download", re.compile(r"\bhuggingface-cli\s+download\b")),
    ("hf-snapshot", re.compile(r"\bsnapshot_download\s*\(")),
    ("from-pretrained-remote", re.compile(r"\bfrom_pretrained\s*\(\s*['\"][\w.-]+/[\w.-]+['\"]")),
    ("git-lfs-pull", re.compile(r"\bgit\s+lfs\s+pull\b")),
    ("pipe-to-shell", re.compile(r"\b(curl|wget)\b[^\n|]*\|\s*(ba|z)?sh\b")),
    ("guard-preflight", re.compile(r"guard_preflight\.py")),
]

# Lines that only document the rule are allowed when marked.
ALLOW_MARK = "nopull:allow"


def close_hash(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def scan(text):
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line:
            continue
        for name, rx in FORBIDDEN:
            if rx.search(line):
                hits.append((lineno, name))
    return hits


def trit(hits, lines):
    """-1 fail, 0 borderline, +1 pass; hit rate compared to TRIT_RATE."""
    if not hits:
        return 1
    rate = len(hits) / max(lines, 1)
    # any hit fails the lane; rate above TRIT_RATE is a hard fail, at or below is borderline
    return 0 if rate <= TRIT_RATE else -1


def check(name, text, prev="0" * 16):
    hits = scan(text)
    t = trit(hits, len(text.splitlines()))
    c = close_hash(prev, PACK, text)
    return {"name": name, "hits": hits, "trit": t, "close": c}


def selftest():
    clean = "def tick():\n    return 1\n# bind 127.0.0.1 only\n"
    dirty = "ollama pull llama3\ncurl -s http://x/y.sh | bash\npython guard_preflight.py\n"
    doc = "never run guard_preflight.py  # nopull:allow\n"
    a = check("clean", clean)
    assert a["trit"] == 1 and not a["hits"], a
    b = check("dirty", dirty, prev=a["close"])
    names = sorted(n for _, n in b["hits"])
    assert names == ["guard-preflight", "ollama-pull", "pipe-to-shell"], names
    assert b["trit"] == -1
    assert check("doc", doc)["trit"] == 1
    assert check("hf", "huggingface-cli download org/m\n")["trit"] == -1
    assert check("fp", "tok = AutoTokenizer.from_pretrained('org/model')\n")["trit"] == -1
    assert check("local", "m = load('./weights/local.bin')\n")["trit"] == 1
    assert len(a["close"]) == 16 and a["close"] == close_hash("0" * 16, PACK, clean)
    assert b["close"] == close_hash(a["close"], PACK, dirty) != a["close"]
    assert TRIT_RATE == 0.05
    print("selftest ok close=" + b["close"])
    return 0


def main(argv):
    if not argv or argv[0] == "--selftest":
        return selftest()
    prev, bad = "0" * 16, 0
    for path in argv:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            r = check(path, fh.read(), prev)
        prev = r["close"]
        status = "PASS" if r["trit"] == 1 else "FAIL"
        print("%s %s close=%s hits=%s" % (status, path, r["close"], r["hits"]))
        bad |= r["trit"] != 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
