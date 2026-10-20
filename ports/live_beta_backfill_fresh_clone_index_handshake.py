#!/usr/bin/env python3
"""Live beta step 5: backfill_check fresh-clone index handshake + port map.

Advances docs/BETA_TO_OS.md, section "Live beta", step 5:
"`python scripts/backfill_check.py` exits 0" -- read on its own premise,
"A fresh clone:".

Already landed for step 5 and not redone here: the exit code (9244444,
adfb204), the coverage of the 12-path EXPECTED list (829a789) and the
quant-error substance gate (JuniorQuant dbafb2e).  This file takes the
fresh-clone side.  backfill_check.py asks Path.is_file() of the working
tree, so on a developer tree it can exit 0 for files a fresh clone will
never get.  In-process, with no git binary, this script:

  1. parses .git/index (v2/v3, stdlib struct) and maps every EXPECTED path
     to an index entry by exact case: untracked-but-present is -1 (passes
     here, fails on a fresh clone), case-only match is -1 (passes on the
     case-insensitive Macs at Home/Flagstaff, fails on Linux FieldCore/OSai);
  2. checks each entry's git mode: 100644/100755 +1, symlink 120000 is an
     advisory (is_file follows it), gitlink 160000 is -1 (empty in a clone);
  3. hashes the working file as a git blob and compares it with the index
     sha: a local edit is an advisory (step 5 attests this disk, the clone
     gets the committed blob);
  4. flags case-folding collisions anywhere on an EXPECTED path's way down;
  5. checks the invocation itself: backfill_check main() exits 0 from a
     foreign cwd (ROOT comes from __file__), its source parses at the
     pyproject requires-python floor, and whether the bare `python` named in
     the step is on PATH (advisory if only python3 is);
  6. port map: the bitnetd 127.0.0.1:8765 and i2sd 127.0.0.1:8767 sources
     are tracked regular files whose committed blob equals the disk copy and
     that carry their port literal next to 127.0.0.1, never the wildcard.

Hand-off between Home, OSai, FieldCore, Flagstaff and JuniorLLM is one trit
pack (+1 pass, 0 advisory, -1 fail), base rate 0.05, closed with
sha256(prev || pack || text)[:16].  No sockets, no subprocess, no shell, no
download, no model pull.  Stdlib only.  mint, socket, layer_one, fetch and
ue5_launch are all false.

    python3 ports/live_beta_backfill_fresh_clone_index_handshake.py
    python3 ports/live_beta_backfill_fresh_clone_index_handshake.py --json
    python3 ports/live_beta_backfill_fresh_clone_index_handshake.py --selftest

Exit 1 if any row is -1 or a flag flipped; advisories keep exit 0.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKFILL = "scripts/backfill_check.py"
GENESIS = "0" * 16
RATE = 0.05
PARTIES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
FLAGS = {
    "bind": "127.0.0.1",
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
    "model_pull": False,
}
PORT_SOURCES = {
    "rails/linux/bitnetd.service": ("bitnetd", 8765),
    "rails/linux/bitnetd.sh": ("bitnetd", 8765),
    "rails/linux/i2sd.py": ("i2sd", 8767),
}
LOOPBACK = "127.0.0.1"
WILDCARD = ".".join(["0"] * 4)
REGULAR = {0o100644, 0o100755}
SYMLINK = 0o120000
GITLINK = 0o160000


def row(key: str, trit: int, note: str) -> dict:
    return {"key": key, "trit": trit, "note": note}


# ---------------------------------------------------------------- git index
def git_dir(root: Path) -> Path | None:
    g = root / ".git"
    if g.is_dir():
        return g
    if g.is_file():
        txt = g.read_text(encoding="utf-8", errors="replace").strip()
        if txt.startswith("gitdir:"):
            p = Path(txt.split(":", 1)[1].strip())
            return p if p.is_absolute() else (root / p).resolve()
    return None


def parse_index(data: bytes) -> dict:
    """Return {path: (mode, sha_hex)} from a v2/v3 git index."""
    if len(data) < 12 or data[:4] != b"DIRC":
        raise ValueError("not a git index")
    version, count = struct.unpack(">II", data[4:12])
    if version not in (2, 3):
        raise ValueError(f"index v{version} not parsed (v2/v3 only)")
    out, off = {}, 12
    for _ in range(count):
        mode = struct.unpack(">I", data[off + 24:off + 28])[0]
        sha = data[off + 40:off + 60].hex()
        flags = struct.unpack(">H", data[off + 60:off + 62])[0]
        head = 62
        if version == 3 and flags & 0x4000:
            head += 2
        end = data.index(b"\0", off + head)
        path = data[off + head:end].decode("utf-8", errors="surrogateescape")
        entry_len = (head + (end - off - head) + 8) & ~7
        off += entry_len
        out[path] = (mode, sha)
    return out


def build_index(entries: dict, version: int = 2) -> bytes:
    """Write a minimal v2/v3 index (selftest only)."""
    body = b""
    for path in sorted(entries):
        mode, sha = entries[path]
        p = path.encode()
        flags = min(len(p), 0xFFF)
        head = struct.pack(">10I", 0, 0, 0, 0, 0, 0, mode, 0, 0, 0)
        head += bytes.fromhex(sha) + struct.pack(">H", flags | (0x4000 if version == 3 else 0))
        if version == 3:
            head += struct.pack(">H", 0)
        n = len(head) + len(p)
        pad = ((n + 8) & ~7) - n
        body += head + p + b"\0" * pad
    data = b"DIRC" + struct.pack(">II", version, len(entries)) + body
    return data + hashlib.sha1(data).digest()


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# --------------------------------------------------------------- backfill
def read_expected(src: str) -> list[str]:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "EXPECTED":
                    return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    return []


def requires_floor(root: Path) -> tuple[int, int] | None:
    pp = root / "pyproject.toml"
    if not pp.is_file():
        return None
    m = re.search(r'requires-python\s*=\s*"\s*>=\s*(\d+)\.(\d+)', pp.read_text(encoding="utf-8"))
    return (int(m.group(1)), int(m.group(2))) if m else None


def run_backfill_foreign_cwd(path: Path) -> tuple[int | None, str]:
    spec = importlib.util.spec_from_file_location("_hp_backfill_fresh", path)
    mod = importlib.util.module_from_spec(spec)
    buf, here = io.StringIO(), os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            with contextlib.redirect_stdout(buf):
                spec.loader.exec_module(mod)
                code = mod.main()
        except Exception as exc:  # report, never crash
            return None, f"{type(exc).__name__}: {exc}"
        finally:
            os.chdir(here)
    return code, buf.getvalue().strip()


# ------------------------------------------------------------------ checks
def index_rows(root: Path, paths: list[str], index: dict, prefix: str) -> list[dict]:
    rows = []
    folded: dict[str, list[str]] = {}
    for p in index:
        folded.setdefault(p.lower(), []).append(p)
    dirs_folded: dict[str, set[str]] = {}
    for p in index:
        parts = p.split("/")
        for i in range(1, len(parts)):
            d = "/".join(parts[:i])
            dirs_folded.setdefault(d.lower(), set()).add(d)
    for p in paths:
        key = f"{prefix}:{p}"
        on_disk = (root / p).is_file()
        if p not in index:
            twins = [q for q in folded.get(p.lower(), []) if q != p]
            if twins:
                rows.append(row(key, -1, f"case-only index match {twins[0]}; Linux clone has no {p}"))
            elif on_disk:
                rows.append(row(key, -1, "on disk but untracked; passes here, missing in a fresh clone"))
            else:
                rows.append(row(key, -1, "not tracked and not on disk"))
            continue
        mode, sha = index[p]
        if mode == GITLINK:
            rows.append(row(key, -1, "gitlink 160000; empty dir in a fresh clone"))
            continue
        if mode == SYMLINK:
            rows.append(row(key, 0, "symlink 120000; is_file follows target, clone may dangle"))
        elif mode not in REGULAR:
            rows.append(row(key, 0, f"unusual mode {mode:o}"))
        else:
            rows.append(row(key, 1, f"tracked {mode:o} {sha[:10]}"))
        if on_disk and mode != SYMLINK:
            disk = blob_sha((root / p).read_bytes())
            if disk != sha:
                rows.append(row(key + ":blob", 0, f"local edit: disk {disk[:10]} != index {sha[:10]}"))
        coll = [q for q in folded.get(p.lower(), []) if q != p]
        parts = p.split("/")
        for i in range(1, len(parts)):
            d = "/".join(parts[:i])
            coll += sorted(x for x in dirs_folded.get(d.lower(), set()) if x != d)
        if coll:
            rows.append(row(key + ":case", -1, f"case-fold collision with {coll[0]} on a Mac clone"))
    return rows


def port_rows(root: Path, index: dict) -> list[dict]:
    rows = index_rows(root, list(PORT_SOURCES), index, "port-src")
    for p, (svc, port) in PORT_SOURCES.items():
        key = f"port:{svc}:{port}:{p}"
        f = root / p
        if not f.is_file():
            rows.append(row(key, -1, "source missing"))
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        if WILDCARD in txt:
            rows.append(row(key, -1, "wildcard bind literal present"))
        elif str(port) in txt and LOOPBACK in txt:
            rows.append(row(key, 1, f"{svc} {LOOPBACK}:{port} literal in committed source"))
        else:
            rows.append(row(key, 0, f"{port} or {LOOPBACK} not literal here (env/default elsewhere)"))
    return rows


def invocation_rows(root: Path) -> list[dict]:
    rows = []
    bf = root / BACKFILL
    if not bf.is_file():
        return [row("invoke:present", -1, f"{BACKFILL} missing")]
    src = bf.read_text(encoding="utf-8")
    code, out = run_backfill_foreign_cwd(bf)
    rows.append(row("invoke:foreign-cwd", 1 if code == 0 else -1,
                    f"main() from temp cwd -> {code} ({out[:60]})"))
    floor = requires_floor(root)
    if floor:
        try:
            ast.parse(src, feature_version=floor)
            rows.append(row("invoke:py-floor", 1, f"parses at requires-python >= {floor[0]}.{floor[1]}"))
        except SyntaxError as exc:
            rows.append(row("invoke:py-floor", -1, f"syntax above floor {floor}: {exc.msg}"))
    else:
        rows.append(row("invoke:py-floor", 0, "no requires-python floor found"))
    shebang = src.splitlines()[0] if src else ""
    if shutil.which("python"):
        rows.append(row("invoke:python", 1, "bare `python` on PATH as the step says"))
    else:
        rows.append(row("invoke:python", 0,
                        f"no bare `python` on PATH; step 5 needs python3 here (shebang {shebang!r})"))
    return rows


def flag_rows() -> list[dict]:
    bad = [k for k, v in FLAGS.items() if k != "bind" and v is not False]
    ok = FLAGS["bind"] == LOOPBACK and not bad
    return [row("flags", 1 if ok else -1, "bind 127.0.0.1; mint/socket/layer_one/fetch/ue5_launch/model_pull false"
                if ok else f"flipped: {bad}")]


def handshake(root: Path) -> dict:
    rows = flag_rows()
    gd = git_dir(root)
    index = None
    if gd is None or not (gd / "index").is_file():
        rows.append(row("index", 0, "no .git/index (tarball?); fresh-clone provenance unknown"))
    else:
        try:
            index = parse_index((gd / "index").read_bytes())
            rows.append(row("index", 1, f"{len(index)} entries parsed in-process"))
        except ValueError as exc:
            rows.append(row("index", 0, str(exc)))
    bf = root / BACKFILL
    expected = read_expected(bf.read_text(encoding="utf-8")) if bf.is_file() else []
    rows.append(row("expected", 1 if expected else -1, f"{len(expected)} EXPECTED paths read via ast"))
    if index is not None:
        rows += index_rows(root, expected + [BACKFILL], index, "fresh")
        rows += port_rows(root, index)
    rows += invocation_rows(root)
    return close(rows)


# ------------------------------------------------------------ pack + close
def pack_trits(trits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            v = v * 3 + (t + 1)
        out.append(v)
    return bytes(out)


def unpack_trits(data: bytes, n: int) -> list[int]:
    out = []
    for b in data:
        for _ in range(5):
            out.append(b % 3 - 1)
            b //= 3
    return out[:n]


def close(rows: list[dict], prev: str = GENESIS) -> dict:
    trits = [r["trit"] for r in rows]
    pack = pack_trits(trits)
    text = "\n".join(f"{r['trit']:+d} {r['key']} {r['note']}" for r in rows)
    seal = hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]
    adv = trits.count(0)
    return {
        "item": "BETA_TO_OS Live beta step 5 (fresh-clone side)",
        "parties": list(PARTIES),
        "rate": RATE,
        "pass": trits.count(1),
        "advisory": adv,
        "fail": trits.count(-1),
        "advisory_share": round(adv / len(trits), 3) if trits else 0.0,
        "prev": prev,
        "pack": pack.hex(),
        "close": seal,
        "flags": FLAGS,
        "rows": rows,
        "text": text,
    }


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok = ok and cond

    tr = [1, 0, -1, 1, 1, -1, 0]
    check(unpack_trits(pack_trits(tr), len(tr)) == tr, "trit pack round-trips")
    a, b = close([row("k", 1, "n")]), close([row("k", 1, "n")])
    check(a["close"] == b["close"] and len(a["close"]) == 16, "close is deterministic, 16 hex")
    check(close([row("k", 0, "n")])["close"] != a["close"], "close moves with the pack")
    check(flag_rows()[0]["trit"] == 1, "flags: bind loopback, all switches false")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = {
            "docs/A.md": b"a\n",
            "docs/B.md": b"b\n",
            "docs/Edit.md": b"new\n",
            "Docs2/C.md": b"c\n",
        }
        for p, d in files.items():
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            (root / p).write_bytes(d)
        (root / "docs/untracked.md").write_bytes(b"u\n")
        idx = {
            "docs/A.md": (0o100644, blob_sha(b"a\n")),
            "docs/b.md": (0o100644, blob_sha(b"b\n")),
            "docs/Edit.md": (0o100644, blob_sha(b"old\n")),
            "docs/link.md": (SYMLINK, blob_sha(b"A.md")),
            "vendor/sub": (GITLINK, "ab" * 20),
            "Docs2/C.md": (0o100644, blob_sha(b"c\n")),
            "docs2/D.md": (0o100644, blob_sha(b"d\n")),
        }
        for v in (2, 3):
            parsed = parse_index(build_index(idx, v))
            check(parsed == idx, f"index v{v} parse matches writer")
        parsed = parse_index(build_index(idx, 2))
        want = ["docs/A.md", "docs/B.md", "docs/untracked.md", "docs/Edit.md",
                "docs/link.md", "vendor/sub", "Docs2/C.md", "docs/none.md"]
        got = {r["key"].split(":", 1)[1]: r for r in index_rows(root, want, parsed, "t")}
        check(got["docs/A.md"]["trit"] == 1, "tracked regular file passes")
        check(got["docs/B.md"]["trit"] == -1 and "case-only" in got["docs/B.md"]["note"],
              "case-only match fails (Mac passes, Linux clone misses)")
        check(got["docs/untracked.md"]["trit"] == -1, "untracked-on-disk fails")
        check(got["docs/Edit.md:blob"]["trit"] == 0, "local edit is an advisory")
        check(got["docs/link.md"]["trit"] == 0, "symlink is an advisory")
        check(got["vendor/sub"]["trit"] == -1, "gitlink fails")
        check(got["Docs2/C.md:case"]["trit"] == -1, "case-fold dir collision fails")
        check(got["docs/none.md"]["trit"] == -1, "absent path fails")
        try:
            parse_index(b"DIRC" + struct.pack(">II", 4, 0))
            check(False, "index v4 refused")
        except ValueError:
            check(True, "index v4 refused (advisory upstream)")

        (root / "scripts").mkdir()
        (root / BACKFILL).write_text(
            "from pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n"
            "EXPECTED = ['docs/A.md']\n"
            "def main():\n    m=[p for p in EXPECTED if not (ROOT/p).is_file()]\n"
            "    print('missing', m or 'none')\n    return 1 if m else 0\n", encoding="utf-8")
        (root / "pyproject.toml").write_text('requires-python = ">=3.11"\n', encoding="utf-8")
        inv = {r["key"]: r for r in invocation_rows(root)}
        check(inv["invoke:foreign-cwd"]["trit"] == 1, "backfill main() exits 0 from a foreign cwd")
        check(inv["invoke:py-floor"]["trit"] == 1, "backfill parses at the requires-python floor")
        check(read_expected((root / BACKFILL).read_text()) == ["docs/A.md"], "EXPECTED read via ast")

        for p, (svc, port) in PORT_SOURCES.items():
            body = f"bind {LOOPBACK}:{port}\n".encode()
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            (root / p).write_bytes(body)
            idx[p] = (0o100644, blob_sha(body))
        pr = [r for r in port_rows(root, idx) if r["key"].startswith("port:")]
        check(all(r["trit"] == 1 for r in pr), "port map: 8765/8767 loopback literals pass")
        (root / "rails/linux/i2sd.py").write_bytes(f"bind {WILDCARD}:8767\n".encode())
        pr = {r["key"]: r for r in port_rows(root, idx)}
        check(pr["port:i2sd:8767:rails/linux/i2sd.py"]["trit"] == -1, "wildcard bind fails")
        check(pr["port-src:rails/linux/i2sd.py:blob"]["trit"] == 0, "edited port source is an advisory")

    print("selftest", "pass" if ok else "fail")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    rep = handshake(Path(a.root).resolve())
    if a.json:
        print(json.dumps({k: v for k, v in rep.items() if k != "text"}, indent=2))
    else:
        print(rep["text"])
        print(f"pass {rep['pass']} advisory {rep['advisory']} fail {rep['fail']} "
              f"rate {RATE} close {rep['close']}")
    return 1 if rep["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
