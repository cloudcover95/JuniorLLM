#!/usr/bin/env python3
"""Live beta step 5 sparse-checkout side: skip-worktree and sparse-pattern trits for backfill_check EXPECTED. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 5
("`python scripts/backfill_check.py` exits 0"), on its "A fresh clone"
premise. Quant Chief pair A, slot 2026-10-21 18:05 MT, non-quant-error side.
Trit Pack & I2_S side: the trit pack and I2_S key. Winsor & Quant Error
covers the quant-error bound side of the same item (JuniorQuant dbafb2e).

Gap it fills (not in BitNet-mlx 9244444 exit-code trits, JuniorLLM 829a789
coverage handshake, 5032022 .git/index trits, c93a5ec index handshake,
e82c972 HEAD-tree trits, e1ea106 step-order writer map, 5b35e4d archive
attributes, stocksnode 0eaa9a6 receipt ledger, JuniorQuant adfb204/dbafb2e
quant-error gates): none of those read sparse-checkout state. The index
reader 5032022 skips the extended-flag word without decoding it, so an
EXPECTED path with the skip-worktree bit set still reads as tracked and
stage 0. A clone made with `git clone --sparse` (or `--filter=blob:none
--sparse`, the light path onto a Home or Flagstaff Mac or a small OSai /
FieldCore image) or narrowed later with `git sparse-checkout set` keeps the
file in HEAD and in the index but never writes it to the working tree, so
backfill_check.py exits 1 while every earlier key passes.

This file reads, offline and without git: .git (dir or `gitdir:` file),
.git/config plus .git/config.worktree when extensions.worktreeConfig is on
(core.sparseCheckout, core.sparseCheckoutCone, index.sparse), the
info/sparse-checkout pattern file in cone form (`/*`, `!/*/`, `/D/`,
`!/D/*/`; anything else falls back to non-cone as git does) or non-cone
gitignore form (last match wins, `!` re-includes, trailing `/` dirs only,
slash-less patterns match a basename at any depth, a directory verdict is
inherited by its subtree), and .git/index v2-v4 with checksum: flags bit
0x8000 assume-valid, extended bit 0x4000 skip-worktree, and sparse-index
directory entries (mode 040000, trailing `/`, `sdir` extension). One
balanced trit per EXPECTED path, plus scripts/backfill_check.py itself:
    +1  no skip-worktree bit, not under a sparse-index directory entry,
        included by the active sparse patterns (or sparse is off), and the
        file is on disk
     0  assume-valid bit set (git will not notice the file going away)
    -1  skip-worktree set, folded into a sparse-index directory, excluded
        by the active patterns, or absent on disk
plus one repo trit: sparse config is coherent (+1) or not (0: patterns
file with sparse off, cone mode with non-cone lines, stray skip-worktree
bits with sparse off, sparse on with no patterns file, `sdir` with
index.sparse off). Also reported, outside the trits: which EXPECTED paths a
default `git clone --sparse` (root-only cone) would leave out.

Verdict: sparse-faithful when no -1 and the 0 fraction is <= RATE (0.05);
release RATE when faithful, else 0.0 (held). EXPECTED is read from
scripts/backfill_check.py with `ast` (literal list only; never imported or
run). No git binary, no subprocess, no other file of ours is imported.

JTR1 (not GGUF): b"JTR1" | gamma <f32 LE, absmean> | n <u32 LE> | 2-bit
payload MSB first (0 -> 00, +1 -> 01, -1 -> 10; 11 rejected).
I2_S block: code = trit + 1, four per byte MSB first, padded with code 1
(zero) to a 128-trit block, as BitNet I2_S stores ternary.
Close rule: sha256(prev || pack || text)[:16]; prev 16-hex ASCII, pack =
JTR1 bytes, text = canonical verdict JSON (sorted keys, compact; includes
i2s_sha). Rate 0.05. Compatible engines: JuniorLLM loader (JTR1),
BitNet.cpp I2_S block layout, bitnet-mlx. Ports touched: Home, OSai,
FieldCore, Flagstaff, JuniorLLM. Bind 127.0.0.1 only (never used: no
sockets). mint/socket/layer_one/fetch/ue5_launch false. No model pull, no
GGUF download, no network, no subprocess.

Usage: python3 scripts/backfill_sparse_checkout_i2s_key.py --selftest
       python3 scripts/backfill_sparse_checkout_i2s_key.py [--root DIR] [--prev HEX16] [--json]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import struct
import sys
import tempfile
from pathlib import Path

RATE = 0.05
GENESIS = "0" * 16
BACKFILL = "scripts/backfill_check.py"
ITEM = "BETA_TO_OS Live beta step 5 (sparse-checkout / skip-worktree side)"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM loader (JTR1)", "BitNet.cpp I2_S block layout", "bitnet-mlx")
FLAGS = {"bind": "127.0.0.1", "mint": False, "socket": False,
         "layer_one": False, "fetch": False, "ue5_launch": False}
ASSUME_VALID = 0x8000
EXTENDED = 0x4000
SKIP_WORKTREE = 0x4000
DIR_MODE = 0o040000


# ------------------------------------------------------------- git dir
def git_dir(root: Path):
    g = root / ".git"
    if g.is_dir():
        return g
    if g.is_file():
        m = re.match(r"gitdir:\s*(.+)", g.read_text(encoding="utf-8", errors="replace").strip())
        if m:
            p = Path(m.group(1).strip())
            return p if p.is_absolute() else (root / p).resolve()
    return None


def parse_config(text: str) -> dict:
    """Minimal git-config reader: {"section.key": value} (lowercased names)."""
    out, sect = {}, ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        m = re.match(r'\[\s*([A-Za-z0-9.-]+)(?:\s+"([^"]*)")?\s*\]\s*(.*)$', line)
        if m:
            sect = m.group(1).lower() + ("." + m.group(2) if m.group(2) is not None else "")
            line = m.group(3).strip()
            if not line:
                continue
        line = re.split(r"\s[#;]", line, maxsplit=1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[sect + "." + k.strip().lower()] = v.strip().strip('"')
        elif line:
            out[sect + "." + line.lower()] = "true"
    return out


def cfg_bool(cfg: dict, key: str, default: bool = False) -> bool:
    v = cfg.get(key)
    if v is None:
        return default
    return v.lower() in ("true", "yes", "on", "1")


def load_config(gd: Path) -> dict:
    cfg = {}
    if gd is not None and (gd / "config").is_file():
        cfg.update(parse_config((gd / "config").read_text(encoding="utf-8", errors="replace")))
        if cfg_bool(cfg, "extensions.worktreeconfig") and (gd / "config.worktree").is_file():
            cfg.update(parse_config((gd / "config.worktree").read_text(encoding="utf-8", errors="replace")))
    return cfg


# --------------------------------------------------------------- index
def _varint(buf: bytes, i: int):
    c = buf[i]; i += 1
    val = c & 0x7F
    while c & 0x80:
        c = buf[i]; i += 1
        val = ((val + 1) << 7) | (c & 0x7F)
    return val, i


def read_index_flags(blob: bytes) -> dict:
    """Return {"ok", "version", "entries": {path: (mode, assume_valid, skip_worktree)}, "exts"}."""
    if len(blob) < 32 or blob[:4] != b"DIRC":
        return {"ok": 0, "version": None, "entries": {}, "exts": [], "note": "no DIRC header"}
    body, tail = blob[:-20], blob[-20:]
    if hashlib.sha1(body).digest() != tail:
        return {"ok": -1, "version": None, "entries": {}, "exts": [], "note": "checksum mismatch"}
    version, count = struct.unpack(">II", blob[4:12])
    if version not in (2, 3, 4):
        return {"ok": 0, "version": version, "entries": {}, "exts": [], "note": "unsupported version"}
    i, prev, ents = 12, b"", {}
    for _ in range(count):
        start = i
        mode = struct.unpack(">I", body[i + 24:i + 28])[0]
        flags = struct.unpack(">H", body[i + 60:i + 62])[0]
        i += 62
        ext = 0
        if version >= 3 and flags & EXTENDED:
            ext = struct.unpack(">H", body[i:i + 2])[0]
            i += 2
        if version == 4:
            strip, i = _varint(body, i)
            end = body.index(b"\0", i)
            name = prev[:len(prev) - strip] + body[i:end]
            i = end + 1
        else:
            end = body.index(b"\0", i)
            name = body[i:end]
            i = start + ((end - start) // 8 + 1) * 8
        prev = name
        ents[name.decode("utf-8", "surrogateescape")] = (
            mode, bool(flags & ASSUME_VALID), bool(ext & SKIP_WORKTREE))
    exts = []
    while i + 8 <= len(body):
        sig, size = body[i:i + 4], struct.unpack(">I", body[i + 4:i + 8])[0]
        exts.append(sig.decode("latin-1"))
        i += 8 + size
    return {"ok": 1, "version": version, "entries": ents, "exts": exts, "note": "checksum ok"}


# ------------------------------------------------------ sparse patterns
def _wild_re(pat: str) -> str:
    out, i, n = [], 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**", i) and (i == 0 or pat[i - 1] == "/") and (i + 2 == n or pat[i + 2] == "/"):
                if i + 2 == n:
                    out.append(".*"); i += 2
                else:
                    out.append("(?:.*/)?"); i += 3
                continue
            while i < n and pat[i] == "*":
                i += 1
            out.append("[^/]*")
            continue
        if c == "?":
            out.append("[^/]")
        elif c == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(pat[i]))
        elif c == "[":
            j = pat.find("]", i + 2)
            if j < 0:
                out.append(re.escape(c))
            else:
                body = pat[i + 1:j]
                neg = body[:1] in ("!", "^")
                body = (body[1:] if neg else body).replace("\\", "\\\\")
                out.append("(?!/)[" + ("^" if neg else "") + body + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def parse_nocone(text: str):
    """Gitignore-style rules: (regex, negate, dironly, basename_only)."""
    rules = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip() or line.startswith("#"):
            continue
        line = re.sub(r"(?<!\\)\s+$", "", line)
        neg = line.startswith("!")
        if neg:
            line = line[1:]
        dironly = line.endswith("/") and len(line) > 1
        line = line.rstrip("/") if dironly else line
        base = "/" not in line
        line = line.lstrip("/")
        if line:
            rules.append((re.compile(_wild_re(line), re.S), neg, dironly, base))
    return rules


def nocone_included(rules, path: str) -> bool:
    parts = path.split("/")
    state = False
    for k in range(1, len(parts) + 1):
        sub, isdir = "/".join(parts[:k]), k < len(parts)
        hit = None
        for rx, neg, dironly, base in rules:
            if dironly and not isdir:
                continue
            target = parts[k - 1] if base else sub
            if rx.fullmatch(target):
                hit = not neg
        if hit is not None:
            state = hit
    return state


CONE_LINE = re.compile(r"^(/\*|!/\*/|/(.+)/|!/(.+)/\*/)$")


def parse_cone(text: str):
    """Return (recursive_dirs, parent_dirs, root_files, bad_lines) or bad lines non-empty."""
    rec, par, root_files, bad = set(), set(), False, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = CONE_LINE.match(line)
        if not m or (m.group(2) and "*" in m.group(2)) or (m.group(3) and "*" in m.group(3)):
            bad.append(line)
            continue
        if line == "/*":
            root_files = True
        elif line == "!/*/":
            pass
        elif m.group(2) is not None:
            rec.add(m.group(2).replace("\\", ""))
        else:
            par.add(m.group(3).replace("\\", ""))
    rec -= par
    return rec, par, root_files, bad


def cone_included(cone, path: str) -> bool:
    rec, par, root_files, _ = cone
    if "/" not in path:
        return root_files
    d = path.rsplit("/", 1)[0]
    parts = d.split("/")
    if any("/".join(parts[:k]) in rec for k in range(1, len(parts) + 1)):
        return True
    return d in par


# --------------------------------------------------------------- audit
def read_expected(root: Path):
    src = (root / BACKFILL).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "EXPECTED" for t in node.targets):
            if isinstance(node.value, (ast.List, ast.Tuple)) and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str)
                    for e in node.value.elts):
                return [e.value for e in node.value.elts]
    raise ValueError("EXPECTED literal list not found")


def sparse_state(root: Path):
    gd = git_dir(root)
    cfg = load_config(gd)
    on = cfg_bool(cfg, "core.sparsecheckout")
    cone_mode = cfg_bool(cfg, "core.sparsecheckoutcone")
    sp = gd / "info" / "sparse-checkout" if gd is not None else None
    text = sp.read_text(encoding="utf-8", errors="replace") if sp is not None and sp.is_file() else None
    idx = {"ok": 0, "version": None, "entries": {}, "exts": [], "note": "no index"}
    if gd is not None and (gd / "index").is_file():
        idx = read_index_flags((gd / "index").read_bytes())
    notes, matcher, used = [], None, "off"
    if on and text is not None:
        if cone_mode:
            cone = parse_cone(text)
            if cone[3]:
                notes.append("cone mode but non-cone lines %r: git falls back to non-cone" % cone[3][:3])
                rules = parse_nocone(text)
                matcher, used = (lambda p: nocone_included(rules, p)), "non-cone(fallback)"
            else:
                matcher, used = (lambda p: cone_included(cone, p)), "cone"
        else:
            rules = parse_nocone(text)
            matcher, used = (lambda p: nocone_included(rules, p)), "non-cone"
    stray = sorted(p for p, e in idx["entries"].items() if e[2])
    if text is not None and not on:
        notes.append("info/sparse-checkout present but core.sparseCheckout off (inert until enabled)")
    if on and text is None:
        notes.append("core.sparseCheckout on but no info/sparse-checkout file")
    if stray and not on:
        notes.append("%d skip-worktree entries with sparse off (manual update-index?)" % len(stray))
    if "sdir" in idx["exts"] and not cfg_bool(cfg, "index.sparse"):
        notes.append("sparse index (sdir) present with index.sparse off")
    if idx["ok"] == -1:
        notes.append("index checksum mismatch")
    repo_trit = 0 if notes else 1
    return {"on": on, "mode": used, "matcher": matcher, "index": idx,
            "notes": notes, "repo_trit": repo_trit}


def path_trit(root: Path, st: dict, rel: str):
    ents = st["index"]["entries"]
    ent = ents.get(rel)
    for d in (p for p, e in ents.items() if e[0] == DIR_MODE and e[2] and p.endswith("/")):
        if rel.startswith(d):
            return -1, "folded into sparse-index dir %s (not on disk)" % d
    if ent is not None and ent[2]:
        return -1, "skip-worktree bit: git keeps it out of the working tree"
    if st["matcher"] is not None and not st["matcher"](rel):
        return -1, "excluded by %s sparse patterns" % st["mode"]
    if not (root / rel).is_file():
        return -1, "absent on disk"
    if ent is not None and ent[1]:
        return 0, "assume-valid bit: git will not notice it going away"
    return 1, "sparse-faithful" + (" (%s, included)" % st["mode"] if st["on"] else "")


def audit(root: Path, expected=None):
    root = Path(root)
    exp = list(expected) if expected is not None else read_expected(root)
    st = sparse_state(root)
    rows, trits = [], []
    for p in exp + [BACKFILL]:
        t, why = path_trit(root, st, p)
        rows.append((p, t, why))
        trits.append(t)
    rows.append(("sparse-config", st["repo_trit"], "; ".join(st["notes"]) or "coherent"))
    trits.append(st["repo_trit"])
    drops = [p for p in exp + [BACKFILL] if "/" in p]
    meta = {"sparse_on": st["on"], "mode": st["mode"], "notes": st["notes"],
            "index_version": st["index"]["version"], "clone_sparse_drops": drops}
    return rows, trits, meta


# --------------------------------------------------------- pack and key
def jtr1_pack(trits) -> bytes:
    n = len(trits)
    gamma = sum(abs(t) for t in trits) / n if n else 0.0
    code = {0: 0, 1: 1, -1: 2}
    buf = bytearray((n + 3) // 4)
    for i, t in enumerate(trits):
        buf[i // 4] |= code[t] << (6 - 2 * (i % 4))
    return b"JTR1" + struct.pack("<fI", gamma, n) + bytes(buf)


def jtr1_unpack(blob: bytes):
    if blob[:4] != b"JTR1":
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    if len(blob) < 12 + (n + 3) // 4:
        raise ValueError("short payload")
    dec = {0: 0, 1: 1, 2: -1}
    out = []
    for i in range(n):
        c = (blob[12 + i // 4] >> (6 - 2 * (i % 4))) & 3
        if c == 3:
            raise ValueError("code 11 rejected")
        out.append(dec[c])
    return gamma, out


def i2s_block(trits) -> bytes:
    codes = [t + 1 for t in trits]
    pad = (-len(codes)) % 128 if codes else 128
    codes += [1] * pad
    out = bytearray()
    for i in range(0, len(codes), 4):
        a, b, c, d = codes[i:i + 4]
        out.append((a << 6) | (b << 4) | (c << 2) | d)
    return bytes(out)


def close_key(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("ascii") + pack + text.encode("utf-8")).hexdigest()[:16]


def verdict(rows, trits, meta, prev=GENESIS):
    n = len(trits)
    zero_frac = sum(1 for t in trits if t == 0) / n if n else 1.0
    faithful = n > 0 and -1 not in trits and zero_frac <= RATE
    pack = jtr1_pack(trits)
    i2s = i2s_block(trits)
    body = {
        "item": ITEM, "rate": RATE, "rows": [[p, t, w] for p, t, w in rows],
        "meta": meta, "zero_frac": round(zero_frac, 6), "faithful": faithful,
        "release": RATE if faithful else 0.0, "i2s_sha": hashlib.sha256(i2s).hexdigest()[:16],
        "ports": list(PORTS), "engines": list(ENGINES), "flags": FLAGS,
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return body, pack, i2s, text, close_key(prev, pack, text)


# ------------------------------------------------------------ selftest
EXP = ["docs/A.md", "STATE.md", "rails/linux/x.sh", "rails/swift/G.swift"]


def _index(entries, version=3, exts=b"") -> bytes:
    """Selftest-only DIRC writer: entries = [(path, mode, assume_valid, skip_worktree)]."""
    body = bytearray(b"DIRC" + struct.pack(">II", version, len(entries)))
    for path, mode, av, sw in sorted(entries):
        name = path.encode()
        ext = version >= 3 and sw
        flags = min(len(name), 0xFFF) | (ASSUME_VALID if av else 0) | (EXTENDED if ext else 0)
        e = bytearray(struct.pack(">10I", 0, 0, 0, 0, 0, 0, mode, 0, 0, 0)) + b"\x11" * 20
        e += struct.pack(">H", flags)
        if ext:
            e += struct.pack(">H", SKIP_WORKTREE)
        e += name
        e += b"\0" * (8 - (len(e) % 8))
        body += e
    body += exts
    return bytes(body) + hashlib.sha1(bytes(body)).digest()


def _mk(base: Path, config="", sparse=None, entries=None, absent=(), exts=b"", version=3):
    for p in EXP + [BACKFILL]:
        if p in absent:
            continue
        f = base / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x\n" if p != BACKFILL else
                      ("EXPECTED = [\n" + "".join('    "%s",\n' % q for q in EXP) + "]\n").encode())
    g = base / ".git"
    (g / "info").mkdir(parents=True, exist_ok=True)
    (g / "config").write_text("[core]\n\trepositoryformatversion = 0\n" + config)
    if sparse is not None:
        (g / "info" / "sparse-checkout").write_text(sparse)
    if entries is None:
        entries = [(p, 0o100644, False, False) for p in EXP + [BACKFILL]]
    (g / "index").write_bytes(_index(entries, version, exts))
    return base


def selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL", msg)

    def run(**kw):
        with tempfile.TemporaryDirectory() as td:
            return audit(_mk(Path(td), **kw))

    allp = EXP + [BACKFILL]
    # 1 plain clone, sparse off -> all +1 faithful
    rows, trits, meta = run()
    check(trits == [1] * 6, "plain %r" % trits)
    body, pack, i2s, text, key = verdict(rows, trits, meta)
    check(body["faithful"] and body["release"] == RATE, "plain faithful")
    check(meta["clone_sparse_drops"] == ["docs/A.md", "rails/linux/x.sh", "rails/swift/G.swift", BACKFILL],
          "clone --sparse drops %r" % meta["clone_sparse_drops"])
    # 2 skip-worktree bit on one EXPECTED path, sparse on, file absent
    ent = [(p, 0o100644, False, p == "rails/linux/x.sh") for p in allp]
    cone_rails = "/*\n!/*/\n/docs/\n/scripts/\n/rails/\n!/rails/*/\n/rails/swift/\n"
    r, t, _ = run(config="\tsparseCheckout = true\n\tsparseCheckoutCone = true\n",
                  sparse=cone_rails, entries=ent, absent=("rails/linux/x.sh",))
    check(t == [1, 1, -1, 1, 1, 1] and "skip-worktree" in r[2][2], "skip-worktree -1 %r" % t)
    # 3 skip-worktree bit while sparse off -> path -1 and repo trit 0
    r, t, m = run(entries=ent)
    check(t[2] == -1 and t[5] == 0 and "update-index" in r[5][2], "stray skip %r" % t)
    # 4 cone patterns exclude a path whose bit is not yet set (next reapply drops it)
    r, t, m = run(config="\tsparseCheckout = true\n\tsparseCheckoutCone = true\n", sparse=cone_rails)
    check(t == [1, 1, -1, 1, 1, 1] and m["mode"] == "cone" and "excluded" in r[2][2], "cone exclude %r" % t)
    # cone parent dir includes direct files only, recursive includes subtree, root files
    cone = parse_cone("/*\n!/*/\n/a/\n!/a/*/\n/a/b/\n")
    check(cone_included(cone, "a/f") and cone_included(cone, "a/b/c/d") and cone_included(cone, "R")
          and not cone_included(cone, "a/z/f") and not cone_included(cone, "q/f"), "cone semantics")
    # 5 cone mode with a non-cone line -> fallback, repo trit 0
    r, t, m = run(config="\tsparseCheckout = true\n\tsparseCheckoutCone = true\n",
                  sparse="/*\n*.md\n/scripts/\n/rails/\n")
    check(m["mode"] == "non-cone(fallback)" and t[5] == 0, "cone fallback %r %r" % (m["mode"], t))
    # 6 non-cone gitignore semantics: last match wins, ! re-excludes, dir inheritance
    check(nocone_included(parse_nocone("/*\n"), "docs/A.md"), "non-cone /* takes every root dir")
    rules = parse_nocone("/*\n!/rails/\n/rails/swift/\n*.md\n!docs/\n")
    check(nocone_included(rules, "STATE.md") and not nocone_included(rules, "rails/linux/x.sh")
          and nocone_included(rules, "rails/swift/G.swift") and nocone_included(rules, "docs/A.md"),
          "non-cone semantics")
    check(not nocone_included(parse_nocone("docs/\n"), "x/docs") and
          nocone_included(parse_nocone("docs/\n"), "x/docs/a"), "dir-only pattern")
    r, t, m = run(config="\tsparseCheckout = true\n", sparse="/STATE.md\n/scripts/\n/rails/\n")
    check(t == [-1, 1, 1, 1, 1, 1] and m["mode"] == "non-cone", "non-cone audit %r" % t)
    # 7 assume-valid -> 0, held
    ent7 = [(p, 0o100644, p == "STATE.md", False) for p in allp]
    r7, t, m7 = run(entries=ent7)
    check(t[1] == 0 and not verdict(r7, t, m7)[0]["faithful"], "assume-valid 0 %r" % t)
    # 8 sparse index directory entry folds rails/linux/ + sdir with index.sparse off
    ent8 = [(p, 0o100644, False, False) for p in allp if not p.startswith("rails/linux/")]
    ent8.append(("rails/linux/", DIR_MODE, False, True))
    r, t, m = run(entries=ent8, exts=b"sdir" + struct.pack(">I", 0), absent=("rails/linux/x.sh",),
                  config="\tsparseCheckout = true\n\tsparseCheckoutCone = true\n", sparse=cone_rails)
    check(t[2] == -1 and "sparse-index dir" in r[2][2] and t[5] == 0, "sparse index %r" % t)
    # 9 inert patterns file with sparse off -> paths +1, repo 0; v2 index read
    r, t, m = run(sparse="/*\n")
    check(t[:5] == [1] * 5 and t[5] == 0, "inert patterns %r" % t)
    _, t, _ = run(version=2)
    check(t == [1] * 6, "index v2 %r" % t)
    # 10 worktreeConfig: sparse set only in config.worktree
    with tempfile.TemporaryDirectory() as td:
        b = _mk(Path(td), config="[extensions]\n\tworktreeConfig = true\n", sparse="/STATE.md\n/scripts/\n")
        (b / ".git" / "config.worktree").write_text("[core]\n\tsparseCheckout = true\n")
        _, t, m = audit(b)
        check(m["sparse_on"] and t[0] == -1 and t[1] == 1, "config.worktree %r" % t)
    # 11 checksum mismatch -> repo trit 0
    with tempfile.TemporaryDirectory() as td:
        b = _mk(Path(td))
        ix = bytearray((b / ".git" / "index").read_bytes())
        ix[-1] ^= 0xFF
        (b / ".git" / "index").write_bytes(bytes(ix))
        _, t, _ = audit(b)
        check(t[5] == 0, "checksum mismatch %r" % t)
    # config parser
    c = parse_config('[core]\n\tsparseCheckout = true ; c\n[remote "origin"]\n\turl = x\n[index]\nsparse\n')
    check(cfg_bool(c, "core.sparsecheckout") and c["remote.origin.url"] == "x"
          and cfg_bool(c, "index.sparse"), "config parse %r" % c)
    # JTR1 / I2_S / close rule
    mixed = [1, 0, -1, 1, 1, -1]
    g, back = jtr1_unpack(jtr1_pack(mixed))
    check(back == mixed and abs(g - 5 / 6) < 1e-6, "jtr1 round trip")
    try:
        jtr1_unpack(b"JTR1" + struct.pack("<fI", 1.0, 1) + b"\xc0")
        check(False, "code 11 accepted")
    except ValueError:
        pass
    check(len(i2s) == 32 and i2s[0] == 0b10101010 and i2s[-1] == 0b01010101, "i2s 128-block")
    check(len(i2s_block(mixed * 30)) == 64, "i2s two blocks")
    check(key == close_key(GENESIS, pack, text) and len(key) == 16, "close rule")
    check(key == verdict(rows, trits, meta)[4], "deterministic")
    check(key != verdict(rows, trits, meta, "f" * 16)[4], "prev chains")
    check(RATE == 0.05 and FLAGS["bind"] == "127.0.0.1" and not any(
        FLAGS[k] for k in ("mint", "socket", "layer_one", "fetch", "ue5_launch")), "flags")
    print("selftest", "pass" if ok else "fail", "key", key, "rate", RATE)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--prev", default=GENESIS)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not re.fullmatch(r"[0-9a-f]{16}", a.prev):
        print("--prev must be 16 hex", file=sys.stderr)
        return 2
    rows, trits, meta = audit(Path(a.root))
    body, pack, i2s, text, key = verdict(rows, trits, meta, a.prev)
    if a.json:
        print(text)
    else:
        for p, t, w in rows:
            print("%+d  %-50s %s" % (t, p, w))
        print("sparse %s mode %s clone --sparse would drop %d of %d"
              % (meta["sparse_on"], meta["mode"], len(meta["clone_sparse_drops"]), len(rows) - 1))
        print("zero_frac %.3f faithful %s release %s" % (body["zero_frac"], body["faithful"], body["release"]))
        print("key", key, "i2s", body["i2s_sha"])
    return 0 if body["faithful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
