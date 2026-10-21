#!/usr/bin/env python3
"""Live beta step 5 archive/attributes side: export-ignore and filter trits for backfill_check EXPECTED. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 5
("`python scripts/backfill_check.py` exits 0"), on its "A fresh clone"
premise. Quant Chief pair A, slot 2026-10-21 10:05 MT, non-quant-error side.
Trit Pack & I2_S side: the trit pack and I2_S key. Winsor & Quant Error
covers the quant-error bound side of the same item (JuniorQuant dbafb2e).

Gap it fills (not in BitNet-mlx 9244444 exit-code trits, JuniorLLM 829a789
coverage handshake, 5032022 .git/index trits, c93a5ec index handshake,
e82c972 HEAD-tree trits, e1ea106 step-order writer map, stocksnode 0eaa9a6
receipt ledger, JuniorQuant adfb204/dbafb2e quant-error gates): none of
those read gitattributes. A box that gets the repo as a GitHub "Download
ZIP" / codeload tarball / `git archive` (the usual path onto a Home or
Flagstaff box without git, or an OSai/FieldCore image build) gets the tree
MINUS every path whose attributes set export-ignore, including every path
under a DIRECTORY that is export-ignored (`docs export-ignore`, `docs/
export-ignore`, `rails/* export-ignore` all drop rails/linux/...). The index
and HEAD tree still hold the file, so every earlier check passes, yet
backfill_check.py on that unpacked tree exits 1 (or is absent itself).
Separately, filter=<driver> (git-lfs and custom smudge) and export-subst
mean the bytes a clone or archive gets depend on the receiving box.

This file evaluates gitattributes itself, offline, the way git's attr.c
does: working-tree .gitattributes stack from the repo root down to each
path's directory, later lines and deeper files win, `attr` / `-attr` /
`!attr` / `attr=value`, built-in `binary` macro and `[attr]` macros (root
file and .git/info/attributes only), quoted patterns, negative patterns
ignored as git does, basename match for slash-less patterns, anchored
wildmatch (`*`, `?`, `[...]`, `**/`, `/**`, `/**/`) otherwise, trailing
`/` patterns match directories only, and each ancestor directory checked as
a directory (archive skips the whole subtree). One balanced trit per
EXPECTED path, plus scripts/backfill_check.py itself:
    +1  no export-ignore on the path or any ancestor, no filter, no
        export-subst, working copy is not a git-lfs pointer
     0  filter=<x> set (smudge-dependent bytes), export-subst set (archive
        rewrites $Format$), or the working copy is an LFS pointer stub
    -1  export-ignore set on the path or an ancestor directory: an
        archive-based fresh tree has no such file and step 5 exits 1
plus one repo trit: .git/info/attributes (local only, never cloned) changes
no EXPECTED verdict (+1) or does (0). EXPECTED is read from
scripts/backfill_check.py with `ast` (literal list only; never imported or
run). No git binary, no subprocess, no other file of ours is imported.

Verdict: archive-faithful when no -1 and the 0 fraction is <= RATE (0.05);
release RATE when faithful, else 0.0 (held).

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

Usage: python3 scripts/backfill_archive_attr_i2s_key.py --selftest
       python3 scripts/backfill_archive_attr_i2s_key.py [--root DIR] [--prev HEX16] [--json]
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
ITEM = "BETA_TO_OS Live beta step 5 (archive export-ignore / filter attributes side)"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM loader (JTR1)", "BitNet.cpp I2_S block layout", "bitnet-mlx")
FLAGS = {"bind": "127.0.0.1", "mint": False, "socket": False,
         "layer_one": False, "fetch": False, "ue5_launch": False}
LFS_HEAD = b"version https://git-lfs.github.com/spec/v1"
BUILTIN_MACROS = {"binary": [("diff", False), ("merge", False), ("text", False)]}
UNSPEC = None  # value states: True set, False unset, None unspecified, str value


# ------------------------------------------------------------ wildmatch
def _wild_to_re(pat: str) -> str:
    out, i, n = [], 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**", i):
                pre = i == 0 or pat[i - 1] == "/"
                post = i + 2 == n or pat[i + 2] == "/"
                if pre and post:
                    if i + 2 == n:
                        out.append(".*")
                        i += 2
                    else:
                        out.append("(?:.*/)?")
                        i += 3
                    continue
                i += 2
                out.append("[^/]*")
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(pat[i]))
        elif c == "[":
            j = i + 1
            if j < n and pat[j] in "!^":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(c))
            else:
                body = pat[i + 1:j]
                neg = body[:1] in ("!", "^")
                if neg:
                    body = body[1:]
                body = body.replace("\\", "\\\\").replace("^", "\\^")
                out.append("(?!/)[" + ("^" if neg else "") + body + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def wildmatch(pat: str, s: str) -> bool:
    return re.fullmatch(_wild_to_re(pat), s, flags=re.S) is not None


# ------------------------------------------------------ attribute files
def _unquote(s: str):
    """C-style quoted pattern at start of s; returns (pattern, rest)."""
    out, i = [], 1
    esc = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "a": "\a", "b": "\b",
           "f": "\f", "r": "\r", "v": "\v"}
    while i < len(s):
        c = s[i]
        if c == '"':
            return "".join(out), s[i + 1:]
        if c == "\\" and i + 1 < len(s):
            d = s[i + 1]
            if d in esc:
                out.append(esc[d])
                i += 2
                continue
            m = re.match(r"[0-7]{3}", s[i + 1:])
            if m:
                out.append(chr(int(m.group(0), 8)))
                i += 4
                continue
        out.append(c)
        i += 1
    return None, ""


def _parse_states(tokens):
    st = []
    for t in tokens:
        if t.startswith("-"):
            st.append((t[1:], False))
        elif t.startswith("!"):
            st.append((t[1:], UNSPEC))
        elif "=" in t:
            k, v = t.split("=", 1)
            st.append((k, v))
        else:
            st.append((t, True))
    return [(k, v) for k, v in st if k]


def parse_attr_file(text: str, base: str, macros_ok: bool):
    """Return (rules, macros, notes). rule = (pattern, nodir, mustbedir, base, states)."""
    rules, macros, notes = [], {}, []
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip(" \t\r")
        if not line or line.startswith("#"):
            continue
        if line.startswith('"'):
            pat, rest = _unquote(line)
            if pat is None:
                notes.append("line %d bad quote" % ln)
                continue
        else:
            parts = line.split(None, 1)
            pat, rest = parts[0], (parts[1] if len(parts) > 1 else "")
        toks = rest.split()
        if pat.startswith("[attr]"):
            name = pat[6:]
            if macros_ok and name:
                macros[name] = _parse_states(toks)
            else:
                notes.append("line %d macro outside root ignored" % ln)
            continue
        if pat.startswith("!"):
            notes.append("line %d negative pattern ignored by git" % ln)
            continue
        must = pat.endswith("/") and len(pat) > 1
        if must:
            pat = pat.rstrip("/")
        nodir = "/" not in pat
        if pat.startswith("/"):
            pat = pat[1:]
        rules.append((pat, nodir, must, base, _parse_states(toks)))
    return rules, macros, notes


def rule_matches(rule, path: str, isdir: bool) -> bool:
    pat, nodir, must, base, _ = rule
    if must and not isdir:
        return False
    if base and not path.startswith(base):
        return False
    if nodir:
        return wildmatch(pat, path.rsplit("/", 1)[-1])
    return wildmatch(pat, path[len(base):])


class AttrView:
    def __init__(self, root: Path, use_info: bool):
        self.root, self.use_info = root, use_info
        self.cache, self.notes = {}, []
        self.macros = dict(BUILTIN_MACROS)
        self._load_root_macros()

    def _read(self, p: Path):
        try:
            if p.is_file() and not p.is_symlink():
                return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        return None

    def _load_root_macros(self):
        for p, on in ((self.root / ".gitattributes", True),
                      (self.root / ".git/info/attributes", self.use_info)):
            t = self._read(p) if on else None
            if t is not None:
                self.macros.update(parse_attr_file(t, "", True)[1])

    def _file_rules(self, d: str):
        if d not in self.cache:
            t = self._read(self.root / d / ".gitattributes")
            if t is None:
                self.cache[d] = []
            else:
                base = d + "/" if d else ""
                r, _, notes = parse_attr_file(t, base, d == "")
                self.notes += ["%s.gitattributes %s" % (base, x) for x in notes]
                self.cache[d] = r
        return self.cache[d]

    def stack(self, dirpart: str):
        parts = dirpart.split("/") if dirpart else []
        rules = list(self._file_rules(""))
        for k in range(1, len(parts) + 1):
            rules += self._file_rules("/".join(parts[:k]))
        if self.use_info:
            t = self._read(self.root / ".git/info/attributes")
            if t is not None:
                rules += parse_attr_file(t, "", True)[0]
        return rules

    def _apply(self, out, states, depth=0):
        for k, v in states:
            if v is True and k in self.macros and depth < 8:
                out[k] = True
                self._apply(out, self.macros[k], depth + 1)
            else:
                out[k] = v

    def attrs(self, path: str, isdir: bool = False) -> dict:
        dirpart = path.rsplit("/", 1)[0] if "/" in path else ""
        out: dict = {}
        for rule in self.stack(dirpart):
            if rule_matches(rule, path, isdir):
                self._apply(out, rule[4])
        return out


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


def is_lfs_pointer(p: Path) -> bool:
    try:
        if not p.is_file() or p.stat().st_size > 1024:
            return False
        return p.read_bytes().startswith(LFS_HEAD)
    except OSError:
        return False


def path_trit(view: AttrView, root: Path, path: str):
    parts = path.split("/")
    for k in range(1, len(parts)):
        d = "/".join(parts[:k])
        if view.attrs(d, isdir=True).get("export-ignore") is True:
            return -1, "ancestor dir %s/ export-ignore: archive drops it" % d
    a = view.attrs(path)
    if a.get("export-ignore") is True:
        return -1, "export-ignore: archive tree lacks it"
    if isinstance(a.get("filter"), str) or a.get("filter") is True:
        return 0, "filter=%s: smudge-dependent bytes" % a.get("filter")
    if a.get("export-subst") is True:
        return 0, "export-subst: archive rewrites content"
    if is_lfs_pointer(root / path):
        return 0, "working copy is an LFS pointer stub"
    return 1, "archive-faithful"


def audit(root: Path, expected=None):
    root = Path(root)
    exp = list(expected) if expected is not None else read_expected(root)
    clone = AttrView(root, use_info=False)
    local = AttrView(root, use_info=True)
    rows, trits, diverge = [], [], []
    for p in exp + [BACKFILL]:
        t, why = path_trit(clone, root, p)
        lt, _ = path_trit(local, root, p)
        if lt != t:
            diverge.append(p)
        rows.append((p, t, why))
        trits.append(t)
    info_trit = 0 if diverge else 1
    rows.append((".git/info/attributes", info_trit,
                 ("local-only attrs change " + ",".join(diverge)) if diverge else "no local divergence"))
    trits.append(info_trit)
    return rows, trits, {"info_trit": info_trit, "notes": sorted(set(clone.notes))}


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
    pad = (-len(codes)) % 128
    if not codes:
        pad = 128
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
        "info_trit": meta["info_trit"], "notes": meta["notes"],
        "zero_frac": round(zero_frac, 6), "faithful": faithful,
        "release": RATE if faithful else 0.0, "i2s_sha": hashlib.sha256(i2s).hexdigest()[:16],
        "ports": list(PORTS), "engines": list(ENGINES), "flags": FLAGS,
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return body, pack, i2s, text, close_key(prev, pack, text)


# ------------------------------------------------------------ selftest
EXP = ["docs/A.md", "docs/B.md", "STATE.md", "rails/linux/x.sh"]


def _mk(base: Path, attrs: dict, extra: dict = None) -> Path:
    files = {p: b"# real content\n" * 4 for p in EXP}
    files[BACKFILL] = ("EXPECTED = [\n" + "".join('    "%s",\n' % p for p in EXP) + "]\n").encode()
    files.update(extra or {})
    for p, data in list(files.items()) + [(k, v.encode()) for k, v in attrs.items()]:
        f = base / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    return base


def selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL", msg)

    # wildmatch semantics
    check(wildmatch("docs/*", "docs/A.md") and not wildmatch("docs/*", "docs/x/A.md"), "* no slash")
    check(wildmatch("rails/**/x.sh", "rails/linux/x.sh") and wildmatch("rails/**/x.sh", "rails/x.sh"), "**/")
    check(wildmatch("**/x.sh", "a/b/x.sh") and wildmatch("docs/**", "docs/a/b"), "leading/trailing **")
    check(wildmatch("[A-C].md", "B.md") and not wildmatch("[!A-C].md", "B.md"), "class")
    check(wildmatch("?.md", "A.md") and not wildmatch("a?b", "a/b"), "? no slash")

    def run(attrs, extra=None):
        with tempfile.TemporaryDirectory() as td:
            r = _mk(Path(td), attrs, extra)
            return audit(r)

    # 1 no attributes -> all +1 faithful
    rows, trits, meta = run({})
    check(trits == [1] * 6, "plain all +1 %r" % trits)
    body, pack, i2s, text, key = verdict(rows, trits, meta)
    check(body["faithful"] and body["release"] == RATE, "plain faithful")
    # 2 file pattern export-ignore
    _, t, _ = run({".gitattributes": "docs/** export-ignore\n"})
    check(t[:2] == [-1, -1] and t[2:] == [1, 1, 1, 1], "docs/** -1 %r" % t)
    # 3 directory export-ignore: bare name, trailing slash, rails/* hits rails/linux/
    _, t, _ = run({".gitattributes": "docs export-ignore\n"})
    check(t[:2] == [-1, -1], "dir name -1 %r" % t)
    r3, t, _ = run({".gitattributes": "docs/ export-ignore\nrails/* export-ignore\n"})
    check(t[:2] == [-1, -1] and t[3] == -1 and "ancestor" in r3[3][2], "dir/ and rails/* -1 %r" % t)
    # trailing slash never matches a file
    _, t, _ = run({".gitattributes": "STATE.md/ export-ignore\n"})
    check(t[2] == 1, "file/ pattern ignored for file")
    # 4 anchored vs basename
    _, t, _ = run({".gitattributes": "/x.sh export-ignore\n"})
    check(t[3] == 1, "anchored /x.sh only at root")
    _, t, _ = run({".gitattributes": "x.sh export-ignore\n"})
    check(t[3] == -1, "basename x.sh any depth")
    # 5 filter / lfs / export-subst -> 0, verdict held
    rows5, t, m5 = run({".gitattributes": "*.md filter=lfs diff=lfs merge=lfs -text\n"})
    check(t[:3] == [0, 0, 0] and t[3] == 1, "filter=lfs 0 %r" % t)
    check(not verdict(rows5, t, m5)[0]["faithful"], "filter held")
    _, t, _ = run({".gitattributes": "STATE.md export-subst\n"})
    check(t[2] == 0, "export-subst 0")
    _, t, _ = run({}, {"docs/B.md": LFS_HEAD + b"\noid sha256:" + b"0" * 64 + b"\nsize 9\n"})
    check(t[1] == 0, "lfs pointer stub 0")
    # 6 deeper file and later line override; ! and - states
    _, t, _ = run({".gitattributes": "*.md export-ignore\n",
                   "docs/.gitattributes": "A.md !export-ignore\nB.md -export-ignore\n"})
    check(t[0] == 1 and t[1] == 1 and t[2] == -1, "deeper override %r" % t)
    _, t, _ = run({".gitattributes": "STATE.md export-ignore\nSTATE.md -export-ignore\n"})
    check(t[2] == 1, "later line wins")
    # negative pattern ignored (git rejects it)
    _, t, m = run({".gitattributes": "*.md export-ignore\n!STATE.md export-ignore\n"})
    check(t[2] == -1 and any("negative" in x for x in m["notes"]), "negative pattern ignored")
    # 7 macros: root ok, subdir ignored, builtin binary harmless
    _, t, _ = run({".gitattributes": "[attr]ship export-ignore -diff\nSTATE.md ship\n*.sh binary\n"})
    check(t[2] == -1 and t[3] == 1, "root macro %r" % t)
    _, t, m = run({"docs/.gitattributes": "[attr]ship export-ignore\nA.md ship\n"})
    check(t[0] == 1 and any("macro" in x for x in m["notes"]), "subdir macro ignored")
    # quoted pattern
    _, t, _ = run({".gitattributes": '"STATE.md" export-ignore\n'})
    check(t[2] == -1, "quoted pattern")
    # 8 backfill_check itself export-ignored
    _, t, _ = run({".gitattributes": "scripts/ export-ignore\n"})
    check(t[4] == -1, "backfill script dropped")
    # 9 local-only .git/info/attributes divergence
    r9, t, m9 = run({".git/info/attributes": "STATE.md export-ignore\n"})
    check(t[2] == 1 and t[5] == 0 and "STATE.md" in r9[5][2], "info divergence %r" % t)
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
        for n in meta["notes"]:
            print("note", n)
        print("info %+d zero_frac %.3f faithful %s release %s"
              % (meta["info_trit"], body["zero_frac"], body["faithful"], body["release"]))
        print("key", key, "i2s", body["i2s_sha"])
    return 0 if body["faithful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
