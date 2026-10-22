#!/usr/bin/env python3
"""Live beta step 5 interpreter side: `python` resolution and min-version trits for backfill_check. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 5
("`python scripts/backfill_check.py` exits 0"), on its "A fresh clone"
premise. Quant Chief pair A, slot 2026-10-22 05:05 MT, non-quant-error side.
Trit Pack & I2_S side: the trit pack and I2_S key. Winsor & Quant Error
covers the quant-error bound side of the same item (JuniorQuant dbafb2e).

Gap it fills (not in BitNet-mlx 9244444 exit-code trits, JuniorLLM 829a789
coverage handshake, 5032022 .git/index trits, c93a5ec index handshake,
e82c972 HEAD-tree trits, e1ea106 step-order writer map, 5b35e4d archive
attributes, 044a549 sparse-checkout, stocksnode 0eaa9a6 receipt ledger,
JuniorQuant adfb204/dbafb2e quant-error gates): every earlier check is about
the TREE and runs under whatever interpreter the checker itself was launched
with. None of them asks whether the literal command in the doc can start.
Step 5 says `python`, not `python3`. macOS 12.3+ ships no `python` (Home and
Flagstaff Macs), Debian/Ubuntu images without python-is-python3 have none
(OSai and FieldCore builds), and an old box may still resolve `python` to
2.7, where backfill_check.py's `from __future__ import annotations` is a
SyntaxError. On each of those the tree is perfect and step 5 still fails:
command not found (exit 127) or exit 1 before main() runs.

This file reads the step-5 command out of BETA_TO_OS.md, derives the minimum
Python that scripts/backfill_check.py needs from its AST (never imported or
run), walks PATH by hand, and reads each candidate interpreter's version
OFFLINE from its resolved file name (python3.12), a pyvenv.cfg, or a single
lib/pythonX.Y beside it. No interpreter is executed: no subprocess, no
network, no git. Trits (balanced, one per row):
  doc_cmd       +1 step-5 line found and names an interpreter, -1 not found
  parse         +1 backfill_check.py parses, -1 SyntaxError
  encoding      +1 ASCII, 0 UTF-8 non-ASCII (no cookie), -1 undecodable
  path_env      +1 PATH set with only absolute entries, 0 relative/empty
                entry (cwd-dependent), -1 PATH unset or empty
  doc_found     +1 the doc's interpreter resolves on PATH, -1 absent
  doc_major3    +1 it is Python 3, -1 Python 2, 0 version unreadable/absent
  doc_min       +1 version >= derived minimum, -1 below, 0 unreadable/absent
  doc_venv      +1 not inside a venv, 0 resolves into a venv (dev-only)
  py3_found     +1 `python3` resolves on PATH, -1 absent
  py3_min       +1 python3 version >= minimum, -1 below, 0 unreadable/absent
  shebang       +1 `#!` interpreter resolves and meets minimum, 0 no
                shebang or unreadable version, -1 names a missing or too-old
                interpreter
  agree         +1 doc interpreter equals shebang interpreter, 0 differ
Verdict: launch-faithful when no -1 and the 0 fraction is <= RATE (0.05);
release RATE when faithful, else 0.0 (held). Also reported, outside the
trits: the derived minimum and which feature set it.

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

Usage: python3 scripts/backfill_interpreter_path_i2s_key.py --selftest
       python3 scripts/backfill_interpreter_path_i2s_key.py [--root DIR] [--path PATHSTR] [--prev HEX16] [--json]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

RATE = 0.05
GENESIS = "0" * 16
BACKFILL = "scripts/backfill_check.py"
DOC = "docs/BETA_TO_OS.md"
ITEM = "BETA_TO_OS Live beta step 5 (interpreter resolution / min-version side)"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM loader (JTR1)", "BitNet.cpp I2_S block layout", "bitnet-mlx")
FLAGS = {"bind": "127.0.0.1", "mint": False, "socket": False,
         "layer_one": False, "fetch": False, "ue5_launch": False}
ROWS = ("doc_cmd", "parse", "encoding", "path_env", "doc_found", "doc_major3",
        "doc_min", "doc_venv", "py3_found", "py3_min", "shebang", "agree")
STEP_RE = re.compile(r"^\s*5\.\s.*?`\s*((?:/\S+/)?python[0-9.]*)\s+scripts/backfill_check\.py", re.M)
NAME_RE = re.compile(r"^python(\d)(?:\.(\d+))?$")


# ------------------------------------------------------ minimum version
def min_version(tree: ast.AST):
    """Smallest (major, minor) whose grammar/runtime the module needs, and why."""
    need = [((3, 0), "py3 baseline")]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(a.name == "annotations" for a in node.names):
                need.append(((3, 7), "from __future__ import annotations"))
        elif isinstance(node, ast.JoinedStr):
            need.append(((3, 6), "f-string"))
        elif isinstance(node, ast.NamedExpr):
            need.append(((3, 8), "walrus :="))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node.args.posonlyargs:
            need.append(((3, 8), "positional-only /"))
        elif hasattr(ast, "Match") and isinstance(node, ast.Match):
            need.append(((3, 10), "match statement"))
        elif hasattr(ast, "TryStar") and isinstance(node, ast.TryStar):
            need.append(((3, 11), "except*"))
        elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
            need.append(((3, 12), "type alias statement"))
        elif isinstance(node, ast.Attribute) and node.attr in ("removeprefix", "removesuffix"):
            need.append(((3, 9), "str." + node.attr))
    return max(need)


# ------------------------------------------------------ PATH, offline
def which(name: str, path: str):
    for d in path.split(os.pathsep):
        if not d:
            continue
        cand = Path(d) / name
        if cand.is_file() and os.access(str(cand), os.X_OK):
            return cand
    return None


def read_version(exe: Path):
    """(major, minor or None, venv) from file names and side files; never executes exe."""
    real = Path(os.path.realpath(str(exe)))
    venv = False
    ver = None
    for p in (exe, real):
        cfg = p.parent.parent / "pyvenv.cfg"
        if cfg.is_file():
            venv = True
            m = re.search(r"^\s*version(?:_info)?\s*=\s*(\d+)\.(\d+)", cfg.read_text("utf-8", "replace"), re.M)
            if m and ver is None:
                ver = (int(m.group(1)), int(m.group(2)))
    if ver is None:
        m = NAME_RE.match(real.name)
        if m and m.group(2) is not None:
            ver = (int(m.group(1)), int(m.group(2)))
    if ver is None:
        libs = sorted(q.name for q in (real.parent.parent / "lib").glob("python[0-9]*.[0-9]*") if q.is_dir())
        if len(libs) == 1:
            mm = re.match(r"python(\d+)\.(\d+)$", libs[0])
            if mm:
                ver = (int(mm.group(1)), int(mm.group(2)))
    if ver is None:
        m = NAME_RE.match(real.name) or NAME_RE.match(exe.name)
        if m:
            return int(m.group(1)), None, venv
        return None, None, venv
    return ver[0], ver[1], venv


def ver_trit(info, need):
    if info is None:
        return 0, "absent"
    major, minor, _ = info
    if major is None:
        return 0, "version unreadable"
    if minor is None:
        return (0, "py%d minor unreadable" % major) if major >= need[0] else (-1, "py%d" % major)
    ok = (major, minor) >= need
    return (1 if ok else -1), "%d.%d %s %d.%d" % (major, minor, ">=" if ok else "<", need[0], need[1])


# ------------------------------------------------------------- audit
def audit(root: Path, path=None):
    path = os.environ.get("PATH", "") if path is None else path
    rows = []

    doc = root / DOC
    m = STEP_RE.search(doc.read_text("utf-8", "replace")) if doc.is_file() else None
    interp = m.group(1) if m else None
    rows.append(("doc_cmd", 1 if interp else -1, interp or "step 5 line not found"))

    src_b = (root / BACKFILL).read_bytes() if (root / BACKFILL).is_file() else None
    tree, src = None, ""
    if src_b is None:
        rows.append(("parse", -1, "backfill_check.py missing"))
    else:
        src = src_b.decode("utf-8", "replace")
        try:
            tree = ast.parse(src_b)
            rows.append(("parse", 1, "ok"))
        except (SyntaxError, ValueError) as e:
            rows.append(("parse", -1, "SyntaxError line %s" % getattr(e, "lineno", "?")))
    if src_b is None:
        rows.append(("encoding", -1, "missing"))
    else:
        try:
            src_b.decode("ascii")
            rows.append(("encoding", 1, "ascii"))
        except UnicodeDecodeError:
            try:
                src_b.decode("utf-8")
                cookie = re.search(rb"^[ \t\f]*#.*?coding[:=]", b"\n".join(src_b.splitlines()[:2]), re.M)
                rows.append(("encoding", 1 if cookie else 0, "utf-8 non-ascii" + (" with cookie" if cookie else "")))
            except UnicodeDecodeError:
                rows.append(("encoding", -1, "not utf-8"))
    need, why = min_version(tree) if tree is not None else ((3, 0), "unparsed")

    entries = path.split(os.pathsep) if path else []
    if not path:
        rows.append(("path_env", -1, "PATH empty"))
    else:
        rel = [e for e in entries if not e or not os.path.isabs(e)]
        rows.append(("path_env", 0 if rel else 1, "%d entries, %d relative" % (len(entries), len(rel))))

    def resolve(name):
        if not name:
            return None, None
        exe = Path(name) if "/" in name else which(name, path)
        if exe is None or not exe.is_file():
            return None, None
        return exe, read_version(exe)

    dname = interp.rsplit("/", 1)[-1] if interp else None
    dexe, dinfo = resolve(interp)
    rows.append(("doc_found", 1 if dexe else -1, "%s %s" % (dname, "found" if dexe else "not on PATH")))
    if dinfo is None or dinfo[0] is None:
        rows.append(("doc_major3", 0, "absent" if dinfo is None else "unreadable"))
    else:
        rows.append(("doc_major3", 1 if dinfo[0] == 3 else -1, "major %d" % dinfo[0]))
    t, w = ver_trit(dinfo, need)
    rows.append(("doc_min", t, w))
    rows.append(("doc_venv", 0 if (dinfo and dinfo[2]) else 1, "venv" if (dinfo and dinfo[2]) else "system or absent"))

    pexe, pinfo = resolve("python3")
    rows.append(("py3_found", 1 if pexe else -1, "python3 " + ("found" if pexe else "not on PATH")))
    t, w = ver_trit(pinfo, need)
    rows.append(("py3_min", t, w))

    first = src.splitlines()[0] if src else ""
    sb = None
    if first.startswith("#!"):
        parts = first[2:].split()
        if parts:
            sb = parts[1] if parts[0].rsplit("/", 1)[-1] == "env" and len(parts) > 1 else parts[0]
    if sb is None:
        rows.append(("shebang", 0, "no shebang"))
    else:
        sexe, sinfo = resolve(sb)
        sname = sb.rsplit("/", 1)[-1]
        if sexe is None:
            rows.append(("shebang", -1, sname + " not on PATH"))
        else:
            t, w = ver_trit(sinfo, need)
            rows.append(("shebang", t, sname + " " + w))
    sname = sb.rsplit("/", 1)[-1] if sb else None
    rows.append(("agree", 1 if (dname and dname == sname) else 0, "%s vs %s" % (dname, sname)))

    assert tuple(r[0] for r in rows) == ROWS
    trits = [r[1] for r in rows]
    meta = {"min_version": "%d.%d" % need, "min_reason": why, "doc_interpreter": dname,
            "shebang_interpreter": sname}
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
DOC_OK = "## Live beta\n\n4. something\n5. `python scripts/backfill_check.py` exits 0\n"
BF_OK = '#!/usr/bin/env python\n"""x"""\nfrom __future__ import annotations\nprint("ok")\n'


def _exe(d: Path, name: str, target=None):
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if target:
        p.symlink_to(target)
    else:
        p.write_bytes(b"\x7fELF fake, never run\n")
        p.chmod(0o755)
    return p


def _repo(base: Path, doc=DOC_OK, bf=BF_OK):
    root = base / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / DOC).write_text(doc, "utf-8")
    (root / BACKFILL).write_bytes(bf.encode("utf-8") if isinstance(bf, str) else bf)
    return root


def selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL", msg)

    def tr(rows, name):
        return dict((r[0], r[1]) for r in rows)[name]

    key = None
    with tempfile.TemporaryDirectory() as td:
        b = Path(td)
        # healthy box: python -> python3.12, python3 -> python3.12
        bin1 = b / "usr1" / "bin"
        _exe(bin1, "python3.12")
        _exe(bin1, "python3", "python3.12")
        _exe(bin1, "python", "python3.12")
        root = _repo(b / "r1")
        rows, trits, meta = audit(root, str(bin1))
        body, pack, i2s, text, key = verdict(rows, trits, meta)
        check(trits == [1] * 12, "healthy all +1 %r" % trits)
        check(body["faithful"] and body["release"] == RATE, "healthy faithful")
        check(meta["min_version"] == "3.7", "min 3.7 from future annotations")

        # macOS / Debian: no `python`, only python3
        bin2 = b / "usr2" / "bin"
        _exe(bin2, "python3.13")
        _exe(bin2, "python3", "python3.13")
        rows, trits, meta = audit(root, str(bin2))
        check(tr(rows, "doc_found") == -1 and tr(rows, "py3_found") == 1, "no python held")
        check(tr(rows, "shebang") == -1, "shebang python missing")
        check(not verdict(rows, trits, meta)[0]["faithful"], "no python not faithful")

        # python -> 2.7
        bin3 = b / "usr3" / "bin"
        _exe(bin3, "python2.7")
        _exe(bin3, "python", "python2.7")
        _exe(bin3, "python3.11")
        _exe(bin3, "python3", "python3.11")
        rows, trits, meta = audit(root, str(bin3))
        check(tr(rows, "doc_major3") == -1 and tr(rows, "doc_min") == -1, "py2 rejected")

        # python -> 3.6 below 3.7
        bin4 = b / "usr4" / "bin"
        _exe(bin4, "python3.6")
        _exe(bin4, "python", "python3.6")
        _exe(bin4, "python3", "python3.6")
        rows, trits, meta = audit(root, str(bin4))
        check(tr(rows, "doc_major3") == 1 and tr(rows, "doc_min") == -1 and tr(rows, "py3_min") == -1, "3.6 too old")

        # walrus raises min to 3.8; 3.7 box fails, unreadable minor is 0
        root5 = _repo(b / "r5", bf=BF_OK + "if (x := 1):\n    pass\n")
        bin5 = b / "usr5" / "bin"
        _exe(bin5, "python3.7")
        _exe(bin5, "python", "python3.7")
        _exe(bin5, "python3")
        rows, trits, meta = audit(root5, str(bin5))
        check(meta["min_version"] == "3.8" and tr(rows, "doc_min") == -1, "walrus 3.8")
        check(tr(rows, "py3_min") == 0, "bare python3 minor unreadable -> 0")

        # lib/pythonX.Y beside a bare python3
        lib6 = b / "usr6"
        _exe(lib6 / "bin", "python3")
        (lib6 / "lib" / "python3.10").mkdir(parents=True)
        rows, trits, meta = audit(root, str(lib6 / "bin"))
        check(tr(rows, "py3_min") == 1, "lib dir version read")

        # venv shadow
        v = b / "venv"
        _exe(v / "bin", "python")
        (v / "pyvenv.cfg").write_text("home = /x\nversion = 3.12.4\n", "utf-8")
        rows, trits, meta = audit(root, str(v / "bin") + os.pathsep + str(bin1))
        check(tr(rows, "doc_venv") == 0 and tr(rows, "doc_min") == 1, "venv shadow 0")

        # relative PATH entry, syntax error, non-ascii, missing step line, python3 shebang
        rows, trits, meta = audit(root, "." + os.pathsep + str(bin1))
        check(tr(rows, "path_env") == 0, "relative PATH 0")
        rows, trits, meta = audit(root, "")
        check(tr(rows, "path_env") == -1 and tr(rows, "doc_found") == -1, "empty PATH")
        r7 = _repo(b / "r7", bf="print 'py2'\n")
        rows, trits, meta = audit(r7, str(bin1))
        check(tr(rows, "parse") == -1 and tr(rows, "shebang") == 0, "syntax error, no shebang")
        r8 = _repo(b / "r8", bf=BF_OK + "# caf\u00e9\n")
        rows, trits, meta = audit(r8, str(bin1))
        check(tr(rows, "encoding") == 0, "non-ascii 0")
        r9 = _repo(b / "r9", doc="5. run the backfill\n", bf=BF_OK.replace("env python\n", "env python3\n"))
        rows, trits, meta = audit(r9, str(bin1))
        check(tr(rows, "doc_cmd") == -1 and tr(rows, "agree") == 0 and tr(rows, "shebang") == 1, "no step line")
        r10 = _repo(b / "r10", bf=b"\xff\xfe junk\n")
        rows, trits, meta = audit(r10, str(bin1))
        check(tr(rows, "encoding") == -1, "undecodable -1")

    # JTR1 / I2_S / close rule
    t = [1, 0, -1, 1, 1, -1, 0]
    g, back = jtr1_unpack(jtr1_pack(t))
    check(back == t and abs(g - 5 / 7) < 1e-6, "jtr1 roundtrip")
    try:
        jtr1_unpack(b"JTR1" + struct.pack("<fI", 1.0, 1) + b"\xc0")
        check(False, "code 11 accepted")
    except ValueError:
        pass
    check(len(i2s_block([1] * 12)) == 32 and len(i2s_block([1] * 129)) == 64, "i2s 128-block")
    pk = jtr1_pack([1] * 12)
    check(close_key(GENESIS, pk, "x") == hashlib.sha256(GENESIS.encode() + pk + b"x").hexdigest()[:16], "close rule")
    check(RATE == 0.05 and FLAGS["bind"] == "127.0.0.1" and not any(
        FLAGS[k] for k in ("mint", "socket", "layer_one", "fetch", "ue5_launch")), "flags")
    print("selftest", "pass" if ok else "fail", "key", key, "rate", RATE)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--path", default=None, help="PATH string to audit (default: $PATH)")
    ap.add_argument("--prev", default=GENESIS)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not re.fullmatch(r"[0-9a-f]{16}", a.prev):
        print("--prev must be 16 hex", file=sys.stderr)
        return 2
    rows, trits, meta = audit(Path(a.root), a.path)
    body, pack, i2s, text, key = verdict(rows, trits, meta, a.prev)
    if a.json:
        print(text)
    else:
        for p, t, w in rows:
            print("%+d  %-11s %s" % (t, p, w))
        print("min python %s (%s)" % (meta["min_version"], meta["min_reason"]))
        print("zero_frac %.3f faithful %s release %s" % (body["zero_frac"], body["faithful"], body["release"]))
        print("key", key, "i2s", body["i2s_sha"])
    return 0 if body["faithful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
