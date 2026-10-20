#!/usr/bin/env python3
"""Live beta step 5 committed-tree side: HEAD-tree trits for backfill_check EXPECTED. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 5
("`python scripts/backfill_check.py` exits 0"), on its "A fresh clone"
premise; docs/COMPILED_BACKLOG.md "A - live beta suite".
Quant Chief pair A, slot 2026-10-20 05:05 MT, non-quant-error side.
Trit Pack & I2_S side: the trit pack and I2_S key. Winsor & Quant Error
covers the quant-error bound side of the same item in a different file.

Gap it fills (not in BitNet-mlx 9244444 exit-code trits, JuniorLLM 829a789
coverage handshake, JuniorLLM 5032022 .git/index trits or c93a5ec index
handshake): those read the working tree and the git INDEX. A fresh clone
gets a COMMIT's tree, not the index. A path that is `git add`-ed but never
committed is stage 0 in the index with a matching blob, so every earlier
check passes, yet a clone of HEAD exits 1. A commit that was never pushed
passes locally and is absent from the clone too. This file resolves HEAD
(loose ref, packed-refs, gitdir file, commondir), reads the commit and its
trees straight from the object store (loose zlib objects and pack v2 + idx
v2, with OFS_DELTA and REF_DELTA applied; every object's sha1 verified; no
git binary, no subprocess, no remote, no fetch) and gives one balanced trit
per EXPECTED path:
    +1  in the HEAD tree as a regular blob (100644/100755) whose id equals
        the working copy's blob id sha1("blob <len>\\0" + bytes)
     0  in the HEAD tree but local copy differs or is missing (the clone
        still gets the file), symlink 120000, or an object is unreadable
    -1  absent from the HEAD tree (staged-only or untracked: a clone of HEAD
        exits 1), a directory, or a gitlink 160000 (empty in a clone)
plus two repo trits: HEAD resolves to a readable commit+tree (+1/-1), and
HEAD equals the LOCAL remote-tracking ref refs/remotes/origin/<branch>
(+1 equal, 0 differs or absent: unpushed or stale; never fetched to check).
EXPECTED is read from scripts/backfill_check.py with `ast` (literal list
only; never imported or run).

Verdict: clone-faithful when no -1 and the 0 fraction is <= RATE (0.05);
release RATE when faithful, else 0.0 (held).

JTR1 (not GGUF): b"JTR1" | gamma <f32 LE, absmean> | n <u32 LE> | 2-bit
payload MSB first (0 -> 00, +1 -> 01, -1 -> 10; 11 rejected).
I2_S block: code = trit + 1, four per byte MSB first, padded with code 1
(zero) to a 128-trit block, as BitNet I2_S stores ternary.
Close rule: sha256(prev || pack || text)[:16]; prev 16-hex ASCII, pack =
JTR1 bytes, text = canonical verdict JSON (sorted keys, compact; includes
i2s_sha). Rate 0.05. Compatible engines: JuniorLLM, BitNet-mlx, i2sd
userspace, JuniorOS overlay. Ports touched: Home, OSai, FieldCore,
Flagstaff, JuniorLLM. Bind 127.0.0.1 only. mint/socket/layer_one/fetch/
ue5_launch false. No model pull, no GGUF download, no hardware calls, no
subprocess, no network.

Usage: python3 scripts/backfill_head_tree_i2s_key.py --selftest
       python3 scripts/backfill_head_tree_i2s_key.py [--root DIR] [--prev HEX16] [--json]
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
import zlib
from pathlib import Path

RATE = 0.05
GENESIS = "0" * 16
BACKFILL = "scripts/backfill_check.py"
ITEM = "BETA_TO_OS Live beta step 5 (committed HEAD tree side)"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM", "BitNet-mlx", "i2sd userspace", "JuniorOS overlay")
FLAGS = {"bind": "127.0.0.1", "mint": False, "socket": False,
         "layer_one": False, "fetch": False, "ue5_launch": False}
HEX40 = re.compile(r"[0-9a-f]{40}")
PACK_TYPES = {1: "commit", 2: "tree", 3: "blob", 4: "tag"}
REG_MODES = ("100644", "100755")


# ---------------------------------------------------------------- git dir
def git_dirs(root: Path):
    """Return (gitdir, commondir) or (None, None)."""
    g = root / ".git"
    if g.is_file():
        t = g.read_text(encoding="utf-8", errors="replace").strip()
        if not t.startswith("gitdir:"):
            return None, None
        p = Path(t[7:].strip())
        g = p if p.is_absolute() else (root / p).resolve()
    if not g.is_dir():
        return None, None
    common = g
    cd = g / "commondir"
    if cd.is_file():
        p = Path(cd.read_text().strip())
        common = p if p.is_absolute() else (g / p).resolve()
    return g, common


def read_ref(gd: Path, common: Path, name: str, depth: int = 0):
    """Resolve a ref name to (sha, final_ref_name) using loose refs then packed-refs."""
    if depth > 5:
        return None, name
    val = None
    for base in (gd, common):
        p = base / name
        if p.is_file():
            val = p.read_text(encoding="utf-8", errors="replace").strip()
            break
    if val is None:
        pr = common / "packed-refs"
        if pr.is_file():
            for line in pr.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line or line[0] in "#^":
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == name:
                    val = parts[0]
                    break
    if val is None:
        return None, name
    if val.startswith("ref:"):
        return read_ref(gd, common, val[4:].strip(), depth + 1)
    return (val if HEX40.fullmatch(val) else None), name


# ----------------------------------------------------------- object store
def apply_delta(src: bytes, delta: bytes) -> bytes:
    i = 0

    def varint() -> int:
        nonlocal i
        r = s = 0
        while True:
            c = delta[i]
            i += 1
            r |= (c & 0x7F) << s
            s += 7
            if not c & 0x80:
                return r

    if varint() != len(src):
        raise ValueError("delta base size mismatch")
    tsize = varint()
    out = bytearray()
    while i < len(delta):
        c = delta[i]
        i += 1
        if c & 0x80:
            off = size = 0
            for k in range(4):
                if c & (1 << k):
                    off |= delta[i] << (8 * k)
                    i += 1
            for k in range(3):
                if c & (1 << (4 + k)):
                    size |= delta[i] << (8 * k)
                    i += 1
            size = size or 0x10000
            out += src[off:off + size]
        elif c:
            out += delta[i:i + c]
            i += c
        else:
            raise ValueError("delta opcode 0")
    if len(out) != tsize:
        raise ValueError("delta target size mismatch")
    return bytes(out)


def obj_id(typ: str, data: bytes) -> str:
    return hashlib.sha1(b"%s %d\0" % (typ.encode(), len(data)) + data).hexdigest()


class Store:
    def __init__(self, common: Path):
        self.objdir = common / "objects"
        self.packs = []
        pd = self.objdir / "pack"
        if pd.is_dir():
            for idx in sorted(pd.glob("*.idx")):
                pk = idx.with_suffix(".pack")
                if pk.is_file():
                    self.packs.append([idx.read_bytes(), pk, None])

    @staticmethod
    def _idx_find(idx: bytes, sha20: bytes):
        if idx[:4] != b"\377tOc" or struct.unpack(">I", idx[4:8])[0] != 2:
            return None
        fan = struct.unpack(">256I", idx[8:1032])
        n = fan[255]
        lo = fan[sha20[0] - 1] if sha20[0] else 0
        hi = fan[sha20[0]]
        names = 1032
        while lo < hi:
            mid = (lo + hi) // 2
            s = idx[names + mid * 20:names + mid * 20 + 20]
            if s == sha20:
                offs = names + n * 24
                o = struct.unpack(">I", idx[offs + mid * 4:offs + mid * 4 + 4])[0]
                if o & 0x80000000:
                    lg = offs + n * 4 + (o & 0x7FFFFFFF) * 8
                    o = struct.unpack(">Q", idx[lg:lg + 8])[0]
                return o
            if s < sha20:
                lo = mid + 1
            else:
                hi = mid
        return None

    def _unpack_at(self, pack: bytes, off: int, depth: int = 0):
        if depth > 64:
            raise ValueError("delta chain too deep")
        start = off
        c = pack[off]
        off += 1
        typ = (c >> 4) & 7
        while c & 0x80:
            c = pack[off]
            off += 1
        if typ == 6:
            c = pack[off]
            off += 1
            back = c & 0x7F
            while c & 0x80:
                c = pack[off]
                off += 1
                back = ((back + 1) << 7) | (c & 0x7F)
            btyp, bdata = self._unpack_at(pack, start - back, depth + 1)
            delta = zlib.decompressobj().decompress(memoryview(pack)[off:])
            return btyp, apply_delta(bdata, delta)
        if typ == 7:
            base = pack[off:off + 20].hex()
            off += 20
            got = self.read(base, depth + 1)
            if got is None:
                raise ValueError("ref-delta base missing")
            delta = zlib.decompressobj().decompress(memoryview(pack)[off:])
            return got[0], apply_delta(got[1], delta)
        if typ not in PACK_TYPES:
            raise ValueError("bad pack object type %d" % typ)
        return PACK_TYPES[typ], zlib.decompressobj().decompress(memoryview(pack)[off:])

    def read(self, hexsha: str, depth: int = 0):
        """Return (type, data) with sha1 verified, or None."""
        loose = self.objdir / hexsha[:2] / hexsha[2:]
        try:
            if loose.is_file():
                raw = zlib.decompress(loose.read_bytes())
                hdr, _, body = raw.partition(b"\0")
                typ, size = hdr.decode().split()
                if int(size) == len(body) and hashlib.sha1(raw).hexdigest() == hexsha:
                    return typ, body
                return None
            sha20 = bytes.fromhex(hexsha)
            for ent in self.packs:
                off = self._idx_find(ent[0], sha20)
                if off is None:
                    continue
                if ent[2] is None:
                    ent[2] = ent[1].read_bytes()
                    if ent[2][:4] != b"PACK":
                        return None
                typ, data = self._unpack_at(ent[2], off, depth)
                return (typ, data) if obj_id(typ, data) == hexsha else None
        except (ValueError, zlib.error, IndexError, OSError, UnicodeDecodeError):
            return None
        return None


def tree_entries(data: bytes) -> dict:
    out, i = {}, 0
    while i < len(data):
        sp = data.index(b" ", i)
        nul = data.index(b"\0", sp)
        name = data[sp + 1:nul].decode("utf-8", "surrogateescape")
        out[name] = (data[i:sp].decode(), data[nul + 1:nul + 21].hex())
        i = nul + 21
    return out


def lookup(store: Store, tree_hex: str, path: str):
    """Return (mode, sha, reason)."""
    parts = path.split("/")
    cur = tree_hex
    for k, part in enumerate(parts):
        got = store.read(cur)
        if got is None or got[0] != "tree":
            return None, None, "tree unreadable"
        ent = tree_entries(got[1]).get(part)
        if ent is None:
            return None, None, "absent from HEAD tree"
        if k < len(parts) - 1:
            if ent[0] != "40000":
                return None, None, "parent not a directory in HEAD"
            cur = ent[1]
        else:
            return ent[0], ent[1], "ok"
    return None, None, "empty path"


# ------------------------------------------------------------- EXPECTED
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


# --------------------------------------------------------------- audit
def audit(root: Path, expected=None):
    root = Path(root)
    if expected is None:
        expected = read_expected(root)
    rows, head_t, up_t, head_sha, branch = [], -1, 0, None, None
    gd, common = git_dirs(root)
    tree = None
    store = None
    if gd is not None:
        head_sha, ref = read_ref(gd, common, "HEAD")
        branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else None
        store = Store(common)
        if head_sha:
            got = store.read(head_sha)
            if got and got[0] == "commit":
                m = re.search(rb"^tree ([0-9a-f]{40})$", got[1], re.M)
                if m and (store.read(m.group(1).decode()) or ("",))[0] == "tree":
                    tree = m.group(1).decode()
                    head_t = 1
        track = "refs/remotes/origin/" + branch if branch else "refs/remotes/origin/HEAD"
        up_sha, _ = read_ref(gd, common, track)
        up_t = 1 if (head_sha and up_sha == head_sha) else 0
    for p in expected:
        if tree is None:
            rows.append((p, 0, "HEAD unresolved"))
            continue
        mode, sha, why = lookup(store, tree, p)
        if mode is None:
            rows.append((p, 0 if why == "tree unreadable" else -1, why))
        elif mode in ("40000", "160000"):
            rows.append((p, -1, "directory in HEAD" if mode == "40000" else "gitlink 160000"))
        elif mode == "120000":
            rows.append((p, 0, "symlink 120000"))
        elif mode in REG_MODES:
            f = root / p
            if not f.is_file():
                rows.append((p, 0, "in HEAD, missing locally"))
            elif obj_id("blob", f.read_bytes()) == sha:
                rows.append((p, 1, "HEAD blob == disk"))
            else:
                rows.append((p, 0, "in HEAD, local copy differs"))
        else:
            rows.append((p, 0, "mode " + mode))
    trits = [t for _, t, _ in rows] + [head_t, up_t]
    return rows, trits, {"head": head_sha, "branch": branch,
                         "head_trit": head_t, "upstream_trit": up_t}


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
    codes += [1] * ((-len(codes)) % 128 or (128 if not codes else 0))
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
        "head_trit": meta["head_trit"], "upstream_trit": meta["upstream_trit"],
        "zero_frac": round(zero_frac, 6), "faithful": faithful,
        "release": RATE if faithful else 0.0, "i2s_sha": hashlib.sha256(i2s).hexdigest()[:16],
        "ports": list(PORTS), "engines": list(ENGINES), "flags": FLAGS,
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return body, pack, i2s, text, close_key(prev, pack, text)


# ------------------------------------------------------------ selftest
def _w_loose(objdir: Path, typ: str, data: bytes) -> str:
    raw = b"%s %d\0" % (typ.encode(), len(data)) + data
    h = hashlib.sha1(raw).hexdigest()
    p = objdir / h[:2] / h[2:]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(zlib.compress(raw))
    return h


def _tree(entries) -> bytes:
    out = b""
    for mode, name, h in sorted(entries, key=lambda e: e[1] + ("/" if e[0] == "40000" else "")):
        out += mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(h)
    return out


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _mk_delta(src: bytes, tgt: bytes) -> bytes:
    k = 0
    while k < min(len(src), len(tgt), 0xFF) and src[k] == tgt[k]:
        k += 1
    d = _varint(len(src)) + _varint(len(tgt))
    if k:
        d += bytes([0x80 | 0x10, k])  # copy offset 0, size k (1 byte)
    rest = tgt[k:]
    for i in range(0, len(rest), 127):
        d += bytes([len(rest[i:i + 127])]) + rest[i:i + 127]
    return d


def _hdr(typ: int, size: int) -> bytes:
    c = (typ << 4) | (size & 15)
    size >>= 4
    out = bytearray()
    while size:
        out.append(c | 0x80)
        c = size & 0x7F
        size >>= 7
    out.append(c)
    return bytes(out)


def _ofs(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        n -= 1
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def _w_pack(objdir: Path, objs):
    """objs: list of (typ_name, data, delta_spec) where delta_spec None|('ofs',i)|('ref',i)."""
    tn = {v: k for k, v in PACK_TYPES.items()}
    body = bytearray(b"PACK" + struct.pack(">II", 2, len(objs)))
    ents, offs = [], []
    for typ, data, spec in objs:
        off = len(body)
        offs.append(off)
        if spec is None:
            rec = _hdr(tn[typ], len(data)) + zlib.compress(data)
        else:
            delta = _mk_delta(objs[spec[1]][1], data)
            if spec[0] == "ofs":
                rec = _hdr(6, len(delta)) + _ofs(off - offs[spec[1]]) + zlib.compress(delta)
            else:
                base = obj_id(objs[spec[1]][0], objs[spec[1]][1])
                rec = _hdr(7, len(delta)) + bytes.fromhex(base) + zlib.compress(delta)
        body += rec
        ents.append((bytes.fromhex(obj_id(typ, data)), zlib.crc32(rec), off))
    psha = hashlib.sha1(body).digest()
    body += psha
    ents.sort()
    fan, j = [], 0
    for b in range(256):
        while j < len(ents) and ents[j][0][0] <= b:
            j += 1
        fan.append(j)
    idx = bytearray(b"\377tOc" + struct.pack(">I", 2) + struct.pack(">256I", *fan))
    for s, _, _ in ents:
        idx += s
    for _, c, _ in ents:
        idx += struct.pack(">I", c)
    for _, _, o in ents:
        idx += struct.pack(">I", o)
    idx += psha
    idx += hashlib.sha1(idx).digest()
    pd = objdir / "pack"
    pd.mkdir(parents=True, exist_ok=True)
    name = "pack-" + psha.hex()
    (pd / (name + ".pack")).write_bytes(bytes(body))
    (pd / (name + ".idx")).write_bytes(bytes(idx))


def _mk_repo(base: Path, files: dict, committed: list, packed: bool, upstream_same: bool):
    root = base
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    for p, data in files.items():
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    gd = root / ".git"
    objdir = gd / "objects"
    objdir.mkdir(parents=True)
    # build nested trees for committed paths
    blobs = {p: files[p] for p in committed}
    dirs: dict = {}
    for p in committed:
        parts = p.split("/")
        for k in range(len(parts)):
            dirs.setdefault("/".join(parts[:k]), set())
        dirs["/".join(parts[:-1])].add(("f", parts[-1]))
        for k in range(1, len(parts)):
            dirs["/".join(parts[:k - 1])].add(("d", parts[k - 1]))
    objs = []
    tree_ids = {}

    def build(d):
        ents = []
        for kind, name in sorted(dirs.get(d, ())):
            full = (d + "/" + name) if d else name
            if kind == "f":
                ents.append(("100644", name, obj_id("blob", blobs[full])))
            else:
                ents.append(("40000", name, build(full)))
        data = _tree(ents)
        objs.append(("tree", data, None))
        tree_ids[d] = obj_id("tree", data)
        return tree_ids[d]

    root_tree = build("")
    commit = (b"tree %s\nauthor t <t@t> 0 +0000\ncommitter t <t@t> 0 +0000\n\nselftest\n"
              % root_tree.encode())
    blob_list = [("blob", blobs[p], None) for p in committed]
    if packed and len(blob_list) >= 3:
        blob_list[1] = ("blob", blob_list[1][1], ("ofs", 0))
        blob_list[2] = ("blob", blob_list[2][1], ("ref", 0))
    allobjs = blob_list + objs + [("commit", commit, None)]
    if packed:
        _w_pack(objdir, allobjs)
    else:
        for typ, data, _ in allobjs:
            _w_loose(objdir, typ, data)
    csha = obj_id("commit", commit)
    (gd / "HEAD").write_text("ref: refs/heads/main\n")
    if packed:
        lines = ["# pack-refs with: peeled fully-peeled sorted", csha + " refs/heads/main"]
        if upstream_same:
            lines.append(csha + " refs/remotes/origin/main")
        (gd / "packed-refs").write_text("\n".join(lines) + "\n")
    else:
        (gd / "refs/heads").mkdir(parents=True)
        (gd / "refs/heads/main").write_text(csha + "\n")
        (gd / "refs/remotes/origin").mkdir(parents=True)
        (gd / "refs/remotes/origin/main").write_text((csha if upstream_same else "1" * 40) + "\n")
    return root


def selftest() -> int:
    exp = ["docs/A.md", "docs/B.md", "STATE.md", "rails/linux/x.sh"]
    bf = ("EXPECTED = [\n" + "".join('    "%s",\n' % p for p in exp) + "]\n").encode()
    a = b"# A\n" + b"alpha line\n" * 20
    files = {"docs/A.md": a, "docs/B.md": a + b"beta tail\n", "STATE.md": a[:40] + b"state\n",
             "rails/linux/x.sh": b"#!/bin/sh\nexit 0\n", BACKFILL: bf}
    allp = list(exp) + [BACKFILL]
    ok = True
    keys = []

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL", msg)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1 loose, all committed, upstream equal -> faithful
        r = _mk_repo(td / "loose", files, allp, packed=False, upstream_same=True)
        rows, trits, meta = audit(r)
        check(trits == [1, 1, 1, 1, 1, 1], "loose all +1 %r" % trits)
        b, pack, i2s, text, key = verdict(rows, trits, meta)
        check(b["faithful"] and b["release"] == RATE, "loose faithful")
        keys.append(key)
        # 2 packed with ofs+ref deltas -> same verdict
        r = _mk_repo(td / "packed", files, allp, packed=True, upstream_same=True)
        rows2, trits2, meta2 = audit(r)
        check(trits2 == trits, "packed equals loose %r" % trits2)
        # 3 staged-only path (on disk, not in HEAD) -> -1, held
        r = _mk_repo(td / "staged", files, [p for p in allp if p != "docs/B.md"],
                     packed=True, upstream_same=True)
        rows3, trits3, meta3 = audit(r)
        check(trits3[1] == -1 and rows3[1][2] == "absent from HEAD tree", "staged-only -1")
        b3 = verdict(rows3, trits3, meta3)[0]
        check(not b3["faithful"] and b3["release"] == 0.0, "staged-only held")
        # 4 local edit + upstream differs -> 0s, held by rate
        r = _mk_repo(td / "edit", files, allp, packed=False, upstream_same=False)
        (r / "STATE.md").write_bytes(b"edited\n")
        rows4, trits4, meta4 = audit(r)
        check(trits4[2] == 0 and trits4[-1] == 0 and -1 not in trits4, "edit/upstream 0 %r" % trits4)
        check(not verdict(rows4, trits4, meta4)[0]["faithful"], "0 frac > rate held")
        # 5 no .git -> head -1, all 0
        r = td / "bare"
        (r / "scripts").mkdir(parents=True)
        (r / BACKFILL).write_bytes(bf)
        rows5, trits5, _ = audit(r)
        check(trits5 == [0, 0, 0, 0, -1, 0], "no git %r" % trits5)
        # 6 corrupted loose object -> unreadable, not silently +1
        r = _mk_repo(td / "corrupt", files, allp, packed=False, upstream_same=True)
        blob = obj_id("blob", files["docs/A.md"])
        (r / ".git/objects" / blob[:2] / blob[2:]).write_bytes(zlib.compress(b"blob 1\0x"))
        rows6, trits6, _ = audit(r)
        check(trits6[0] == 1, "blob content not needed for tree lookup")
        tsha = None
        for p in (r / ".git/objects").rglob("*"):
            if p.is_file():
                raw = zlib.decompress(p.read_bytes())
                if raw.startswith(b"tree") and b"A.md" in raw:
                    tsha = p
        if tsha is not None:
            tsha.write_bytes(zlib.compress(b"tree 0\0"))
        rows6, trits6, _ = audit(r)
        check(trits6[0] == 0 and rows6[0][2] == "tree unreadable", "corrupt tree -> 0 %r" % rows6[0:1])
    # JTR1 / I2_S / close rule
    g, back = jtr1_unpack(pack)
    check(back == trits and abs(g - 1.0) < 1e-6, "jtr1 round trip")
    try:
        jtr1_unpack(b"JTR1" + struct.pack("<fI", 1.0, 1) + b"\xc0")
        check(False, "code 11 accepted")
    except ValueError:
        pass
    check(len(i2s) == 32 and i2s[0] == 0b10101010, "i2s 128-block")
    check(len(i2s_block(trits3 * 30)) == 64, "i2s two blocks")
    check(key == close_key(GENESIS, pack, text) and len(key) == 16, "close rule")
    check(key == verdict(rows, trits, meta)[4], "deterministic")
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
        print("head %+d upstream %+d zero_frac %.3f faithful %s release %s"
              % (meta["head_trit"], meta["upstream_trit"], body["zero_frac"],
                 body["faithful"], body["release"]))
        print("key", key, "i2s", body["i2s_sha"])
    return 0 if body["faithful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
