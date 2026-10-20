#!/usr/bin/env python3
"""Live beta step 5 fresh-clone fidelity: git-index trits for backfill_check EXPECTED. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 5
("`python scripts/backfill_check.py` exits 0"), which the doc scopes to
"A fresh clone"; docs/COMPILED_BACKLOG.md "A - live beta suite".
Quant Chief pair A, slot 2026-10-19 18:05 MT. Trit Pack & I2_S side: the
trit pack and I2_S key. Winsor & Quant Error covers the quant-error bound side
of the same item in a different file.

Gap it fills (not in BitNet-mlx 9244444 exit-code trits, JuniorQuant adfb204
SPRT gate, or JuniorLLM 829a789 coverage handshake): backfill_check.py tests
`(ROOT / p).is_file()` on the working tree. A dev tree can exit 0 while a fresh
clone exits 1, because an EXPECTED path is untracked or ignored, or the copy on
disk differs from what is committed. This file reads the local `.git/index`
directly (binary DIRC v2/v3/v4 parse, trailing SHA-1 checksum verified; no git
subprocess, no remote, no fetch) and gives one balanced trit per EXPECTED path:
    +1  tracked, stage 0, regular file mode (100644/100755), and the working
        tree blob id sha1("blob <len>\\0" + bytes) equals the index blob id
     0  tracked but not fresh-clone-identical: working copy modified, merge
        stage != 0, symlink (120000) or gitlink (160000)
    -1  not in the index (local pass a fresh clone would not reproduce), or
        indexed but missing on disk
plus one index-integrity trit (+1 checksum ok, -1 bad, 0 unsupported/absent).
EXPECTED is read from scripts/backfill_check.py with `ast` (literal list only;
never imported or run).

Verdict: fresh-clone faithful when no -1 and the 0 fraction is <= RATE (0.05);
release RATE when faithful, else 0.0 (held).

JTR1 (not GGUF): b"JTR1" | gamma <f32 LE, absmean> | n <u32 LE> | 2-bit
payload MSB first (0 -> 00, +1 -> 01, -1 -> 10; 11 rejected).
I2_S block: code = trit + 1, four per byte MSB first, padded with code 1 (zero)
to a 128-trit block, as BitNet I2_S stores ternary.
Close rule: sha256(prev || pack || text)[:16]; prev 16-hex ASCII, pack = JTR1
bytes, text = canonical verdict JSON (sorted keys, compact; includes i2s_sha).
Rate 0.05. Compatible engines: JuniorLLM, BitNet-mlx, i2sd userspace,
JuniorOS overlay. Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no hardware calls, no subprocess, no network.

Usage: python3 scripts/backfill_fresh_clone_index_i2s_key.py --selftest
       python3 scripts/backfill_fresh_clone_index_i2s_key.py [--root DIR]
              [--index PATH] [--prev HEX16]
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
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM", "BitNet-mlx", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md Live beta step 5: backfill_check "
                "exits 0 on a fresh clone (COMPILED_BACKLOG A)")
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
BACKFILL = "scripts/backfill_check.py"
REGULAR = (0o100644, 0o100755)
_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


# ---- git index (DIRC) reader, offline ------------------------------------
def _varint(buf, i):
    c = buf[i]; i += 1
    val = c & 0x7F
    while c & 0x80:
        c = buf[i]; i += 1
        val = ((val + 1) << 7) | (c & 0x7F)
    return val, i


def parse_index(blob: bytes) -> dict:
    """Return {"ok": trit, "version": v, "entries": {path: (mode, sha_hex, stage)}}."""
    if len(blob) < 32 or blob[:4] != b"DIRC":
        return {"ok": 0, "version": None, "entries": {}, "note": "no DIRC header"}
    body, tail = blob[:-20], blob[-20:]
    if hashlib.sha1(body).digest() != tail:
        return {"ok": -1, "version": None, "entries": {}, "note": "checksum mismatch"}
    version, count = struct.unpack(">II", blob[4:12])
    if version not in (2, 3, 4):
        return {"ok": 0, "version": version, "entries": {}, "note": "unsupported version"}
    i, prev, ents = 12, b"", {}
    for _ in range(count):
        start = i
        mode = struct.unpack(">I", body[i + 24:i + 28])[0]
        sha = body[i + 40:i + 60].hex()
        flags = struct.unpack(">H", body[i + 60:i + 62])[0]
        i += 62
        if version >= 3 and flags & 0x4000:
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
        ents[name.decode("utf-8", "surrogateescape")] = (mode, sha, (flags >> 12) & 3)
    return {"ok": 1, "version": version, "entries": ents, "note": "checksum ok"}


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def expected_from_source(src: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "EXPECTED" for t in node.targets):
            val = ast.literal_eval(node.value)
            if isinstance(val, (list, tuple)) and all(isinstance(p, str) for p in val):
                return list(val)
    raise ValueError("EXPECTED literal list not found")


def path_trit(root: Path, rel: str, ents: dict):
    p = root / rel
    ent = ents.get(rel)
    if ent is None:
        if p.is_file():
            return -1, "untracked: local pass, fresh clone would miss it"
        return -1, "absent from index and disk"
    mode, sha, stage = ent
    if stage:
        return 0, "merge stage %d" % stage
    if mode not in REGULAR:
        return 0, "mode %o not a regular file" % mode
    if p.is_symlink() or not p.is_file():
        return -1 if not p.exists() else 0, "indexed but working copy missing or not regular"
    if blob_id(p.read_bytes()) != sha:
        return 0, "working copy differs from index blob"
    return 1, "tracked, clean, blob %s" % sha[:12]


# ---- packing ---------------------------------------------------------------
def pack_jtr1(trits) -> bytes:
    n = len(trits)
    gamma = (sum(abs(t) for t in trits) / n) if n else 0.0
    buf = bytearray((2 * n + 7) // 8)
    for k, t in enumerate(trits):
        if t not in _ENC:
            raise ValueError("not a balanced trit: %r" % (t,))
        buf[(2 * k) // 8] |= _ENC[t] << (6 - (2 * k) % 8)
    return MAGIC + struct.pack("<fI", gamma, n) + bytes(buf)


def unpack_jtr1(blob: bytes):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    pay = blob[12:]
    if len(pay) != (2 * n + 7) // 8:
        raise ValueError("payload length mismatch")
    out = []
    for k in range(n):
        c = (pay[(2 * k) // 8] >> (6 - (2 * k) % 8)) & 3
        if c == 3:
            raise ValueError("reserved code 0b11 at %d" % k)
        out.append(_DEC[c])
    return gamma, out


def pack_i2s(trits) -> bytes:
    n = len(trits)
    padded = n + ((-n) % I2S_BLOCK) if n else I2S_BLOCK
    codes = [t + 1 for t in trits] + [1] * (padded - n)
    out = bytearray(padded // 4)
    for k, c in enumerate(codes):
        out[k // 4] |= c << (6 - 2 * (k % 4))
    return bytes(out)


def close_key(prev: str, pack: bytes, text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    return hashlib.sha256(prev.encode("ascii") + pack + text.encode("utf-8")).hexdigest()[:16]


def build(root: Path, index_blob: bytes, expected, prev: str = GENESIS) -> dict:
    idx = parse_index(index_blob) if index_blob else {
        "ok": 0, "version": None, "entries": {}, "note": "no index"}
    rows = [[rel] + list(path_trit(root, rel, idx["entries"])) for rel in expected]
    trits = [r[1] for r in rows] + [idx["ok"]]
    zero = trits.count(0) / len(trits)
    faithful = (-1 not in trits) and zero <= RATE
    pk, i2s = pack_jtr1(trits), pack_i2s(trits)
    v = {
        "item": ROADMAP_ITEM, "rows": rows,
        "index": {"trit": idx["ok"], "version": idx["version"], "note": idx["note"]},
        "zero_frac": round(zero, 6), "faithful": faithful,
        "release": RATE if faithful else 0.0, "rate": RATE,
        "i2s_sha": hashlib.sha256(i2s).hexdigest()[:16], "i2s_bytes": len(i2s),
        "bind": BIND, "ports": list(PORTS), "engines": list(ENGINES),
        "mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
        "fetch": FETCH, "ue5_launch": UE5_LAUNCH,
    }
    text = json.dumps(v, sort_keys=True, separators=(",", ":"))
    v["jtr1_hex"], v["prev"], v["key"] = pk.hex(), prev, close_key(prev, pk, text)
    return v


# ---- selftest ---------------------------------------------------------------
def _write_index(entries, version=2) -> bytes:
    """Selftest-only DIRC writer: entries = [(path, mode, sha_hex, stage)]."""
    out, prev = bytearray(b"DIRC" + struct.pack(">II", version, len(entries))), b""
    for path, mode, sha, stage in sorted(entries):
        name = path.encode()
        e = bytearray(struct.pack(">10I", 0, 0, 0, 0, 0, 0, mode, 0, 0, 0))
        e += bytes.fromhex(sha) + struct.pack(">H", (stage << 12) | min(len(name), 0xFFF))
        if version == 4:
            common = 0
            while common < min(len(prev), len(name)) and prev[common] == name[common]:
                common += 1
            strip = len(prev) - common
            assert strip < 0x80
            e += bytes([strip]) + name[common:] + b"\0"
        else:
            e += name
            e += b"\0" * (8 - (len(e) % 8))
        out += e
        prev = name
    return bytes(out) + hashlib.sha1(bytes(out)).digest()


def selftest() -> int:
    assert RATE == 0.05 and BIND == "127.0.0.1"
    assert not any((MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH))
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    for t in ([], [1], [-1, 0, 1], [1, -1, 0, 0, 1, -1, 1]):
        assert unpack_jtr1(pack_jtr1(t))[1] == t
    blk = pack_i2s([-1, 0, 1, 1])
    assert len(blk) == I2S_BLOCK // 4 and blk[0] == 0b00011010 and blk[1] == 0b01010101
    assert len(pack_i2s([1] * 129)) == 64
    assert blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    src = 'EXPECTED = [\n    "docs/A.md",\n    "docs/B.md",\n    "docs/C.md",\n]\n'
    exp = expected_from_source(src)
    assert exp == ["docs/A.md", "docs/B.md", "docs/C.md"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "docs").mkdir()
        data = {p: ("body of %s\n" % p).encode() for p in exp}
        for p, d in data.items():
            (root / p).write_bytes(d)
        clean = [(p, 0o100644, blob_id(d), 0) for p, d in data.items()]
        for ver in (2, 3, 4):
            v = build(root, _write_index(clean, ver), exp)
            assert v["faithful"] and v["release"] == RATE, (ver, v["rows"])
            assert [r[1] for r in v["rows"]] == [1, 1, 1] and v["index"]["trit"] == 1
        good = build(root, _write_index(clean), exp)
        # untracked: local backfill passes, fresh clone would not
        v = build(root, _write_index(clean[:2]), exp)
        assert v["rows"][2][1] == -1 and "untracked" in v["rows"][2][2]
        assert not v["faithful"] and v["release"] == 0.0
        # modified working copy -> 0 (1 of 4 trits = 0.25 > RATE)
        (root / "docs/B.md").write_bytes(b"edited\n")
        v = build(root, _write_index(clean), exp)
        assert v["rows"][1][1] == 0 and not v["faithful"]
        (root / "docs/B.md").write_bytes(data["docs/B.md"])
        # merge stage and symlink mode -> 0
        st = [clean[0], ("docs/B.md", 0o100644, clean[1][2], 2), clean[2]]
        assert build(root, _write_index(st), exp)["rows"][1][1] == 0
        sl = [clean[0], clean[1], ("docs/C.md", 0o120000, clean[2][2], 0)]
        assert build(root, _write_index(sl), exp)["rows"][2][1] == 0
        # indexed but deleted on disk -> -1
        (root / "docs/A.md").unlink()
        assert build(root, _write_index(clean), exp)["rows"][0][1] == -1
        (root / "docs/A.md").write_bytes(data["docs/A.md"])
        # corrupt checksum -> index trit -1, missing index -> 0
        bad = bytearray(_write_index(clean)); bad[-1] ^= 1
        v = build(root, bytes(bad), exp)
        assert v["index"]["trit"] == -1 and not v["faithful"]
        assert build(root, b"", exp)["index"]["trit"] == 0
        # close rule recomputed independently, chaining, determinism
        text = json.dumps({k: good[k] for k in good if k not in ("jtr1_hex", "prev", "key")},
                          sort_keys=True, separators=(",", ":"))
        pk = bytes.fromhex(good["jtr1_hex"])
        assert good["key"] == hashlib.sha256(GENESIS.encode() + pk + text.encode()).hexdigest()[:16]
        assert build(root, _write_index(clean), exp)["key"] == good["key"]
        assert build(root, _write_index(clean), exp, prev=good["key"])["key"] != good["key"]
        g, tr = unpack_jtr1(pk)
        assert tr == [1, 1, 1, 1] and abs(g - 1.0) < 1e-6
    print("selftest ok key=%s release=%s rate=%s" % (good["key"], good["release"], RATE))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--index", help="git index file (default ROOT/.git/index)")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    root = Path(a.root)
    exp = expected_from_source((root / BACKFILL).read_text(encoding="utf-8"))
    ipath = Path(a.index) if a.index else root / ".git" / "index"
    blob = ipath.read_bytes() if ipath.is_file() else b""
    v = build(root, blob, exp, a.prev)
    print(json.dumps(v, sort_keys=True, indent=1))
    return 0 if v["faithful"] else 1


if __name__ == "__main__":
    sys.exit(main())
