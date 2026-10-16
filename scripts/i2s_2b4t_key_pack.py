"""I2_S 2B4T key pack for JuniorOS overlay. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md — JuniorOS overlay "I2_S 2B4T only if
GGUF already on disk" — plus JuniorHome docs/ROADMAP.md step 3 "i2sd userspace —
127.0.0.1:8767. No docker.sock." (Quant Chief pair_B JuniorOS overlay, 2026-10-16).
Step 2 GGUF: header-only local probe when JUNIOR_GGUF points at an on-disk file;
never download.

New capability vs scripts/i2s_key_builder.py and BitNet-mlx scripts/i2sd_jtr1_pack.py:
2B4T unit framing (4 trits -> 1 byte via I2_S 2-bit codes) plus an OS-overlay key
material builder that closes under the standing rule and optionally reads a local
GGUF magic/version header. Not a wrapper or re-derivation of those files.

2B4T layout (I2_S, not GGUF):
  each unit packs 4 trits as four 2-bit codes MSB-first (0->0, 1->1, -1->2)
  key material = concat of 2B4T units + close tip

Close rule: sha256(prev || pack || text)[:16], pack = I2_S / 2B4T payload bytes.
Rate 0.05 held. Compatible engines: JuniorLLM, BitNet-mlx, i2sd userspace,
JuniorOS overlay, web3node.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
The i2sd endpoint is named only (http://127.0.0.1:8767); no socket is opened here.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false. No docker.sock.
No model pull, no GGUF download, no network.

Usage: python3 i2s_2b4t_key_pack.py [--selftest] [--gguf PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
I2SD_PORT = 8767
I2SD_ENDPOINT = "http://%s:%d" % (BIND, I2SD_PORT)
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay I2_S 2B4T (GGUF on-disk only); "
    "JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767"
)
GENESIS = "0" * 16
UNIT = 4  # 2B4T: 4 trits per byte
GGUF_MAGIC = b"GGUF"

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def pack_i2s(trits) -> bytes:
    """Balanced trits (-1/0/1) as 2-bit I2_S codes, MSB first, zero-padded."""
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def unpack_i2s(data: bytes, n: int) -> list:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def pack_2b4t(trits) -> bytes:
    """Pack trits into 2B4T units (exactly 4 trits per byte; pad with 0)."""
    padded = list(trits)
    while len(padded) % UNIT:
        padded.append(0)
    return pack_i2s(padded)


def unpack_2b4t(data: bytes, n: int) -> list:
    """Unpack 2B4T bytes back to n trits (strip pad)."""
    return unpack_i2s(data, n)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def gguf_header_local(path: str) -> dict:
    """Read GGUF magic + version only from a local file. Never download.

    Returns ok=False if path missing or not a GGUF. Does not open sockets or
    fetch remote URLs.
    """
    if not path or path.startswith(("http://", "https://", "ftp://")):
        return {"ok": False, "why": "remote_or_empty"}
    if not os.path.isfile(path):
        return {"ok": False, "why": "missing", "path": path}
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError as e:
        return {"ok": False, "why": "io", "error": str(e)}
    if len(head) < 8 or head[:4] != GGUF_MAGIC:
        return {"ok": False, "why": "not_gguf", "path": path}
    version = struct.unpack("<I", head[4:8])[0]
    return {
        "ok": True,
        "magic": "GGUF",
        "version": version,
        "path": path,
        "bytes_read": 8,
        "download": False,
    }


def build_2b4t_key(
    trits,
    prev: str = GENESIS,
    text: str = "junioros|i2s|2b4t|0.05",
    gguf_path: str | None = None,
) -> dict:
    """Build OS-overlay I2_S 2B4T key material and close under the standing rule.

    Optional gguf_path: local GGUF header probe only (step 2); never download.
    """
    if BIND != "127.0.0.1" or BIND == "0.0.0.0":
        return {"ok": False, "why": "bind"}
    if not I2SD_ENDPOINT.startswith("http://127.0.0.1:"):
        return {"ok": False, "why": "endpoint"}
    if MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH:
        return {"ok": False, "why": "flags"}

    gguf = None
    if gguf_path:
        gguf = gguf_header_local(gguf_path)
        # Overlay path allows missing GGUF; key still builds. On-disk only when present.
        if gguf.get("ok") is False and gguf.get("why") == "remote_or_empty":
            return {"ok": False, "why": "gguf_remote"}

    pk = pack_2b4t(trits)
    tip = close_rule(prev, pk, text)
    return {
        "ok": True,
        "pack": pk.hex(),
        "n": len(trits),
        "units": (len(trits) + UNIT - 1) // UNIT,
        "prev": prev,
        "tip": tip,
        "rate": RATE,
        "held": True,
        "released": 0,
        "mint": MINT,
        "socket": SOCKET,
        "layer_one": LAYER_ONE,
        "fetch": FETCH,
        "ue5_launch": UE5_LAUNCH,
        "bind": BIND,
        "endpoint": I2SD_ENDPOINT,
        "ports": list(PORTS),
        "text": text,
        "roadmap": ROADMAP_ITEM,
        "gguf": gguf,
        "layout": "2B4T",
    }


def rate_step(x: float) -> float:
    return x * RATE


def self_check() -> dict:
    assert BIND == "127.0.0.1" and I2SD_ENDPOINT == "http://127.0.0.1:8767"
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert abs(RATE - 0.05) < 1e-12

    # Cross-engine I2_S vector (same bytes as other engines).
    t = [1, 0, -1, -1, 1, 0, 1]
    p = pack_i2s(t)
    assert p == bytes([0b01001010, 0b01000100]), p
    assert unpack_i2s(p, len(t)) == t

    tip = close_rule(GENESIS, p, "web3")
    assert tip == hashlib.sha256(GENESIS.encode() + p + b"web3").hexdigest()[:16]
    assert len(tip) == 16 and all(c in "0123456789abcdef" for c in tip)

    # 2B4T: 4 trits -> 1 byte; pad to unit boundary.
    u = [1, 0, -1, -1]
    b = pack_2b4t(u)
    assert len(b) == 1 and b[0] == 0b01001010
    assert unpack_2b4t(b, 4) == u
    odd = [1, 0, -1]
    b2 = pack_2b4t(odd)
    assert len(b2) == 1 and unpack_2b4t(b2, 3) == odd

    text = "junioros|i2s|2b4t|0.05"
    a = build_2b4t_key(t, GENESIS, text)
    bkey = build_2b4t_key(t, a["tip"], text + "|next")
    expected = close_rule(GENESIS, pack_2b4t(t), text)
    assert a["ok"] and bkey["ok"]
    assert a["tip"] == expected and len(a["tip"]) == 16
    assert bkey["prev"] == a["tip"]
    assert a["rate"] == 0.05 and a["held"] is True and a["released"] == 0
    assert a["bind"] == "127.0.0.1" and a["endpoint"] == I2SD_ENDPOINT
    assert a["layout"] == "2B4T" and a["units"] == 2  # 7 trits -> 2 bytes
    assert abs(rate_step(1.0) - 0.05) < 1e-12

    # Round-trip pack through key material.
    recovered = unpack_2b4t(bytes.fromhex(a["pack"]), a["n"])
    assert recovered == t

    # Local GGUF header: reject remote; missing is soft; synthetic local ok.
    assert gguf_header_local("https://example.com/m.gguf")["why"] == "remote_or_empty"
    assert gguf_header_local("/no/such/file.gguf")["why"] == "missing"
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tf:
        tf.write(GGUF_MAGIC + struct.pack("<I", 3) + b"\x00" * 16)
        gpath = tf.name
    try:
        gh = gguf_header_local(gpath)
        assert gh["ok"] and gh["magic"] == "GGUF" and gh["version"] == 3
        assert gh["download"] is False and gh["bytes_read"] == 8
        keyed = build_2b4t_key(t, gguf_path=gpath)
        assert keyed["ok"] and keyed["gguf"]["ok"] is True
    finally:
        os.unlink(gpath)

    bad = build_2b4t_key(t, gguf_path="https://evil.example/x.gguf")
    assert bad["ok"] is False and bad["why"] == "gguf_remote"

    return {
        "selftest": "pass",
        "close": a["tip"],
        "endpoint": I2SD_ENDPOINT,
        "roadmap": ROADMAP_ITEM,
        "ports": list(PORTS),
        "rate": RATE,
        "close_rule": "sha256(prev || pack || text)[:16]",
        "layout": "2B4T",
        "units": a["units"],
    }


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="I2_S 2B4T key pack (JuniorOS overlay)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gguf", default=None, help="local GGUF path (header only; never download)")
    args = ap.parse_args(argv)
    if args.selftest:
        print(json.dumps(self_check(), sort_keys=True))
        return 0
    # Demo key from a short trit vector; optional local GGUF header.
    demo = [1, 0, -1, -1, 1, 0, 1, -1, 0, 1, -1, 1]
    env_gguf = os.environ.get("JUNIOR_GGUF") or args.gguf
    m = build_2b4t_key(demo, gguf_path=env_gguf)
    print(json.dumps(m, sort_keys=True))
    return 0 if m.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
