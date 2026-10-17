"""i2sd wire-transcript JTR1 key. Stdlib only, offline, no socket.

Roadmap item: JuniorHome docs/ROADMAP.md step 3, "i2sd userspace - 127.0.0.1:8767.
No docker.sock." (Quant Chief, pair A, slot 2026-10-17 05:05 MT). This file is the
I2_S / trit pack side of that item; the quant-error bound side belongs to
Winsor & Quant Error in a different repo path.

New capability: an offline auditor for i2sd line-protocol transcripts (the
PING / PACK / UNPACK request lines and their OK / PONG / ERR replies spoken by
the userspace daemon at 127.0.0.1:8767). It reads transcript text that was
already captured, checks every reply for wire integrity, bridges each valid
PACK reply into a JTR1 frame (gamma = the reply's absmean scale), and folds the
whole transcript into one I2_S transcript key. It opens no socket, imports no
team module, and does not wrap, test or re-derive any other team file.

Per-exchange trit (one per request/reply pair):
  +1 reply is well formed and self-consistent
   0 PING/PONG liveness only, or an ERR reply to an unknown verb
  -1 broken: wrong n, payload length != ceil(2n/8), I2_S code 3, non-zero pad
     bits, non-finite or negative scale, UNPACK reply != decoded request, or a
     reply from a non-loopback peer

JTR1 layout (not GGUF):
  magic  : 4 bytes b'JTR1'
  gamma  : 8 bytes big-endian IEEE-754 float64
  n      : 4 bytes big-endian uint32 (trit count)
  payload: ceil(2n/8) bytes, 2-bit I2_S codes MSB-first (0->0, 1->1, -1->2)

Close rule: sha256(prev || pack || text)[:16]; per exchange pack = the I2_S
payload bytes (empty for PING), text = canonical exchange JSON; the transcript
key closes the per-exchange trit vector packed as JTR1. Rate 0.05 held.
Compatible engines: BitNet-mlx, JuniorLLM, i2sd userspace, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (endpoint named, never dialed). No docker.sock.
mint/socket/layer_one/fetch/ue5_launch false. No model pull, no GGUF download.

Transcript format, one line each, peer optional:
  > PACK 0.5,-1.2,0.01
  < OK 3 0.570000 64
  [127.0.0.1] < PONG

Usage: python3 i2sd_wire_jtr1_key.py [--selftest] [--transcript PATH] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
I2SD_ENDPOINT = "%s:%d" % (BIND, I2SD_PORT)
LOOPBACK_PEERS = ("127.0.0.1", "::1", "localhost")
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("BitNet-mlx", "JuniorLLM", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767 no docker.sock"
GENESIS = "0" * 16
MAGIC = b"JTR1"

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def pack_i2s(trits) -> bytes:
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


def strict_unpack(data: bytes, n: int) -> list:
    """Decode n trits; raise on code 3, wrong length, or non-zero pad bits."""
    if len(data) != (2 * n + 7) // 8:
        raise ValueError("payload length != ceil(2n/8)")
    out = []
    for i, b in enumerate(data):
        for s in (6, 4, 2, 0):
            idx = i * 4 + (6 - s) // 2
            code = (b >> s) & 3
            if idx >= n:
                if code:
                    raise ValueError("non-zero pad bits")
                continue
            if code == 3:
                raise ValueError("invalid I2_S code 3")
            out.append(_DEC[code])
    return out


def encode_jtr1(trits, gamma: float = 1.0) -> bytes:
    trits = list(trits)
    return MAGIC + struct.pack(">dI", float(gamma), len(trits)) + pack_i2s(trits)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def rate_step(x: float) -> float:
    return x * (1.0 + RATE)


_LINE = re.compile(r"^\s*(?:\[([^\]]+)\]\s*)?([<>])\s?(.*)$")


def parse_transcript(text: str) -> list:
    """Pair each '>' request with the next '<' reply. Returns list of dicts."""
    pairs, cur = [], None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if not m:
            raise ValueError("bad transcript line: %r" % raw)
        peer, arrow, body = m.group(1) or BIND, m.group(2), m.group(3).strip()
        if arrow == ">":
            if cur is not None:
                pairs.append(cur)
            cur = {"req": body, "reply": None, "peer": peer}
        else:
            if cur is None:
                raise ValueError("reply without request")
            cur["reply"], cur["peer"] = body, peer
            pairs.append(cur)
            cur = None
    if cur is not None:
        pairs.append(cur)
    return pairs


def check_exchange(req: str, reply, peer: str) -> dict:
    """Return {'trit', 'verb', 'why', 'pack', 'gamma', 'n'} for one exchange."""
    verb = req.split(" ", 1)[0] if req else ""
    r = {"verb": verb, "trit": 1, "why": "ok", "pack": b"", "gamma": 1.0, "n": 0}

    def bad(why):
        r.update(trit=-1, why=why)
        return r

    if peer not in LOOPBACK_PEERS:
        return bad("non-loopback peer")
    if reply is None:
        return bad("no reply")
    if verb == "PING":
        if reply != "PONG":
            return bad("PING reply not PONG")
        r.update(trit=0, why="liveness")
        return r
    if verb == "PACK":
        xs = [x for x in req[5:].split(",") if x.strip()]
        parts = reply.split()
        if len(parts) not in (3, 4) or parts[0] != "OK":
            return bad("PACK reply grammar")
        try:
            n, scale = int(parts[1]), float(parts[2])
            blob = bytes.fromhex(parts[3]) if len(parts) == 4 else b""
        except ValueError:
            return bad("PACK reply fields")
        if n != len(xs):
            return bad("n != request length")
        if not math.isfinite(scale) or scale < 0:
            return bad("scale not finite >= 0")
        try:
            strict_unpack(blob, n)
        except ValueError as e:
            return bad(str(e))
        r.update(pack=blob, gamma=scale, n=n)
        return r
    if verb == "UNPACK":
        f = req.split()
        if len(f) != 3:
            return bad("UNPACK request grammar")
        try:
            n = int(f[2])
            blob = bytes.fromhex(f[1])
            want = strict_unpack(blob, n)
        except ValueError as e:
            return bad("UNPACK request: %s" % e)
        if not reply.startswith("OK"):
            return bad("UNPACK reply grammar")
        body = reply[2:].strip()
        try:
            got = [int(x) for x in body.split(",") if x.strip()]
        except ValueError:
            return bad("UNPACK reply fields")
        if got != want:
            return bad("UNPACK reply != decoded request")
        r.update(pack=blob, n=n)
        return r
    if reply == "ERR":
        r.update(trit=0, why="unknown verb, ERR")
        return r
    return bad("unknown verb without ERR")


def transcript_key(text: str, prev: str = GENESIS) -> dict:
    chain = prev
    rows, frames, trits = [], [], []
    for i, ex in enumerate(parse_transcript(text)):
        c = check_exchange(ex["req"], ex["reply"], ex["peer"])
        row = {"i": i, "verb": c["verb"], "trit": c["trit"], "why": c["why"], "n": c["n"]}
        chain = close_rule(chain, c["pack"], json.dumps(row, sort_keys=True, separators=(",", ":")))
        row["key"] = chain
        rows.append(row)
        trits.append(c["trit"])
        if c["verb"] == "PACK" and c["trit"] == 1:
            frames.append(encode_jtr1(strict_unpack(c["pack"], c["n"]), c["gamma"]).hex())
    vec = encode_jtr1(trits, 1.0)
    summary = {
        "item": ROADMAP_ITEM,
        "endpoint": I2SD_ENDPOINT,
        "exchanges": len(rows),
        "admit": bool(rows) and all(t != -1 for t in trits),
        "rate": RATE,
        "transcript_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    stext = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return {
        "summary": summary,
        "rows": rows,
        "jtr1_frames": frames,
        "vector_frame_hex": vec.hex(),
        "key": close_rule(chain, vec[16:], stext),
        "prev": prev,
    }


GOOD = """# captured offline from i2sd userspace
> PING
< PONG
> PACK 0.5,-1.2,0.01,0.9
< OK 4 0.652500 64
> UNPACK 64 4
< OK 1,-1,1,0
> BOGUS
< ERR
"""

BAD = """> PACK 0.5,-1.2
< OK 3 0.850000 60
> UNPACK 7c 3
< OK 1,3,0
> PING
[10.0.0.9] < PONG
"""


def selftest() -> int:
    assert RATE == 0.05 and abs(rate_step(1.0) - 1.05) < 1e-12
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert BIND == "127.0.0.1" and I2SD_PORT == 8767
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    # pack / strict unpack round trip, pad and code-3 rejection
    t = [1, -1, 0, 1, -1]
    p = pack_i2s(t)
    assert p == bytes([0b01100001, 0b10000000]) and strict_unpack(p, 5) == t
    for blob, n in ((bytes([0b01100001, 0b10000001]), 5), (bytes([0b11000000]), 1), (b"\x00", 5)):
        try:
            strict_unpack(blob, n)
            raise AssertionError("strict_unpack accepted bad blob")
        except ValueError:
            pass
    # close rule matches the literal definition
    assert close_rule(GENESIS, b"\x01", "x") == hashlib.sha256(b"0" * 16 + b"\x01" + b"x").hexdigest()[:16]
    g = transcript_key(GOOD)
    assert [r["trit"] for r in g["rows"]] == [0, 1, 1, 0], g["rows"]
    assert g["summary"]["admit"] is True and len(g["jtr1_frames"]) == 1
    f = bytes.fromhex(g["jtr1_frames"][0])
    assert f[:4] == MAGIC and struct.unpack(">dI", f[4:16]) == (0.6525, 4) and f[16:] == b"\x64"
    v = bytes.fromhex(g["vector_frame_hex"])
    assert v[:4] == MAGIC and struct.unpack(">I", v[12:16])[0] == 4
    assert g["key"] == transcript_key(GOOD)["key"] and len(g["key"]) == 16
    assert transcript_key(GOOD, prev="f" * 16)["key"] != g["key"]
    b = transcript_key(BAD)
    assert [r["trit"] for r in b["rows"]] == [-1, -1, -1], b["rows"]
    assert b["rows"][2]["why"] == "non-loopback peer"
    assert b["summary"]["admit"] is False and b["jtr1_frames"] == []
    assert transcript_key("")["summary"]["admit"] is False
    print("selftest ok key=%s rate=%.2f ports=%s" % (g["key"], RATE, ",".join(PORTS)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--transcript")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    text = open(a.transcript, encoding="utf-8").read() if a.transcript else GOOD
    print(json.dumps(transcript_key(text, a.prev), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
