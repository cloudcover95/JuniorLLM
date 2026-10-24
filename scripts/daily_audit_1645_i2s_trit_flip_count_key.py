#!/usr/bin/env python3
"""16:45 daily audit: day-over-day I2_S trit flip count, split by kind, chained JTR1 key.

Roadmap anchor: JuniorHome docs/ROADMAP.md line 7, Automations (current) ON,
16:45 daily audit. Spec: JuniorLLM 7401889 (osai vault finding
2026-10-24_0255, "day-over-day trit flip count on I2_S packs, XOR+popcount").

Chief assignment 2026-10-24 05:05 MT, pair A: I2_S buddy file for Winsor's
JuniorQuant src/daily_audit_1645_f32_gamma_receipt.py @ 55ac5fb. Winsor's
file owns the f32 gamma scalars (absmean, junior_drop, p95) and the quant
error bound. This file owns the I2_S side: it keeps each day's ternary
weights only as a JTR1 I2_S pack and counts how many trits flipped since
yesterday straight from the two packs, with XOR + popcount, in pure stdlib.
It references 55ac5fb by name and sha and never writes into JuniorQuant.

New capability over the 7401889 note (which used numpy and only a total):
one XOR pass splits the flips by kind, answering that note's "next question":
  codes: 0 -> -1 (0b00), 1 -> 0 (0b01), 2 -> +1 (0b10); 0b11 is invalid.
  per 2-bit field x = a ^ b:
    lo bit set           -> zero crossing (a change to or from 0)
    hi set and lo clear  -> sign flip (-1 <-> +1)
  total = popcount((x | x >> 1) & M), zero = popcount(x & M),
  sign = popcount((x >> 1) & ~x & M), with M = 0x5555... over the pack.

JTR1 frame (not GGUF): b"JTR1" + gamma f32 LE + n u32 LE + 2-bit payload,
four codes per byte, low bits first, n padded with code 1 (trit 0).

Rate 0.05: the synthetic daily drift re-draws each weight with probability
0.05, so roughly that share of trits can move each day.
Close rule: key = sha256(prev || pack || text)[:16], chained day to day.

Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Flags: bind 127.0.0.1 only (nothing is bound); mint/socket/layer_one/fetch/
ue5_launch false. No network, no model pull, no GGUF read, stdlib only.

Usage: python3 daily_audit_1645_i2s_trit_flip_count_key.py --selftest
       python3 daily_audit_1645_i2s_trit_flip_count_key.py [YYYY-MM-DD] [days]
"""
import datetime as dt
import hashlib
import json
import random
import struct
import sys

RATE = 0.05
BIND = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
MAGIC = b"JTR1"
GENESIS = "0" * 16
N = 4096
WINSOR_REF = {"repo": "JuniorQuant",
              "path": "src/daily_audit_1645_f32_gamma_receipt.py",
              "sha": "55ac5fb", "gammas": ["absmean", "junior_drop", "p95"]}
SPEC_REF = "JuniorLLM 7401889"
ROADMAP = "JuniorHome docs/ROADMAP.md line 7: 16:45 daily audit"


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def ternarize(w):
    """BitNet absmean rule: gamma = mean|W| (as f32), t = clip(round(W/gamma))."""
    gamma = f32(sum(abs(v) for v in w) / len(w))
    eps = 1e-8
    out = []
    for v in w:
        q = round(v / (gamma + eps))
        out.append(-1 if q < -1 else (1 if q > 1 else int(q)))
    return gamma, out


def pack(gamma, trits):
    n = len(trits)
    codes = [t + 1 for t in trits]
    codes += [1] * ((-n) % 4)
    body = bytearray()
    for i in range(0, len(codes), 4):
        c = codes[i:i + 4]
        body.append(c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6))
    return MAGIC + struct.pack("<fI", gamma, n) + bytes(body)


def unpack(blob):
    if len(blob) < 12 or blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    body = blob[12:]
    if len(body) != (n + 3) // 4:
        raise ValueError("short payload")
    trits = []
    for i in range(n):
        c = (body[i // 4] >> (2 * (i % 4))) & 3
        if c == 3:
            raise ValueError("invalid code 3")
        trits.append(c - 1)
    for i in range(n, len(body) * 4):
        if (body[i // 4] >> (2 * (i % 4))) & 3 != 1:
            raise ValueError("bad padding")
    return gamma, trits


def flip_counts(pa, pb):
    """XOR + popcount on two JTR1 packs: (total, sign, zero)."""
    unpack(pa), unpack(pb)  # validate both frames
    if pa[8:12] != pb[8:12]:
        raise ValueError("n mismatch")
    a = int.from_bytes(pa[12:], "little")
    b = int.from_bytes(pb[12:], "little")
    nbits = 8 * (len(pa) - 12)
    m = int("01" * (nbits // 2), 2) if nbits else 0
    x = a ^ b
    total = ((x | (x >> 1)) & m).bit_count()
    zero = (x & m).bit_count()
    sign = ((x >> 1) & ~x & m).bit_count()
    return total, sign, zero


def flip_counts_naive(ta, tb):
    total = sign = zero = 0
    for u, v in zip(ta, tb):
        if u != v:
            total += 1
            if u == 0 or v == 0:
                zero += 1
            else:
                sign += 1
    return total, sign, zero


def close(prev, blob, text):
    h = hashlib.sha256(prev.encode() + blob + text.encode()).hexdigest()
    return h[:16]


def slot_utc(day):
    """16:45 America/Denver on `day` as UTC, US DST rule (stdlib, no tzdata)."""
    y = day.year
    mar1 = dt.date(y, 3, 1)
    dst_start = mar1 + dt.timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = dt.date(y, 11, 1)
    dst_end = nov1 + dt.timedelta(days=(6 - nov1.weekday()) % 7)
    off = 6 if dst_start <= day < dst_end else 7
    t = dt.datetime(y, day.month, day.day, 16, 45) + dt.timedelta(hours=off)
    return t.strftime("%Y-%m-%dT%H:%MZ")


def day_weights(start, ndays, n=N):
    """Date-seeded weights; each day re-draws each weight with prob RATE."""
    rng = random.Random("jtr1-flip|" + start.isoformat())
    w = [rng.gauss(0.0, 0.02) for _ in range(n)]
    out = [list(w)]
    for _ in range(1, ndays):
        w = [rng.gauss(0.0, 0.02) if rng.random() < RATE else v for v in w]
        out.append(list(w))
    return out


def run(start, ndays, n=N, tamper=None):
    prev, prev_blob, rows = GENESIS, None, []
    for i, w in enumerate(day_weights(start, ndays, n)):
        day = start + dt.timedelta(days=i)
        gamma, trits = ternarize(w)
        blob = pack(gamma, trits)
        if tamper is not None and i == tamper:
            blob = blob[:12] + bytes([blob[12] ^ 0b10]) + blob[13:]
        if prev_blob is None:
            total = sign = zero = 0
        else:
            total, sign, zero = flip_counts(prev_blob, blob)
        row = {"day": day.isoformat(), "slot_utc": slot_utc(day), "n": n,
               "gamma_absmean_f32": gamma, "flips": total, "sign": sign,
               "zero": zero, "pack_sha": hashlib.sha256(blob).hexdigest()[:16],
               "winsor_ref": WINSOR_REF, "spec": SPEC_REF, "rate": RATE}
        text = json.dumps(row, sort_keys=True, separators=(",", ":"))
        prev = close(prev, blob, text)
        row["key"] = prev
        rows.append(row)
        prev_blob = blob
    return rows


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and bool(cond)

    check("rate 0.05", RATE == 0.05)
    check("bind 127.0.0.1", BIND == "127.0.0.1")
    check("flags all false", not any(FLAGS.values()))
    check("ports", PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"))
    check("names 55ac5fb", WINSOR_REF["sha"] == "55ac5fb")
    # roundtrip, including padding
    for n in (1, 3, 4, 7, 64):
        rng = random.Random(n)
        t = [rng.choice((-1, 0, 1)) for _ in range(n)]
        g, back = unpack(pack(f32(0.5), t))
        check(f"roundtrip n={n}", back == t and g == f32(0.5))
    # every pair of codes maps to the right kind
    for u in (-1, 0, 1):
        for v in (-1, 0, 1):
            got = flip_counts(pack(1.0, [u]), pack(1.0, [v]))
            check(f"kind {u}->{v}", got == flip_counts_naive([u], [v]))
    # 30 days of drift: packed counts equal naive counts
    start = dt.date(2026, 10, 1)
    ws = day_weights(start, 30, 1024)
    ts = [ternarize(w)[1] for w in ws]
    ps = [pack(ternarize(w)[0], t) for w, t in zip(ws, ts)]
    same = all(flip_counts(ps[i - 1], ps[i]) == flip_counts_naive(ts[i - 1], ts[i])
               for i in range(1, 30))
    check("30-day packed == naive", same)
    tot = [flip_counts(ps[i - 1], ps[i]) for i in range(1, 30)]
    check("sign + zero == total", all(s + z == t for t, s, z in tot))
    check("flips happen and stay under 2x rate",
          all(0 < t < 2 * RATE * 1024 for t, _, _ in tot))
    check("pack is 4x smaller than int8", len(ps[0]) - 12 == 1024 // 4)
    # tamper and frame errors
    for name, blob in (("bad magic", b"JTR0" + ps[0][4:]),
                       ("short", ps[0][:-1]),
                       ("code 3", ps[0][:12] + b"\xff" + ps[0][13:])):
        try:
            unpack(blob)
            check("reject " + name, False)
        except ValueError:
            check("reject " + name, True)
    try:
        flip_counts(ps[0], pack(1.0, [0] * 1028))
        check("reject n mismatch", False)
    except ValueError:
        check("reject n mismatch", True)
    # DST slots
    check("slot MDT", slot_utc(dt.date(2026, 10, 24)) == "2026-10-24T22:45Z")
    check("slot MST", slot_utc(dt.date(2026, 11, 2)) == "2026-11-02T23:45Z")
    # chained keys: deterministic and tamper-sensitive
    a = run(start, 7)
    b = run(start, 7)
    c = run(start, 7, tamper=3)
    check("keys deterministic", [r["key"] for r in a] == [r["key"] for r in b])
    check("tamper changes day 3 on", all(a[i]["key"] != c[i]["key"] for i in range(3, 7))
          and all(a[i]["key"] == c[i]["key"] for i in range(3)))
    check("close rule", a[0]["key"] == close(GENESIS, pack(*ternarize(day_weights(start, 1)[0])),
                                             json.dumps({k: v for k, v in a[0].items() if k != "key"},
                                                        sort_keys=True, separators=(",", ":"))))
    print("selftest key", a[-1]["key"])
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    args = [a for a in argv if not a.startswith("-")]
    start = dt.date.fromisoformat(args[0]) if args else dt.date.today()
    ndays = int(args[1]) if len(args) > 1 else 7
    for row in run(start, ndays):
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
