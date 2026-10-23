#!/usr/bin/env python3
"""16:45 daily audit: 3-day I2_S word pack with in-word and cross-word backfill flags.

Roadmap anchor: JuniorHome docs/ROADMAP.md, Automations ON, "16:45 daily audit"
(writes docs/AUTOMATION_AUDIT.md). Chief pair A, slot 2026-10-22 18:05 MT,
"remaining engines only": the JuniorLLM engine had a vault finding for this item
(f483bf0) but no packing code. This file is the engine-side I2_S/trit pack piece.

New capability (answers the "Next question" of the 14:55 vault note): pack the
10 field trits per audit day three days to one 64-bit I2_S word (30 trit slots
plus 2 zero pad slots, 0->0, 1->1, -1->2, MSB first), then compute the per-day
failed flag and the +1 -> -1 regression flag with SWAR masks directly on the
packed words. Day d vs d+1 inside a word is a 20-bit shift; day 2 of word k vs
day 0 of word k+1 is the cross-word carry (a 40-bit shift). Flags are checked
exactly against a naive per-trit reference. The word stream is framed as JTR1
(magic b"JTR1" + gamma f64 + n u32 + 2-bit payload; n counts trit slots
including pad, gamma = mean |t| of the real trits; not GGUF) and closed with
sha256(prev || pack || text)[:16] per word, then once over the frame.

Field order matches the 16:45 audit receipt fields: time, tip, audit_row,
overnight, slot19, weekly, paused_absent, paused_note, no_actions, tz.
Input is read-only interop: a JSON list of 10-trit lists, or a ledger JSON
with days[].fields (name -> trit). Nothing is fetched, written, or bound.

Rate 0.05 (pass level held by level + 0.05 * (x - level) per day).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Compatible engines: JuniorLLM, BitNet-mlx (I2_S 2-bit layout), i2sd userspace.
Bind 127.0.0.1 only (no socket opened). mint/socket/layer_one/fetch/ue5_launch false.
Stdlib only. Run: python3 daily_audit_1645_word3_i2s_key.py --selftest
"""
import argparse
import hashlib
import json
import struct
import sys

RATE = 0.05
BIND = "127.0.0.1"
MINT = SOCKET = LAYER_ONE = FETCH = UE5_LAUNCH = False
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM", "BitNet-mlx", "i2sd userspace")
GENESIS = "0" * 16
MAGIC = b"JTR1"
FIELDS = ("time", "tip", "audit_row", "overnight", "slot19", "weekly",
          "paused_absent", "paused_note", "no_actions", "tz")
F = len(FIELDS)          # 10 trits per day
DPW = 3                  # days per 64-bit word
SLOTS = 32               # 2-bit slots per word (30 used + 2 pad)
_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def _bit(slot):
    return 62 - 2 * slot   # low bit position of a slot, MSB-first slots


M_LO = sum(1 << _bit(s) for s in range(F * DPW))
DAY_MASK = [sum(1 << _bit(j * F + i) for i in range(F)) for j in range(DPW)]
SHIFT_DAY = 2 * F        # 20 bits between consecutive days


def close_rule(prev, pk, text):
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def rate_step(level, x):
    return level + RATE * (x - level)


def pack_words(days):
    words = []
    for k in range(0, len(days), DPW):
        w = 0
        for j, day in enumerate(days[k:k + DPW]):
            if len(day) != F:
                raise ValueError("each day needs %d trits" % F)
            for i, t in enumerate(day):
                if t not in _ENC:
                    raise ValueError("trit must be -1, 0 or 1")
                w |= _ENC[t] << _bit(j * F + i)
        words.append(w)
    return words


def unpack_words(words, n_days):
    out = []
    for d in range(n_days):
        w = words[d // DPW]
        j = d % DPW
        out.append([_DEC[(w >> _bit(j * F + i)) & 3] for i in range(F)])
    return out


def swar_flags(words, n_days):
    """Per-day failed and regression flags straight from packed words."""
    fails, plus = [], []
    for w in words:
        lo = w & M_LO
        hi = (w >> 1) & M_LO
        fails.append(hi & ~lo)
        plus.append(lo & ~hi)
    failed, regress = [], []
    for k, w in enumerate(words):
        fa, pl = fails[k], plus[k]
        inword = pl & (fa << SHIFT_DAY) & (DAY_MASK[0] | DAY_MASK[1])
        carry = 0
        if k + 1 < len(words):
            carry = pl & (fails[k + 1] >> (SHIFT_DAY * 2)) & DAY_MASK[2]
        reg = inword | carry
        for j in range(DPW):
            d = k * DPW + j
            if d >= n_days:
                break
            if fa & DAY_MASK[j]:
                failed.append(d)
            if reg & DAY_MASK[j] and d + 1 < n_days:
                regress.append(d)
    return failed, regress


def naive_flags(days):
    failed = [d for d, r in enumerate(days) if -1 in r]
    regress = [d for d in range(len(days) - 1)
               if any(a == 1 and b == -1 for a, b in zip(days[d], days[d + 1]))]
    return failed, regress


def frame(words, real_trits):
    gamma = (sum(abs(t) for t in real_trits) / len(real_trits)) if real_trits else 0.0
    payload = b"".join(struct.pack(">Q", w) for w in words)
    return MAGIC + struct.pack(">dI", gamma, SLOTS * len(words)) + payload, gamma


def build_key(days, prev=GENESIS):
    n = len(days)
    words = pack_words(days)
    failed, regress = swar_flags(words, n)
    ref = naive_flags(days)
    exact = (failed, regress) == ref and unpack_words(words, n) == [list(r) for r in days]
    chain = prev
    for k, w in enumerate(words):
        chain = close_rule(chain, struct.pack(">Q", w), "w%d|1645" % k)
    flat = [t for r in days for t in r]
    fr, gamma = frame(words, flat)
    level = 1.0
    fset = set(failed)
    for d in range(n):
        level = rate_step(level, 0.0 if d in fset else 1.0)
    text = "1645|days=%d|failed=%d|regress=%d|exact=%s|rate=%.2f" % (
        n, len(failed), len(regress), exact, RATE)
    key = close_rule(chain, fr, text)
    return {"key": key, "chain": chain, "days": n, "words": len(words),
            "bytes_word3": 8 * len(words), "bytes_3b_per_day": 3 * n,
            "failed_days": failed, "regression_days": regress, "exact": exact,
            "gamma": round(gamma, 6), "pass_level": round(level, 6),
            "frame_head": fr[:16].hex(), "ports": list(PORTS),
            "engines": list(ENGINES), "bind": BIND}


def load_days(path):
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if isinstance(obj, dict) and "days" in obj:
        return [[int(d["fields"][n]) for n in FIELDS] for d in obj["days"]]
    return [[int(t) for t in r] for r in obj]


def _synth(n, seed=b"1645"):
    out, i = [], 0
    while len(out) < n:
        h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        i += 1
        row = []
        for b in h[:F]:
            row.append(-1 if b < 13 else (0 if b < 39 else 1))
        out.append(row)
    return out


def selftest():
    assert RATE == 0.05 and abs(rate_step(1.0, 0.0) - 0.95) < 1e-12
    assert BIND == "127.0.0.1"
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert close_rule(GENESIS, b"\x01", "x") == hashlib.sha256(b"0" * 16 + b"\x01" + b"x").hexdigest()[:16]
    one = [1] * F
    w = pack_words([one])[0]
    assert w == sum(1 << _bit(i) for i in range(F)) and w & 0xF == 0
    # in-word regression (day 0 -> 1), cross-word regression (day 2 -> 3)
    a = [1] + [0] * (F - 1)
    b = [-1] + [0] * (F - 1)
    z = [0] * F
    for days, want in (([a, b, z], ([1], [0])),
                       ([z, z, a, b], ([3], [2])),
                       ([z, a, b, z, z], ([2], [1])),
                       ([a], ([], [])), ([b], ([0], [])),
                       ([z, z, a], ([], [])), ([], ([], []))):
        words = pack_words(days)
        assert swar_flags(words, len(days)) == want == naive_flags(days), (days, want)
    for n in (1, 2, 3, 4, 5, 364, 365, 366, 3650):
        days = _synth(n)
        r = build_key(days)
        assert r["exact"], n
    r = build_key(_synth(3650))
    fr, _ = frame(pack_words([one]), one)
    assert fr[:4] == MAGIC and struct.unpack(">dI", fr[4:16]) == (1.0, 32) and len(fr) == 24
    assert r["bytes_word3"] == 8 * 1217 and r["bytes_3b_per_day"] == 10950
    print("selftest ok key=%s exact=%s days=%d failed=%d regress=%d bytes %d vs %d rate=%.2f ports=%s"
          % (r["key"], r["exact"], r["days"], len(r["failed_days"]), len(r["regression_days"]),
             r["bytes_word3"], r["bytes_3b_per_day"], RATE, ",".join(PORTS)))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--input", help="JSON list of 10-trit days or ledger JSON with days[].fields")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.input:
        ap.error("--input or --selftest required")
    print(json.dumps(build_key(load_days(a.input), a.prev), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
