"""Live beta gate trit pack (JTR1) + I2_S gate key. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" (steps 1-3:
`juniorctl health` / `ask` / `night` / `security`, `tests/test_junior_aie.py`,
`tests/test_linux_security.py`) and docs/COMPILED_BACKLOG.md "A - live beta suite".
Quant Chief pair A, slot 2026-10-16 14:05 MT. Trit Pack & I2_S side: the gate
vector pack and its I2_S key. Winsor & Quant Error covers the quant-error bound
side of the same item in a different file.

New capability: a six-gate live-beta verdict vector, one balanced trit per gate
(+1 pass, 0 not run, -1 fail), packed as a JTR1 record and keyed with I2_S
2-bit codes under the close rule. It does not run juniorctl or the tests, and it
does not wrap, test, or re-derive any other team file. Results come only from an
optional local JSON file ({"health": "pass", ...}) or the built-in selftest.

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected)
  gamma = passed gates / n (absmean of the vector's positive mass)

Close rule: sha256(prev || pack || text)[:16], pack = full JTR1 bytes.
Rate 0.05: beta cohort ramp step; a fully green gate vector releases exactly
0.05 of the cohort, anything else releases 0 (held).
Compatible engines: JuniorLLM, BitNet-mlx, i2sd userspace, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage: python3 scripts/live_beta_gate_jtr1.py [--selftest] [--results PATH] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM", "BitNet-mlx", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl health/ask/night/security "
    "+ test_junior_aie + test_linux_security (COMPILED_BACKLOG A)"
)
MAGIC = b"JTR1"
GENESIS = "0" * 16
GATES = (
    "health",
    "ask",
    "night",
    "security",
    "test_junior_aie",
    "test_linux_security",
)
_VERDICT = {"pass": 1, "ok": 1, "skip": 0, "none": 0, "not_run": 0, "fail": -1, "error": -1}
_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def to_trit(v) -> int:
    if isinstance(v, bool):
        return 1 if v else -1
    if isinstance(v, int) and v in _ENC:
        return v
    if v is None:
        return 0
    key = str(v).strip().lower()
    if key not in _VERDICT:
        raise ValueError("unknown verdict: %r" % (v,))
    return _VERDICT[key]


def gate_vector(results: dict) -> list:
    """Order results by GATES; missing gates are 0 (not run). Unknown keys rejected."""
    extra = set(results) - set(GATES)
    if extra:
        raise ValueError("unknown gates: %s" % ",".join(sorted(extra)))
    return [to_trit(results.get(g)) for g in GATES]


def pack_codes(trits) -> bytes:
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


def unpack_codes(data: bytes, n: int) -> list:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            c = (b >> s) & 3
            if c == 3:
                raise ValueError("code 0b11 is invalid")
            out.append(_DEC[c])
    if len(out) < n:
        raise ValueError("payload short")
    return out[:n]


def gamma_of(trits) -> float:
    return sum(1 for t in trits if t == 1) / len(trits) if trits else 0.0


def jtr1_pack(trits) -> bytes:
    trits = list(trits)
    return MAGIC + struct.pack("<fI", gamma_of(trits), len(trits)) + pack_codes(trits)


def jtr1_unpack(blob: bytes) -> dict:
    if len(blob) < 12 or blob[:4] != MAGIC:
        raise ValueError("not JTR1")
    gamma, n = struct.unpack("<fI", blob[4:12])
    payload = blob[12:]
    if len(payload) != (n + 3) // 4:
        raise ValueError("payload length mismatch")
    return {"gamma": gamma, "n": n, "trits": unpack_codes(payload, n)}


def close_rule(prev: str, pack: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def release_fraction(trits) -> float:
    """0.05 of the beta cohort only when every gate is +1; otherwise 0 (held)."""
    return RATE if trits and all(t == 1 for t in trits) else 0.0


def gate_key(results: dict, prev: str = GENESIS) -> dict:
    trits = gate_vector(results)
    pack = jtr1_pack(trits)
    text = "live_beta|" + ",".join(GATES) + "|%.2f" % RATE
    tip = close_rule(prev, pack, text)
    i2s_key = pack[12:] + bytes.fromhex(tip)
    return {
        "item": ROADMAP_ITEM,
        "gates": dict(zip(GATES, trits)),
        "green": all(t == 1 for t in trits),
        "gamma": round(gamma_of(trits), 6),
        "jtr1_hex": pack.hex(),
        "i2s_key_hex": i2s_key.hex(),
        "prev": prev,
        "close": tip,
        "rate": RATE,
        "release": release_fraction(trits),
        "held": release_fraction(trits) == 0.0,
        "ports": list(PORTS),
        "engines": list(ENGINES),
        "bind": BIND,
        "mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
        "fetch": FETCH, "ue5_launch": UE5_LAUNCH,
    }


def load_results(path: str) -> dict:
    if not path or "://" in path:
        raise ValueError("results must be a local file path")
    if not os.path.isfile(path):
        raise ValueError("results file missing: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("results must be a JSON object")
    return data


def selftest() -> int:
    assert abs(RATE - 0.05) < 1e-12
    assert BIND == "127.0.0.1"
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert set(PORTS) == {"Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"}

    # code round trip on every trit pattern of length 6
    import itertools
    for combo in itertools.product((-1, 0, 1), repeat=len(GATES)):
        blob = jtr1_pack(combo)
        back = jtr1_unpack(blob)
        assert back["trits"] == list(combo) and back["n"] == 6
        assert len(blob) == 12 + 2

    green = {g: "pass" for g in GATES}
    a = gate_key(green)
    assert a["green"] and a["release"] == 0.05 and not a["held"]
    assert abs(a["gamma"] - 1.0) < 1e-6
    pack = bytes.fromhex(a["jtr1_hex"])
    text = "live_beta|" + ",".join(GATES) + "|0.05"
    want = hashlib.sha256(GENESIS.encode() + pack + text.encode()).hexdigest()[:16]
    assert a["close"] == want and len(a["close"]) == 16
    assert a["i2s_key_hex"].endswith(want)

    # chaining: prev feeds the next close
    b = gate_key(green, prev=a["close"])
    assert b["close"] != a["close"] and b["prev"] == a["close"]

    # one failing security gate holds the ramp
    red = dict(green, security="fail")
    r = gate_key(red)
    assert r["gates"]["security"] == -1 and r["held"] and r["release"] == 0.0
    assert r["close"] != a["close"]

    # missing gate is not-run (0), still held
    partial = {g: "pass" for g in GATES[:4]}
    p = gate_key(partial)
    assert p["gates"]["test_linux_security"] == 0 and p["held"]
    assert abs(p["gamma"] - 4 / 6) < 1e-6

    # rejects
    for bad in ({"bogus": "pass"}, {"health": "maybe"}):
        try:
            gate_key(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected reject")
    try:
        jtr1_unpack(b"GGUF" + b"\x00" * 12)
    except ValueError:
        pass
    else:
        raise AssertionError("GGUF accepted as JTR1")
    try:
        unpack_codes(b"\xff", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("0b11 accepted")
    try:
        load_results("https://example.invalid/r.json")
    except ValueError:
        pass
    else:
        raise AssertionError("remote results accepted")

    # local results file path
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "r.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(green, f)
        assert gate_key(load_results(fp))["close"] == a["close"]

    print(json.dumps({"selftest": "pass", "close": a["close"], "rate": RATE,
                      "engines": list(ENGINES), "ports": list(PORTS)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--results", help="local JSON of gate verdicts")
    ap.add_argument("--prev", default=GENESIS)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if len(args.prev) != 16:
        ap.error("--prev must be 16 hex chars")
    results = load_results(args.results) if args.results else {}
    print(json.dumps(gate_key(results, prev=args.prev), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
