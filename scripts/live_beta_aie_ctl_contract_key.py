"""Live beta test_junior_aie + juniorctl health/ask/night contract key (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 1
(`PYTHONPATH=. python rails/linux/juniorctl.py health`), step 2
(`juniorctl ask` / `night`) and step 3 (`python tests/test_junior_aie.py`);
docs/COMPILED_BACKLOG.md "A - live beta suite". Quant Chief pair A, slot
2026-10-18 05:05 MT. Trit Pack & I2_S side: the per-test and per-contract trit
pack and its I2_S key. Winsor & Quant Error covers the quant-error bound side of
the same item in a different file.

New capability (not in scripts/live_beta_gate_jtr1.py, which packs one coarse
pass/fail trit per gate, nor BitNet-mlx scripts/live_beta_steps45_i2s_key.py,
which covers steps 4-5): a fine-grained contract audit of *captured outputs*.
  * test_junior_aie: a verbose unittest transcript (`-v` output saved to a file)
    becomes one trit per test case (ok +1, skipped/expected failure 0,
    FAIL/ERROR -1) plus a summary-consistency trit ("Ran N tests" vs the
    per-case lines and the OK/FAILED footer).
  * juniorctl health JSON: bitnetd bound to 127.0.0.1:8765, no wildcard bind, the
    cmds list advertises health / ask <q> / night, ports list non-empty.
  * juniorctl ask JSON: ok is a bool; ok -> non-empty text; not ok -> reasons.
  * juniorctl night JSON: an object, no wildcard bind anywhere in it.
It never runs juniorctl, the tests, a subprocess, or the network; it reads only
local files you point it at (or the built-in selftest fixtures). It does not
wrap, test, or re-derive any other team file.

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  JTR1 codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected)
  gamma = mean |trit| (share of trits that were actually decided)
I2_S block: code = trit + 1 (-1 -> 0, 0 -> 1, +1 -> 2), four per byte MSB first,
  padded with code 1 (zero) to a 128-trit block, as BitNet I2_S stores ternary.

Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16), pack = full JTR1 bytes, text = canonical JSON of the contract.
Rate 0.05: a green verdict needs zero -1 trits and at most 5% undecided (0)
trits; otherwise the verdict is "hold".
Compatible engines: JuniorLLM, BitNet-mlx, i2sd userspace, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage:
  python3 scripts/live_beta_aie_ctl_contract_key.py --selftest
  python3 scripts/live_beta_aie_ctl_contract_key.py [--aie-log P] [--health P]
      [--ask P] [--night P] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
BITNETD = "127.0.0.1:8765"
WILDCARD = ".".join(["0"] * 4)  # deny-check only; never bound
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM", "BitNet-mlx", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md Live beta steps 1-3: juniorctl health/ask/night "
    "+ tests/test_junior_aie.py (COMPILED_BACKLOG A)"
)
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
REQUIRED_CMDS = ("health", "ask <q>", "night")
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}

_CASE = re.compile(r"^(test\w*) \(([\w.]+)\)(?:\s*\n?.*?)? \.\.\. (.+)$")
_RAN = re.compile(r"^Ran (\d+) tests? in ")


# ---------------------------------------------------------------- parsing
def parse_unittest_verbose(text: str) -> tuple[list[tuple[str, int]], int]:
    """Return ([(case_id, trit)], summary_trit) from a `-v` unittest transcript."""
    cases: list[tuple[str, int]] = []
    ran = None
    footer = None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = _CASE.match(line)
        if m:
            name, owner, status = m.group(1), m.group(2), m.group(3).strip().lower()
            parts = owner.split(".")
            cls = parts[-2] if len(parts) > 1 and parts[-1] == name else parts[-1]
            cid = f"{cls}.{name}"
            if status == "ok":
                t = 1
            elif status.startswith("skipped") or status == "expected failure":
                t = 0
            else:  # FAIL, ERROR, unexpected success
                t = -1
            cases.append((cid, t))
            continue
        r = _RAN.match(line)
        if r:
            ran = int(r.group(1))
            continue
        if line == "OK" or line.startswith("OK ("):
            footer = "OK"
        elif line.startswith("FAILED"):
            footer = "FAILED"
    if ran is None or footer is None:
        summary = 0
    else:
        any_bad = any(t < 0 for _, t in cases)
        consistent = ran == len(cases) and ((footer == "OK") != any_bad)
        summary = 1 if consistent and footer == "OK" else -1
    return cases, summary


def _has_wildcard(obj) -> bool:
    return WILDCARD in json.dumps(obj, sort_keys=True, default=str)


def health_trits(h) -> list[tuple[str, int]]:
    if h is None:
        return [("health.bind", 0), ("health.no_wildcard", 0), ("health.cmds", 0), ("health.ports", 0)]
    if not isinstance(h, dict):
        return [("health.bind", -1), ("health.no_wildcard", -1), ("health.cmds", -1), ("health.ports", -1)]
    cmds = h.get("cmds") or []
    ports = h.get("ports") or []
    return [
        ("health.bind", 1 if h.get("bitnetd") == BITNETD else -1),
        ("health.no_wildcard", -1 if _has_wildcard(h) else 1),
        ("health.cmds", 1 if all(c in cmds for c in REQUIRED_CMDS) else -1),
        ("health.ports", 1 if isinstance(ports, list) and len(ports) > 0 else -1),
    ]


def ask_trits(a) -> list[tuple[str, int]]:
    if a is None:
        return [("ask.shape", 0), ("ask.body", 0)]
    if not isinstance(a, dict) or not isinstance(a.get("ok"), bool):
        return [("ask.shape", -1), ("ask.body", -1)]
    if a["ok"]:
        body = isinstance(a.get("text"), str) and a["text"].strip() != ""
    else:
        body = isinstance(a.get("reasons"), list) and len(a["reasons"]) > 0
    return [("ask.shape", 1), ("ask.body", 1 if body else -1)]


def night_trits(n) -> list[tuple[str, int]]:
    if n is None:
        return [("night.shape", 0), ("night.no_wildcard", 0)]
    return [
        ("night.shape", 1 if isinstance(n, dict) else -1),
        ("night.no_wildcard", -1 if _has_wildcard(n) else 1),
    ]


# ---------------------------------------------------------------- packing
def pack_jtr1(trits: list[int]) -> bytes:
    n = len(trits)
    gamma = (sum(abs(t) for t in trits) / n) if n else 0.0
    out = bytearray(MAGIC + struct.pack("<fI", gamma, n))
    for i in range(0, n, 4):
        b = 0
        chunk = trits[i:i + 4]
        for j in range(4):
            code = _JTR[chunk[j]] if j < len(chunk) else 0
            b |= code << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def unpack_jtr1(blob: bytes) -> tuple[float, list[int]]:
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    trits: list[int] = []
    for k in range(n):
        code = (blob[12 + k // 4] >> (6 - 2 * (k % 4))) & 3
        if code == 3:
            raise ValueError("code 0b11 rejected")
        trits.append(_JTR_DEC[code])
    return gamma, trits


def pack_i2s(trits: list[int]) -> bytes:
    n = len(trits)
    padded = n + ((-n) % I2S_BLOCK) if n else I2S_BLOCK
    codes = [t + 1 for t in trits] + [1] * (padded - n)
    out = bytearray()
    for i in range(0, padded, 4):
        b = 0
        for j in range(4):
            b |= codes[i + j] << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def close_key(prev: str, pack: bytes, text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits: list[int]) -> str:
    if not trits or any(t < 0 for t in trits):
        return "hold"
    undecided = sum(1 for t in trits if t == 0) / len(trits)
    return "green" if undecided <= RATE else "hold"


def build(aie_text, health, ask, night, prev: str = GENESIS) -> dict:
    cases, summary = parse_unittest_verbose(aie_text) if aie_text is not None else ([], 0)
    rows = [("aie." + c, t) for c, t in cases] + [("aie.summary", summary)]
    rows += health_trits(health) + ask_trits(ask) + night_trits(night)
    trits = [t for _, t in rows]
    pack = pack_jtr1(trits)
    i2s = pack_i2s(trits)
    contract = {
        "item": ROADMAP_ITEM,
        "rows": rows,
        "verdict": verdict(trits),
        "rate": RATE,
        "i2s_sha": hashlib.sha256(i2s).hexdigest()[:16],
        "bind": BIND,
        "ports": list(PORTS),
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["key"] = close_key(prev, pack, text)
    contract["prev"] = prev
    contract["n"] = len(trits)
    contract["engines"] = list(ENGINES)
    return contract


# ---------------------------------------------------------------- selftest
_AIE_GREEN = """test_ask_and_cache (__main__.FrameworkTests.test_ask_and_cache) ... ok
test_guard (__main__.FrameworkTests.test_guard) ... ok
test_router (__main__.FrameworkTests.test_router) ... ok
test_orch (__main__.StackTests.test_orch) ... ok
test_mcp (__main__.StackTests.test_mcp) ... ok
test_consensus_escalate (__main__.StackTests.test_consensus_escalate) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.012s

OK
"""
_AIE_RED = """test_ask_and_cache (__main__.FrameworkTests.test_ask_and_cache) ... ok
test_guard (__main__.FrameworkTests.test_guard) ... FAIL
test_router (__main__.FrameworkTests.test_router) ... skipped 'no router'

----------------------------------------------------------------------
Ran 3 tests in 0.004s

FAILED (failures=1, skipped=1)
"""
_HEALTH = {"product": "JuniorOS overlay", "bitnetd": BITNETD,
           "ports": ["home", "osai"], "cmds": ["health", "port list", "ask <q>", "night", "security"]}
_ASK_OK = {"ok": True, "text": "[JuniorAstraReason] red feather granite", "visibility": "public"}
_ASK_GUARD = {"ok": False, "reasons": ["injection"]}
_NIGHT = {"ticks": 32, "done": True, "bind": BITNETD}


def selftest() -> str:
    assert RATE == 0.05 and BIND == "127.0.0.1"
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert set(PORTS) == {"Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"}
    # parser
    cases, summary = parse_unittest_verbose(_AIE_GREEN)
    assert len(cases) == 6 and all(t == 1 for _, t in cases) and summary == 1
    assert cases[0][0] == "FrameworkTests.test_ask_and_cache"
    cases, summary = parse_unittest_verbose(_AIE_RED)
    assert [t for _, t in cases] == [1, -1, 0] and summary == -1
    assert parse_unittest_verbose("garbage")[1] == 0
    # lying footer: says OK but a case failed
    lie = _AIE_RED.replace("FAILED (failures=1, skipped=1)", "OK")
    assert parse_unittest_verbose(lie)[1] == -1
    # contracts
    assert [t for _, t in health_trits(_HEALTH)] == [1, 1, 1, 1]
    bad = dict(_HEALTH, bitnetd=WILDCARD + ":8765")
    assert [t for _, t in health_trits(bad)] == [-1, -1, 1, 1]
    assert [t for _, t in health_trits(None)] == [0, 0, 0, 0]
    assert [t for _, t in ask_trits(_ASK_OK)] == [1, 1]
    assert [t for _, t in ask_trits(_ASK_GUARD)] == [1, 1]
    assert [t for _, t in ask_trits({"ok": False})] == [1, -1]
    assert [t for _, t in ask_trits({"ok": "yes"})] == [-1, -1]
    assert [t for _, t in night_trits(_NIGHT)] == [1, 1]
    assert [t for _, t in night_trits({"bind": WILDCARD + ":1"})] == [1, -1]
    # JTR1 round trip incl. partial byte
    for vec in ([1, 0, -1], [1, -1, 0, 1, -1], [], [0] * 9):
        g, back = unpack_jtr1(pack_jtr1(vec))
        assert back == vec
        if vec:
            assert abs(g - sum(abs(t) for t in vec) / len(vec)) < 1e-6
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 1.0, 1) + b"\xc0")
        raise AssertionError("0b11 accepted")
    except ValueError:
        pass
    # I2_S block
    blk = pack_i2s([-1, 0, 1, 1])
    assert len(blk) == I2S_BLOCK // 4 and blk[0] == 0b00011010 and blk[1] == 0b01010101
    # rate rule: 1 undecided out of 20 is exactly 0.05 -> green; 2/20 -> hold
    assert verdict([1] * 19 + [0]) == "green"
    assert verdict([1] * 18 + [0, 0]) == "hold"
    assert verdict([1] * 19 + [-1]) == "hold"
    # close rule
    pk = pack_jtr1([1, 1])
    want = hashlib.sha256(GENESIS.encode() + pk + b"t").hexdigest()[:16]
    assert close_key(GENESIS, pk, "t") == want
    try:
        close_key("xyz", pk, "t")
        raise AssertionError("bad prev accepted")
    except ValueError:
        pass
    # full builds
    g1 = build(_AIE_GREEN, _HEALTH, _ASK_OK, _NIGHT)
    g2 = build(_AIE_GREEN, _HEALTH, _ASK_OK, _NIGHT)
    assert g1["verdict"] == "green" and g1["n"] == 6 + 1 + 4 + 2 + 2
    assert g1["key"] == g2["key"]
    chained = build(_AIE_GREEN, _HEALTH, _ASK_GUARD, _NIGHT, prev=g1["key"])
    assert chained["key"] != g1["key"] and chained["verdict"] == "green"
    red = build(_AIE_RED, bad, _ASK_OK, None)
    assert red["verdict"] == "hold" and red["key"] != g1["key"]
    empty = build(None, None, None, None)
    assert empty["verdict"] == "hold"
    return g1["key"]


def _load_json(path):
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--aie-log")
    ap.add_argument("--health")
    ap.add_argument("--ask")
    ap.add_argument("--night")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        key = selftest()
        print(json.dumps({"selftest": "pass", "key": key, "rate": RATE,
                          "ports": list(PORTS), "engines": list(ENGINES)}))
        return 0
    aie = None
    if a.aie_log:
        with open(a.aie_log, encoding="utf-8") as f:
            aie = f.read()
    out = build(aie, _load_json(a.health), _load_json(a.ask), _load_json(a.night), prev=a.prev)
    print(json.dumps(out, sort_keys=True))
    return 0 if out["verdict"] == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
