"""Live beta step 6 guard key: no deletes, no >8GB pull, bitnetd 127.0.0.1:8765 only (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 6
("No deletes. No >8GB pull. bitnetd on `127.0.0.1:8765` only"), run end to end
on the JuniorOS overlay host; docs/COMPILED_BACKLOG.md "A - live beta suite".
Quant Chief pair A, slot 2026-10-18 14:05 MT (Verifier's known open gap: step 6).
Trit Pack & I2_S side: the step-6 exit-code trit pack and its I2_S key. Winsor &
Quant Error covers the quant-error bound side of the same item in a different file.

New capability (steps 1-5 already landed in scripts/live_beta_gate_jtr1.py,
scripts/live_beta_aie_ctl_contract_key.py and BitNet-mlx
scripts/live_beta_steps45_i2s_key.py; none of them audits step 6): three offline
guards over *captured* host transcripts, each turned into trits, plus
end-to-end exit-code trits for the capture commands and step 6 itself.
  * no deletes: a `git diff --name-status` / `git show --name-status` capture.
    One trit per path: A/M/C +1, T 0 (type change, undecided), D -1, and
    R -1 (a rename removes the old path). An empty diff is one +1 trit.
  * no >8GB pull: a size manifest, one "<bytes> <name>" per line (the shape of
    `du -b` output) for anything fetched this run. One trit per entry
    (<= 8e9 bytes +1, larger -1, unparseable 0) plus a total trit (sum <= 8e9).
  * bitnetd 127.0.0.1:8765 only: an `ss -ltnp` capture. Trits: bitnetd is
    listening on 127.0.0.1:8765; every bitnetd listener is exactly that address;
    nothing else holds port 8765; no wildcard listener (all-zeros IPv4, [::], *) for
    bitnetd anywhere.
  * exit codes: an optional rc log, "<label> <rc>" per line, for the capture
    commands (diff / sizes / ss). rc 0 -> +1, nonzero -> -1, missing -> 0.
    step6.rc is the derived step-6 exit code: +1 when every guard trit is +1,
    -1 when any is -1, else 0. The CLI exits 0 only on a green verdict.
It never runs git, ss, du, a subprocess, a socket, or the network; it reads only
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
  python3 scripts/live_beta_step6_guard_i2s_key.py --selftest
  python3 scripts/live_beta_step6_guard_i2s_key.py [--diff P] [--sizes P]
      [--ss P] [--rc-log P] [--prev HEX16]
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
BITNETD_PORT = 8765
BITNETD = f"{BIND}:{BITNETD_PORT}"
PULL_LIMIT = 8 * 10 ** 9  # "No >8GB pull"
WILDCARDS = (".".join(["0"] * 4), "[::]", "*", "::")  # deny-check only; never bound
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM", "BitNet-mlx", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md Live beta step 6: no deletes, no >8GB pull, "
    "bitnetd on 127.0.0.1:8765 only (COMPILED_BACKLOG A)"
)
RC_LABELS = ("diff", "sizes", "ss")
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}

_STATUS = re.compile(r"^([ACDMRTUX])(\d{0,3})\t(.+)$")
_SIZE = re.compile(r"^\s*(\d+)\s+(\S.*)$")
_PROC = re.compile(r'\("([^"]+)"')


# ---------------------------------------------------------------- guards
def diff_trits(text):
    """No deletes: one trit per path in a --name-status capture."""
    if text is None:
        return [("nodel.capture", 0)]
    rows = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        m = _STATUS.match(line)
        if not m:
            continue  # commit headers etc. in `git show` output
        code, path = m.group(1), m.group(3).split("\t")[0]
        if code in ("A", "M", "C"):
            t = 1
        elif code == "T":
            t = 0
        else:  # D, R (old path removed), U/X (unmerged/unknown)
            t = -1
        rows.append((f"nodel.{code}:{path}", t))
    return rows or [("nodel.empty", 1)]


def size_trits(text):
    """No >8GB pull: one trit per manifest entry plus a total trit."""
    if text is None:
        return [("pull.capture", 0)]
    rows = []
    total = 0
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _SIZE.match(raw)
        if not m:
            rows.append((f"pull.unparsed:{raw.strip()[:40]}", 0))
            continue
        b, name = int(m.group(1)), m.group(2).strip()
        total += b
        rows.append((f"pull.{name}", 1 if b <= PULL_LIMIT else -1))
    rows.append(("pull.total", 1 if total <= PULL_LIMIT else -1))
    return rows


def _split_addr(local):
    host, _, port = local.rpartition(":")
    if host.startswith("[") and host.endswith("]") and host != "[::]":
        host = host[1:-1]
    host = host.split("%")[0]
    return host, port


def parse_ss(text):
    """Return [(host, port, proc)] listeners from an `ss -ltnp` capture."""
    out = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 5 or parts[0] in ("State", "Netid"):
            continue
        # ss -ltnp: State Recv-Q Send-Q Local Peer [Process]; -a adds Netid first
        off = 1 if parts[0] in ("tcp", "udp") else 0
        if len(parts) < off + 5:
            continue
        local = parts[off + 3]
        if ":" not in local:
            continue
        host, port = _split_addr(local)
        rest = " ".join(parts[off + 5:])
        pm = _PROC.search(rest)
        out.append((host, port, pm.group(1) if pm else ""))
    return out


def bind_trits(text):
    if text is None:
        return [("bind.present", 0), ("bind.only", 0), ("bind.port_owner", 0), ("bind.no_wildcard", 0)]
    ls = parse_ss(text)
    bit = [(h, p) for h, p, proc in ls if proc == "bitnetd"]
    on_port = [(h, proc) for h, p, proc in ls if p == str(BITNETD_PORT)]
    present = (BIND, str(BITNETD_PORT)) in bit
    only = bool(bit) and all((h, p) == (BIND, str(BITNETD_PORT)) for h, p in bit)
    owner = bool(on_port) and all(proc == "bitnetd" and h == BIND for h, proc in on_port)
    wild = any(h in WILDCARDS for h, _ in bit) or any(h in WILDCARDS for h, _ in on_port)
    return [
        ("bind.present", 1 if present else -1),
        ("bind.only", 1 if only else -1),
        ("bind.port_owner", 1 if owner else -1),
        ("bind.no_wildcard", -1 if wild else 1),
    ]


def rc_trits(text):
    seen = {}
    if text is not None:
        for raw in text.splitlines():
            parts = raw.split()
            if len(parts) == 2 and parts[0] in RC_LABELS and re.fullmatch(r"-?\d+", parts[1]):
                seen[parts[0]] = int(parts[1])
    return [(f"rc.{lab}", 0 if lab not in seen else (1 if seen[lab] == 0 else -1)) for lab in RC_LABELS]


def step6_rc(guard_rows):
    ts = [t for _, t in guard_rows]
    if any(t < 0 for t in ts):
        return -1
    return 1 if ts and all(t == 1 for t in ts) else 0


# ---------------------------------------------------------------- packing
def pack_jtr1(trits):
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


def unpack_jtr1(blob):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    trits = []
    for k in range(n):
        code = (blob[12 + k // 4] >> (6 - 2 * (k % 4))) & 3
        if code == 3:
            raise ValueError("code 0b11 rejected")
        trits.append(_JTR_DEC[code])
    return gamma, trits


def pack_i2s(trits):
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


def close_key(prev, pack, text):
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits):
    if not trits or any(t < 0 for t in trits):
        return "hold"
    undecided = sum(1 for t in trits if t == 0) / len(trits)
    return "green" if undecided <= RATE else "hold"


def build(diff, sizes, ss, rc_log, prev=GENESIS):
    guards = diff_trits(diff) + size_trits(sizes) + bind_trits(ss)
    rows = guards + rc_trits(rc_log) + [("step6.rc", step6_rc(guards))]
    trits = [t for _, t in rows]
    pack = pack_jtr1(trits)
    i2s = pack_i2s(trits)
    contract = {
        "item": ROADMAP_ITEM,
        "rows": rows,
        "verdict": verdict(trits),
        "rate": RATE,
        "pull_limit": PULL_LIMIT,
        "bitnetd": BITNETD,
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
_W = WILDCARDS[0]
_DIFF_OK = "A\tscripts/new_tool.py\nM\tdocs/BETA_TO_OS.md\n"
_DIFF_BAD = "commit abc\n\nA\tscripts/x.py\nD\trails/linux/juniorctl.py\nR100\told.py\tnew.py\nT\tlink\n"
_SIZES_OK = "1187000000\tggml-model-i2_s.gguf.header-only\n4096\tos-release.junior\n"
_SIZES_BAD = "8000000001\tbig.bin\nnot-a-size\n"
_SS_OK = f"""State  Recv-Q Send-Q Local Address:Port Peer Address:Port Process
LISTEN 0      128    127.0.0.1:8765     {_W}:*          users:(("bitnetd",pid=411,fd=3))
LISTEN 0      128    127.0.0.1:8767     {_W}:*          users:(("i2sd",pid=412,fd=3))
"""
_SS_WILD = _SS_OK.replace("127.0.0.1:8765", _W + ":8765", 1)
_SS_EXTRA = _SS_OK + 'LISTEN 0 128 [::]:8765 [::]:* users:(("bitnetd",pid=411,fd=4))\n'
_SS_SQUAT = _SS_OK.replace('"bitnetd"', '"python3"', 1)
_RC_OK = "diff 0\nsizes 0\nss 0\n"


def selftest():
    assert RATE == 0.05 and BIND == "127.0.0.1" and BITNETD == "127.0.0.1:8765"
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert set(PORTS) == {"Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"}
    assert PULL_LIMIT == 8_000_000_000
    # no deletes
    assert [t for _, t in diff_trits(_DIFF_OK)] == [1, 1]
    assert [t for _, t in diff_trits(_DIFF_BAD)] == [1, -1, -1, 0]
    assert diff_trits("") == [("nodel.empty", 1)]
    assert diff_trits(None) == [("nodel.capture", 0)]
    # no >8GB pull
    assert [t for _, t in size_trits(_SIZES_OK)] == [1, 1, 1]
    assert [t for _, t in size_trits(_SIZES_BAD)] == [-1, 0, -1]
    assert [t for _, t in size_trits("5000000000 a\n4000000000 b\n")] == [1, 1, -1]
    assert [t for _, t in size_trits(f"{PULL_LIMIT} edge\n")] == [1, 1]
    # bitnetd bind
    assert parse_ss(_SS_OK)[0] == ("127.0.0.1", "8765", "bitnetd")
    assert [t for _, t in bind_trits(_SS_OK)] == [1, 1, 1, 1]
    assert [t for _, t in bind_trits(_SS_WILD)] == [-1, -1, -1, -1]
    assert [t for _, t in bind_trits(_SS_EXTRA)] == [1, -1, -1, -1]
    assert [t for _, t in bind_trits(_SS_SQUAT)] == [-1, -1, -1, 1]
    assert [t for _, t in bind_trits(None)] == [0, 0, 0, 0]
    # rc log
    assert [t for _, t in rc_trits(_RC_OK)] == [1, 1, 1]
    assert [t for _, t in rc_trits("diff 0\nss 2\n")] == [1, 0, -1]
    assert [t for _, t in rc_trits(None)] == [0, 0, 0]
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
    # rate rule: 1/20 undecided is exactly 0.05 -> green; 2/20 -> hold
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
    g1 = build(_DIFF_OK, _SIZES_OK, _SS_OK, _RC_OK)
    g2 = build(_DIFF_OK, _SIZES_OK, _SS_OK, _RC_OK)
    assert g1["verdict"] == "green" and g1["n"] == 2 + 3 + 4 + 3 + 1
    assert g1["rows"][-1] == ("step6.rc", 1) and g1["key"] == g2["key"]
    chained = build(_DIFF_OK, _SIZES_OK, _SS_OK, _RC_OK, prev=g1["key"])
    assert chained["key"] != g1["key"] and chained["verdict"] == "green"
    red = build(_DIFF_BAD, _SIZES_OK, _SS_WILD, _RC_OK)
    assert red["verdict"] == "hold" and red["rows"][-1] == ("step6.rc", -1)
    no_rc = build(_DIFF_OK, _SIZES_OK, _SS_OK, None)
    assert no_rc["verdict"] == "hold" and no_rc["rows"][-1] == ("step6.rc", 1)
    empty = build(None, None, None, None)
    assert empty["verdict"] == "hold" and empty["rows"][-1] == ("step6.rc", 0)
    return g1["key"]


def _read(path):
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--diff")
    ap.add_argument("--sizes")
    ap.add_argument("--ss")
    ap.add_argument("--rc-log")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        key = selftest()
        print(json.dumps({"selftest": "pass", "key": key, "rate": RATE,
                          "ports": list(PORTS), "engines": list(ENGINES)}))
        return 0
    out = build(_read(a.diff), _read(a.sizes), _read(a.ss), _read(a.rc_log), prev=a.prev)
    print(json.dumps(out, sort_keys=True))
    return 0 if out["verdict"] == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
