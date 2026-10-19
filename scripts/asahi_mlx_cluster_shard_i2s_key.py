"""Asahi MLX offline probe: P/E-cluster I2_S row-shard plan keyed to JTR1. Stdlib only.

Roadmap item: JuniorHome docs/ROADMAP.md #4, "Asahi MLX - probe only from
automations". The Quant Chief assignment visible for this run (slot
2026-10-19 05:05 MT) was older than one hour at the 10:20 MT run, so this run
takes the standing fallback: #4 as an offline probe. rails/linux/ASAHI.md says
an Asahi box that misses `import mlx` falls back to cpu AbsMean; this file
plans how that CPU lane would split I2_S weight rows across Apple SoC core
clusters of unequal speed (efficiency and performance clusters).

New capability (not in BitNet-mlx scripts/asahi_mlx_i2s_page_probe.py, which
plans 16 KiB page tiles, nor ports/asahi_mlx_lane_port_map.py, which maps the
mlx / llama.cpp / i2sd lane choice): heterogeneous-cluster row sharding.
  * topology is caller supplied, never read live: either --topology
    "NAME:CORES:CAPACITY,..." or --capacity-file pointing at a captured text
    with one "cpuN CAPACITY" line per CPU (for example a saved copy of the
    per-CPU cpu_capacity values). CPUs with equal capacity form one cluster.
  * rows of a ternary matrix (rows x cols) are split across clusters in
    proportion to throughput = cores * capacity, largest-remainder rounding,
    whole rows only, so every shard starts on a whole I2_S 128-trit block
    (32 bytes) boundary.
  * per-cluster trits: +1 assigned and within rate, 0 dropped because its
    throughput share is under RATE (not worth a thread pool), -1 malformed
    (zero cores, non-positive capacity, duplicate name) or over rate.
  * imbalance = max_i (rows_i / thr_i) / (rows / thr_total) - 1 over the
    assigned clusters; it must stay <= RATE for a green verdict.
It never reads /proc, /sys, device-tree, or sysctl itself, never imports mlx,
never runs a subprocess, opens a socket, or touches the network. It only
reads a local text file when you name one. It does not wrap, test, or
re-derive any other team file.

Side covered: I2_S / trit-pack key side (Trit Pack & I2_S). The quant-error
bound side for the same item belongs to the buddy (Winsor & Quant Error) in a
different file and path.

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  JTR1 codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected)
  gamma = mean |trit|
I2_S block: code = trit + 1 (-1 -> 0, 0 -> 1, +1 -> 2), four per byte MSB first,
  padded with code 1 (zero) to a 128-trit block (32 bytes).

Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16). Chained once per shard (pack = that shard's I2_S block bytes
for its trit, text = shard JSON), then closed over the whole plan
(pack = full JTR1 bytes, text = canonical plan JSON).
Rate 0.05: green needs zero -1 trits and imbalance <= 0.05.
Compatible engines: JuniorLLM (cpu AbsMean lane), BitNet-mlx, i2sd userspace,
MLX CPU on Asahi (only if `import mlx` already works; never installed here).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (named, never opened). mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage:
  python3 scripts/asahi_mlx_cluster_shard_i2s_key.py --selftest
  python3 scripts/asahi_mlx_cluster_shard_i2s_key.py --topology E:4:512,P:4:1024 \
      --rows 2560 --cols 6912 [--prev HEX16]
  python3 scripts/asahi_mlx_cluster_shard_i2s_key.py --capacity-file caps.txt --rows 2560 --cols 6912
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM", "BitNet-mlx", "i2sd userspace", "MLX CPU on Asahi")
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #4 Asahi MLX - probe only from automations"
GENESIS = "0" * 16
MAGIC = b"JTR1"
BLOCK_TRITS = 128
BLOCK_BYTES = BLOCK_TRITS // 4
JTR1_CODE = {0: 0b00, 1: 0b01, -1: 0b10}
JTR1_DECODE = {v: k for k, v in JTR1_CODE.items()}


# ---------------------------------------------------------------- topology
def parse_topology(spec):
    """'NAME:CORES:CAPACITY,...' -> list of dicts (bad entries kept, flagged)."""
    out = []
    for raw in [s.strip() for s in spec.split(",") if s.strip()]:
        parts = raw.split(":")
        c = {"name": raw, "cores": 0, "capacity": 0.0, "bad": "syntax"}
        if len(parts) == 3:
            try:
                c = {"name": parts[0], "cores": int(parts[1]),
                     "capacity": float(parts[2]), "bad": None}
            except ValueError:
                pass
        out.append(c)
    return out


def parse_capacity_text(text):
    """Captured 'cpuN CAPACITY' lines -> clusters grouped by equal capacity."""
    groups = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(":", " ").split()
        if len(parts) != 2 or not parts[0].startswith("cpu"):
            continue
        try:
            cap = float(parts[1])
        except ValueError:
            continue
        groups.setdefault(cap, []).append(parts[0])
    out = []
    for i, cap in enumerate(sorted(groups)):
        out.append({"name": "c%d_cap%g" % (i, cap), "cores": len(set(groups[cap])),
                    "capacity": cap, "bad": None})
    return out


def validate(clusters):
    seen = set()
    for c in clusters:
        if c["bad"]:
            continue
        if c["cores"] <= 0:
            c["bad"] = "cores<=0"
        elif c["capacity"] <= 0:
            c["bad"] = "capacity<=0"
        elif c["name"] in seen:
            c["bad"] = "duplicate"
        seen.add(c["name"])
    return clusters


# ---------------------------------------------------------------- planning
def row_bytes(cols):
    return -(-cols // BLOCK_TRITS) * BLOCK_BYTES


def plan(clusters, rows, cols):
    clusters = validate([dict(c) for c in clusters])
    good = [c for c in clusters if not c["bad"]]
    thr_all = sum(c["cores"] * c["capacity"] for c in good) or 1.0
    for c in clusters:
        c["throughput"] = 0.0 if c["bad"] else c["cores"] * c["capacity"]
        c["share"] = c["throughput"] / thr_all
    live = [c for c in good if c["share"] >= RATE]
    thr_live = sum(c["throughput"] for c in live) or 1.0
    # largest-remainder split of whole rows
    ideal = {c["name"]: rows * c["throughput"] / thr_live for c in live}
    alloc = {n: int(v) for n, v in ideal.items()}
    left = rows - sum(alloc.values())
    for n in sorted(ideal, key=lambda k: (-(ideal[k] - alloc[k]), k))[:left]:
        alloc[n] += 1
    rb = row_bytes(cols)
    imbalance = 0.0
    if live and rows > 0:
        base = rows / thr_live
        imbalance = max(alloc[c["name"]] / c["throughput"] for c in live) / base - 1.0
    shards, row0 = [], 0
    for c in clusters:
        n_rows = alloc.get(c["name"], 0)
        if c["bad"]:
            trit, why = -1, c["bad"]
        elif c["share"] < RATE:
            trit, why = 0, "share<rate"
        elif n_rows == 0:
            trit, why = -1, "no rows"
        else:
            local = (n_rows / c["throughput"]) / (rows / thr_live) - 1.0
            trit, why = (1, "ok") if local <= RATE else (-1, "over rate")
        shards.append({
            "cluster": c["name"], "cores": c["cores"], "capacity": c["capacity"],
            "share": round(c["share"], 6), "row0": row0, "rows": n_rows,
            "byte0": row0 * rb, "bytes": n_rows * rb,
            "block_aligned": (row0 * rb) % BLOCK_BYTES == 0, "trit": trit, "why": why,
        })
        row0 += n_rows
    trits = [s["trit"] for s in shards]
    trits.append(1 if live and imbalance <= RATE else -1)      # global balance
    trits.append(1 if row0 == rows and rows > 0 else -1)        # full coverage
    trits.append(1 if all(s["block_aligned"] for s in shards) else -1)
    verdict = "green" if (-1 not in trits and imbalance <= RATE) else "hold"
    return {"rows": rows, "cols": cols, "row_bytes": rb, "shards": shards,
            "imbalance": round(imbalance, 6), "trits": trits, "verdict": verdict}


# ---------------------------------------------------------------- packing
def pack_jtr1(trits):
    gamma = (sum(abs(t) for t in trits) / len(trits)) if trits else 0.0
    out = bytearray(MAGIC + struct.pack("<fI", gamma, len(trits)))
    for i in range(0, len(trits), 4):
        b = 0
        for j, t in enumerate(trits[i:i + 4]):
            b |= JTR1_CODE[t] << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def unpack_jtr1(blob):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    trits = []
    for i in range(n):
        code = (blob[12 + i // 4] >> (6 - 2 * (i % 4))) & 0b11
        if code == 0b11:
            raise ValueError("code 0b11 rejected")
        trits.append(JTR1_DECODE[code])
    return gamma, trits


def pack_i2s(trits):
    codes = [t + 1 for t in trits]
    pad = (-len(codes)) % BLOCK_TRITS
    codes += [1] * (pad if codes else BLOCK_TRITS)
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j in range(4):
            b |= codes[i + j] << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def close_key(prev, pack, text):
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def key_plan(p, prev=GENESIS):
    chain = prev
    for s in p["shards"]:
        chain = close_key(chain, pack_i2s([s["trit"]]), canon(s))
        s["key"] = chain
    jtr1 = pack_jtr1(p["trits"])
    i2s = pack_i2s(p["trits"])
    meta = {k: p[k] for k in ("rows", "cols", "row_bytes", "imbalance", "trits", "verdict")}
    meta.update({"rate": RATE, "item": ROADMAP_ITEM, "bind": BIND})
    p["jtr1_hex"] = jtr1.hex()
    p["i2s_bytes"] = len(i2s)
    p["key"] = close_key(chain, jtr1, canon(meta))
    return p


# ---------------------------------------------------------------- selftest
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    check("rate is 0.05", RATE == 0.05)
    check("flags false", not any((MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH)))
    check("bind 127.0.0.1", BIND == "127.0.0.1")
    # example fixture capacities, not measured hardware values
    good = key_plan(plan(parse_topology("E:4:512,P:4:1024"), 2560, 6912))
    rows = [s["rows"] for s in good["shards"]]
    check("good: rows cover matrix", sum(rows) == 2560)
    check("good: P gets about 2x E", rows == [853, 1707])
    check("good: imbalance within rate", good["imbalance"] <= RATE)
    check("good: verdict green", good["verdict"] == "green")
    check("good: row bytes 54 blocks", good["row_bytes"] == 54 * BLOCK_BYTES)
    check("good: shards block aligned", all(s["block_aligned"] for s in good["shards"]))
    check("good: key deterministic",
          good["key"] == key_plan(plan(parse_topology("E:4:512,P:4:1024"), 2560, 6912))["key"])
    g, t = unpack_jtr1(bytes.fromhex(good["jtr1_hex"]))
    check("good: JTR1 roundtrip", t == good["trits"] and abs(g - 1.0) < 1e-6)
    check("good: I2_S one 128 block", good["i2s_bytes"] == BLOCK_BYTES)
    check("close rule formula", close_key(GENESIS, b"", "x")
          == hashlib.sha256(b"0" * 16 + b"x").hexdigest()[:16])
    caps = "cpu0 512\ncpu1 512\ncpu2 512\ncpu3 512\ncpu4 1024\ncpu5 1024\ncpu6 1024\ncpu7 1024\n"
    cf = key_plan(plan(parse_capacity_text(caps), 2560, 6912))
    check("capacity file groups 2 clusters", [s["cores"] for s in cf["shards"]] == [4, 4])
    check("capacity file same rows", [s["rows"] for s in cf["shards"]] == rows)
    tiny = plan(parse_topology("X:1:10,P:8:1024"), 2560, 6912)
    check("tiny cluster dropped as 0", tiny["shards"][0]["trit"] == 0 and tiny["shards"][0]["rows"] == 0)
    check("tiny: still green", tiny["verdict"] == "green")
    bad = plan(parse_topology("E:0:512,P:4:1024,P:4:1024,junk"), 2560, 6912)
    check("malformed clusters -1", [s["trit"] for s in bad["shards"]][:1] == [-1]
          and bad["shards"][2]["why"] == "duplicate" and bad["shards"][3]["why"] == "syntax")
    check("malformed: hold", bad["verdict"] == "hold")
    over = plan(parse_topology("E:4:512,P:4:1024"), 4, 6912)
    check("4 rows over rate: hold", over["imbalance"] > RATE and over["verdict"] == "hold")
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 0.0, 1) + b"\xc0")
        check("0b11 rejected", False)
    except ValueError:
        check("0b11 rejected", True)
    print("selftest key", good["key"])
    print("selftest", "pass" if ok else "fail")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--topology")
    ap.add_argument("--capacity-file")
    ap.add_argument("--rows", type=int, default=2560)
    ap.add_argument("--cols", type=int, default=6912)
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest or not (a.topology or a.capacity_file):
        return selftest()
    if a.capacity_file:
        with open(a.capacity_file, encoding="utf-8") as fh:
            clusters = parse_capacity_text(fh.read())
    else:
        clusters = parse_topology(a.topology)
    print(json.dumps(key_plan(plan(clusters, a.rows, a.cols), a.prev), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
