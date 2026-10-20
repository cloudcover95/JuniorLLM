---
local: true
date: 2026-10-20
wave: "2026-10-20 05:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (I2_S pack step of the native f32 quantize path on overlay hosts)"
chief_assignment: "pair_A Live beta step 5, non-quant-error side; pair_B JuniorOS overlay remaining gaps (Quant Chief 05:08 MT, last_tips.json)"
verdict: viable
tags: [ternary, osai, findings, junioros-overlay, i2s, pack, swar, numpy, quantize]
---

# SWAR uint32 I2_S pack vs the shift-or chain: same bytes, about 11x faster

Previous note: [[2026-10-20_0255_f32-quantize-path-vs-f64-stdlib-gamma]]

## Question

The 02:55 note found the I2_S pack was the largest cost in the f32 quantize path (about 32 ms of 45 ms at 2560 x 2560) and asked whether a different pack kernel could be at least 1.5x faster than the shift-or chain over a (n/4, 4) view while producing byte-identical output. Candidates: one uint8 matmul against [64, 16, 4, 1], a lookup table, and (added here) a SWAR kernel that packs four trits held in one uint32 with a single multiply.

## Method

- Trits come from the 02:55 f32 pairwise path (gamma = f32 mean |w|, rint(w / (gamma + 1e-7)), clip to [-1, 1], int8). Shapes 2560x2560 and 2560x6912 (the 2B4T attention and FFN sizes), Gaussian and Student-t(4) weights at scale 0.02, seeds 555 and 1555.
- Mapping is `junior_bitnet.i2s.pack` in this repo: 0 -> 00, +1 -> 01, -1 -> 10, four trits per byte, first trit in the high bits.
- Kernels: (1) shift-or with an int64 `np.where` temp (02:55 main bench), (2) shift-or on uint8 codes (02:55 supplement, the baseline here), (3) uint8 codes (n/4, 4) @ [64, 16, 4, 1], (4) a 65,536-entry uint16 LUT mapping two raw int8 trits to a nibble, then nibble pairs to a byte, (5) SWAR: view the int8 trits as uint32, code = (x & 0x01010101) + ((x >> 7) & 0x01010101) per byte, then byte = (code * (2^30 + 2^20 + 2^10 + 1)) >> 24 in uint32. The multiply lands the four 2-bit codes at bits 30, 28, 26 and 24; every cross term lands at or below bit 21 without carries, or wraps out of the word.
- Checks: all 81 four-trit patterns against the stdlib `junior_bitnet.i2s.pack`; on every tensor, each kernel's bytes against kernel (1), and the first 65,536 trits against the stdlib pack.
- Timing: 15 repeats per kernel after one warm-up (7 for the quantize and end-to-end rows), median (min-max) in ms, `time.perf_counter`; peak traced allocation with `tracemalloc`. numpy 2.5.4, 8-core box, one thread (numpy elementwise), nothing bound, no network, no installs.

Command (script kept at `/workspace/tbench/r20261020_0555/i2s_pack_kernel_bench.py`, reproduced in full below):

```
cd /workspace/tbench/r20261020_0555 && python3 i2s_pack_kernel_bench.py
```

## Results

### A. Pack kernel time (ms, median (min-max), 15 repeats) and speedup over the uint8 shift-or

| shape | kind | seed | shift-or, int64 where (02:55 main) | shift-or, uint8 codes (02:55 supplement) | uint8 matmul @ [64,16,4,1] | 2-trit uint16 LUT gather | SWAR uint32 multiply |
|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 555 | 26.06 (23.98-28.22), 0.99x | 25.76 (23.38-29.27), 1.00x | 27.54 (24.37-38.33), 0.94x | 8.52 (7.95-9.39), 3.02x | 2.20 (2.01-2.46), 11.73x |
| 2560x2560 | gauss | 1555 | 27.41 (25.36-30.16), 0.95x | 25.92 (23.43-27.30), 1.00x | 26.78 (24.29-28.59), 0.97x | 8.60 (7.90-9.70), 3.01x | 2.41 (2.03-3.09), 10.75x |
| 2560x2560 | t4 | 555 | 24.78 (22.69-27.17), 0.98x | 24.36 (22.12-26.56), 1.00x | 25.29 (23.29-28.28), 0.96x | 8.53 (7.56-9.59), 2.86x | 2.12 (2.01-2.44), 11.48x |
| 2560x2560 | t4 | 1555 | 24.22 (22.84-26.68), 1.01x | 24.36 (22.32-25.89), 1.00x | 26.00 (23.12-29.75), 0.94x | 8.43 (7.71-9.76), 2.89x | 2.04 (2.02-2.42), 11.95x |
| 2560x6912 | gauss | 555 | 69.66 (66.34-79.85), 0.96x | 67.05 (64.55-71.65), 1.00x | 70.14 (66.39-90.09), 0.96x | 22.22 (21.46-28.52), 3.02x | 5.60 (5.43-6.35), 11.98x |
| 2560x6912 | gauss | 1555 | 68.90 (64.99-71.78), 0.96x | 66.47 (64.45-73.29), 1.00x | 68.50 (65.50-73.69), 0.97x | 21.79 (20.90-23.18), 3.05x | 5.52 (5.39-6.39), 12.04x |
| 2560x6912 | t4 | 555 | 63.53 (61.33-67.11), 0.97x | 61.68 (60.21-65.28), 1.00x | 69.30 (66.35-73.57), 0.89x | 22.49 (21.38-24.46), 2.74x | 5.94 (5.48-6.56), 10.39x |
| 2560x6912 | t4 | 1555 | 65.32 (61.87-68.04), 0.97x | 63.16 (60.71-69.57), 1.00x | 64.88 (62.71-68.59), 0.97x | 22.48 (21.51-24.46), 2.81x | 5.77 (5.59-6.57), 10.95x |

### B. Bytes, identity and peak memory

| shape | kind | seed | trits | pack bytes | zero-trit fraction | all 5 kernels identical | stdlib slice match | pack sha16 | peak MB shift-or u8 / LUT / SWAR |
|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 555 | 6553600 | 1638400 | 0.3101 | yes | yes | `2da44830797e0a56` | 13.1 / 4.9 / 13.1 |
| 2560x2560 | gauss | 1555 | 6553600 | 1638400 | 0.3099 | yes | yes | `9413d6b92f0be0df` | 13.1 / 4.9 / 13.1 |
| 2560x2560 | t4 | 555 | 6553600 | 1638400 | 0.3569 | yes | yes | `8046cdf90be6f940` | 13.1 / 4.9 / 13.1 |
| 2560x2560 | t4 | 1555 | 6553600 | 1638400 | 0.3564 | yes | yes | `a1bc78127dcfa053` | 13.1 / 4.9 / 13.1 |
| 2560x6912 | gauss | 555 | 17694720 | 4423680 | 0.3101 | yes | yes | `17487e4c1d49bbb8` | 35.4 / 13.3 / 35.4 |
| 2560x6912 | gauss | 1555 | 17694720 | 4423680 | 0.3101 | yes | yes | `43836b0536f9ffce` | 35.4 / 13.3 / 35.4 |
| 2560x6912 | t4 | 555 | 17694720 | 4423680 | 0.3567 | yes | yes | `9ed42365d456008a` | 35.4 / 13.3 / 35.4 |
| 2560x6912 | t4 | 1555 | 17694720 | 4423680 | 0.3566 | yes | yes | `b4957c6948694a7a` | 35.4 / 13.3 / 35.4 |

All 81 four-trit patterns: shiftor_i64 match, shiftor_u8 match, matmul_u8 match, lut16 match, swar_u32 match.

### C. End to end: quantize + pack (ms, median (min-max), 7 repeats)

| shape | kind | seed | quantize only | quantize + shift-or u8 | quantize + SWAR | speedup | pack share before / after |
|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 555 | 11.89 (10.17-17.97) | 42.31 (37.90-64.04) | 14.09 (12.53-18.01) | 3.00x | 61% / 16% |
| 2560x2560 | gauss | 1555 | 11.11 (10.39-11.78) | 38.19 (34.74-42.89) | 12.80 (12.65-14.93) | 2.98x | 68% / 19% |
| 2560x2560 | t4 | 555 | 9.35 (8.04-9.89) | 35.87 (31.43-37.58) | 10.69 (10.12-11.86) | 3.36x | 68% / 20% |
| 2560x2560 | t4 | 1555 | 10.84 (10.50-13.16) | 36.34 (34.56-42.05) | 12.28 (12.13-13.16) | 2.96x | 67% / 17% |
| 2560x6912 | gauss | 555 | 46.23 (42.87-61.03) | 148.87 (141.67-187.78) | 67.61 (61.40-71.52) | 2.20x | 45% / 8% |
| 2560x6912 | gauss | 1555 | 48.57 (45.03-52.35) | 144.89 (131.75-146.82) | 64.95 (60.81-70.46) | 2.23x | 46% / 8% |
| 2560x6912 | t4 | 555 | 43.01 (41.79-52.05) | 135.32 (129.49-166.34) | 66.54 (59.75-73.64) | 2.03x | 46% / 9% |
| 2560x6912 | t4 | 1555 | 43.11 (40.90-54.64) | 142.42 (134.73-160.85) | 66.73 (66.00-80.17) | 2.13x | 44% / 9% |

## Verdict

**Viable.** The SWAR uint32 pack is 11.60x (10.39-12.04x) faster than the uint8 shift-or chain (median over 8 tensors, range in brackets), with byte-identical output on every tensor, on all 81 patterns and against the stdlib `junior_bitnet.i2s.pack`. Error is zero by construction: the pack is lossless and the bytes, and so any close key over them, do not change. Peak memory is the same as the shift-or (two n/4 uint32 temps). End to end, quantize + pack drops by 2.60x (2.03-3.36x), and the pack's share of the path falls from about 53% to about 12%.

- **The uint8 matmul is not the answer** (0.96x (0.89-0.97x)): numpy's integer matmul has no BLAS path, so it does the same work as the shift-or more slowly.
- **The 2-trit uint16 LUT is a solid second** (2.95x (2.74-3.05x)) and uses the least memory (one n/2 gather), but it is a 64 KB table and a random gather, so it stays well behind SWAR.
- **Preconditions for SWAR:** int8 trits strictly in {-1, 0, 1}, C-contiguous, length a multiple of 4 (pad with zeros as the stdlib pack does), and a little-endian host (x86-64 and arm64 Linux both are); a big-endian host would need the byte order of the multiply reversed. A stray value such as 2 or -2 would silently produce a wrong code, so the clip stays in front of it.
- **What it does not change:** the pack bytes and the JTR1/I2_S format are untouched, so Trit Pack's and Winsor's keys, the close rules and `rails/linux/i2s_pack.c` stay valid. This is a drop-in for the numpy fallback pack only; it says nothing about the C pack, which was not measured here.
- **Caveat:** synthetic weights; pack time depends only on the trit count, not the values, so real 2B4T tensors should land in the same place. Timings are single-thread numpy on this box.

## This wave's pair files

- pair_A (Live beta step 5, non-quant-error side): Trit Pack [JuniorLLM e82c972 `scripts/backfill_head_tree_i2s_key.py`](https://github.com/cloudcover95/JuniorLLM/blob/e82c972/scripts/backfill_head_tree_i2s_key.py). Winsor's 05:20 push was not seen in the local copies at 06:00 MT.
- pair_B (JuniorOS overlay remaining gaps): Handshake [JuniorLLM 83ade39 `ports/overlay_tarball_owner_mode_handshake.py`](https://github.com/cloudcover95/JuniorLLM/blob/83ade39/ports/overlay_tarball_owner_mode_handshake.py). Side-Commit's 05:35 push was not seen in the local copies.

## Next question

With the pack at about 12% of the path, ternarize (w / gamma, rint, clip, astype int8) is now the main cost. Is a threshold ternarize that skips the divide and the rint (two comparisons of w against +/- gamma/2 written straight into int8, with the tie rule matched to rint half-even) at least 1.5x faster than the current path, with zero trit flips?

## Script

```python
#!/usr/bin/env python3
"""I2_S 2-bit pack kernels (JuniorOS overlay quantize path): is a SWAR uint32 multiply pack,
a uint8 matmul against [64,16,4,1], or a 2-trit uint16 LUT at least 1.5x faster than the
current shift-or chain over a (n/4, 4) view, byte for byte identical?
Mapping is junior_bitnet.i2s.pack: 0->00, +1->01, -1->10, MSB first, 4 trits per byte.
numpy + stdlib only, offline, no sockets, no installs. Writes results.json."""
import json, sys, time, tracemalloc, hashlib
import numpy as np
sys.path.insert(0, "/workspace/hp-20261020-0535/JuniorLLM")
from junior_bitnet.i2s import pack as ref_pack  # stdlib reference (pure Python)

assert sys.byteorder == "little"
SHAPES = [(2560, 2560), (2560, 6912)]
KINDS = ["gauss", "t4"]
SEEDS = [555, 1555]
REP = 15
EPS = 1e-7
M = np.uint32((1 << 30) | (1 << 20) | (1 << 10) | 1)
LO = np.uint32(0x01010101)

def tstat(f, n=REP):
    f(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return {"med": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75)), "n": n}

def peak(f):
    tracemalloc.start(); tracemalloc.reset_peak(); f()
    p = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return p

def make(kind, shape, rng):
    if kind == "gauss":
        return rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)
    return (rng.standard_t(4, size=shape) * 0.02).astype(np.float32)

def quant(w):  # 02:55 f32 pairwise path: gamma + x/g + rint + clip -> int8 trits
    g = np.float32(np.abs(w).mean(dtype=np.float32))
    return np.clip(np.rint(w / np.float32(g + np.float32(EPS))), -1, 1).astype(np.int8)

# ---- pack kernels (q: C-contiguous int8 trits, size % 4 == 0)
def pack_shiftor_i64(q):   # 02:55 main bench pack2 (np.where int64 temp)
    c = np.where(q < 0, 2, q).astype(np.uint8).reshape(-1, 4)
    return (c[:, 0] << 6 | c[:, 1] << 4 | c[:, 2] << 2 | c[:, 3]).astype(np.uint8)

def pack_shiftor_u8(q):    # 02:55 supplement pack2u (uint8 codes)
    c = np.where(q < 0, np.uint8(2), q.view(np.uint8)).reshape(-1, 4)
    return (c[:, 0] << 6 | c[:, 1] << 4 | c[:, 2] << 2 | c[:, 3])

W4 = np.array([64, 16, 4, 1], np.uint8)
def pack_matmul_u8(q):     # codes (n/4,4) uint8 @ [64,16,4,1]
    c = np.where(q < 0, np.uint8(2), q.view(np.uint8)).reshape(-1, 4)
    return c @ W4

LUT16 = np.zeros(65536, np.uint8)
for a in (-1, 0, 1):
    for b in (-1, 0, 1):
        k = (a & 0xFF) | ((b & 0xFF) << 8)          # little-endian: first trit is the low byte
        LUT16[k] = ((0 if a == 0 else (1 if a == 1 else 2)) << 2) | (0 if b == 0 else (1 if b == 1 else 2))
def pack_lut16(q):         # 2 trits -> nibble via uint16 view gather, then nibble pair -> byte
    nib = LUT16[q.view(np.uint16)].reshape(-1, 2)
    return nib[:, 0] << 4 | nib[:, 1]

def pack_swar(q):          # uint32 lanes: code = (x & 1) + (x>>7 & 1) per byte; byte = (code*M)>>24
    x = q.view(np.uint32)
    c = (x & LO) + ((x >> 7) & LO)
    c *= M
    c >>= 24
    return c.astype(np.uint8)

KERNELS = {"shiftor_i64": pack_shiftor_i64, "shiftor_u8": pack_shiftor_u8,
           "matmul_u8": pack_matmul_u8, "lut16": pack_lut16, "swar_u32": pack_swar}

def main():
    out = {"numpy": np.__version__, "rep": REP, "rows": []}
    # exhaustive identity: every 4-trit pattern (81) against the stdlib reference
    pats = np.array(np.meshgrid(*[[-1, 0, 1]] * 4, indexing="ij"), np.int8).reshape(4, -1).T.copy().ravel()
    refp = np.frombuffer(ref_pack(pats.tolist()), np.uint8)
    out["exhaustive81"] = {k: bool(np.array_equal(f(pats), refp)) for k, f in KERNELS.items()}
    print("exhaustive81", out["exhaustive81"], flush=True)
    for shape in SHAPES:
        for kind in KINDS:
            for seed in SEEDS:
                w = make(kind, shape, np.random.default_rng(seed)); q = quant(w); n = q.size
                base = pack_shiftor_i64(q)
                sl = q[:65536 // shape[1] + 1].ravel()[:65536]
                ref_ok = bool(np.frombuffer(ref_pack(sl.tolist()), np.uint8).tobytes() == base[:16384].tobytes())
                r = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "n": n,
                     "pack_bytes": int(base.nbytes), "sha256_16": hashlib.sha256(base.tobytes()).hexdigest()[:16],
                     "ref_slice_ok": ref_ok, "frac0": float((q == 0).mean()),
                     "t_quant": tstat(lambda: quant(w), 7), "k": {}}
                for name, f in KERNELS.items():
                    p = f(q)
                    r["k"][name] = {"t": tstat(lambda: f(q)), "peak": peak(lambda: f(q)),
                                    "identical": bool(p.dtype == np.uint8 and p.tobytes() == base.tobytes())}
                b = r["k"]["shiftor_u8"]["t"]["med"]
                for name in KERNELS:
                    r["k"][name]["speedup_vs_u8"] = b / r["k"][name]["t"]["med"]
                    r["k"][name]["speedup_vs_i64"] = r["k"]["shiftor_i64"]["t"]["med"] / r["k"][name]["t"]["med"]
                r["e2e_u8"] = tstat(lambda: pack_shiftor_u8(quant(w)), 7)
                r["e2e_swar"] = tstat(lambda: pack_swar(quant(w)), 7)
                out["rows"].append(r)
                print(r["shape"], kind, seed, "ref", ref_ok, "quant %.2f" % r["t_quant"]["med"],
                      " ".join("%s %.2f(%.2f-%.2f) %.2fx id=%s pk=%.1fMB" % (k, v["t"]["med"], v["t"]["min"], v["t"]["max"],
                               v["speedup_vs_u8"], v["identical"], v["peak"] / 1e6) for k, v in r["k"].items()),
                      "e2e u8 %.2f swar %.2f" % (r["e2e_u8"]["med"], r["e2e_swar"]["med"]), flush=True)
    json.dump(out, open("results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
```

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-20_0255_f32-quantize-path-vs-f64-stdlib-gamma.md.

close: 495ba3cd2ee43f4b
