---
local: true
date: 2026-10-21
wave: "2026-10-21 14:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (end-to-end native f32 quantize-and-pack path on overlay hosts under bitnetd MemoryMax)"
chief_assignment: "pair_A JuniorOS overlay remaining gaps; pair_B Live beta step 5, non-quant-error side (Quant Chief 14:08 MT, last_tips.json)"
verdict: viable
tags: [ternary, osai, findings, junioros-overlay, i2s, swar, threshold, gamma, numpy, quantize, memory, blocking]
---

# One blocked threshold-and-pack pass: same packed bytes, 1.74x faster than the staged lean path, no full-size scratch

Previous note: [[2026-10-21_1055_gamma-chunk-split-tree-vs-abs-mean]]

## Question

The 10:55 note asked: with gamma, ternarize and pack each lean on their own, does running threshold ternarize and the SWAR pack block by block while the block is still in L2 (gamma on the split tree as its own pass, then one more pass over w) give another 1.5x end to end on the quantize path, with bit-identical packed bytes?

## Method

- Tensors: 2560x2560 and 2560x6912 (2B4T attention and FFN sizes), Gaussian and Student-t(4) weights at scale 0.02, new seeds 1455 and 2455, f32.
- Paths, all ending in I2_S 2-bit codes packed four to a byte:
  - **orig**: the 02:55 path, `np.abs(w).mean` gamma, divide, rint, clip to int8, SWAR pack ([[2026-10-20_0555_swar-u32-i2s-pack-vs-shiftor-chain]]).
  - **staged** (baseline): the lean path as landed in the last three notes, split-tree gamma ([[2026-10-21_1055_gamma-chunk-split-tree-vs-abs-mean]]), `nextafter` cutoff T and full-size threshold ternarize ([[2026-10-21_0555_threshold-ternarize-vs-rint-divide]]), then full-size SWAR pack.
  - **blocked**: split-tree gamma and T as above, then per block of B elements: `greater`/`less` into preallocated bool buffers, int8 subtract, SWAR pack in preallocated uint32 buffers, write into the output slice.
  - **blocked codes** (new): same, but skips the int8 trit array. Each byte's code is `gt + (lt << 1)` read as uint32 lanes (0, +1 gives 1, -1 gives 2), the same codes `(x & 1) + (x >> 7 & 1)` gives on int8 trits, then the same multiply-shift.
  - Block sizes B of 16 Ki, 64 Ki and 256 Ki elements.
- Checks: packed bytes of every path compared byte for byte with orig on all 8 tensors; identity sweep of both blocked variants at all three block sizes against staged over 132 arrays (60 random sizes up to 3M plus 4, 8, 128 and sizes around 4x64 Ki, both distributions; n a multiple of 4 as the I2_S pack needs), 792 comparisons.
- Timing: 15 repeats after one warm-up, median (min-max) ms with `time.perf_counter`; peak traced allocation per call with `tracemalloc`. The blocked paths' fixed buffers are outside the trace and are stated separately. numpy 2.5.4, 8-core box, nothing bound, no network, no installs. The script ran twice; tables are run 2, run 1 is summarised in the verdict.

Command (script kept at `/workspace/tbench/r20261021_1455/fused_quant_pack_bench.py`, reproduced in full below):

```
cd /workspace/tbench/r20261021_1455 && python3 fused_quant_pack_bench.py
```

## Results

### A. End-to-end quantize and pack time (ms, median (min-max), 15 repeats), speedup over staged

| shape | kind | seed | packed sha16 | orig | staged (baseline) | blocked codes 16 Ki | blocked 64 Ki | blocked codes 64 Ki | blocked codes 256 Ki |
|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 1455 | `4962a77ca9b2f981` | 12.58 (12.25-15.14), 0.54x | 6.85 (6.68-7.15), 1.00x | 5.78 (5.62-6.11), 1.19x | 5.04 (4.91-5.30), 1.36x | 4.54 (4.38-4.70), 1.51x | 4.78 (4.49-5.41), 1.43x |
| 2560x2560 | gauss | 2455 | `6aace9e7452b1c47` | 12.30 (12.10-17.19), 0.55x | 6.76 (6.64-6.85), 1.00x | 5.73 (5.62-6.01), 1.18x | 4.96 (4.88-5.12), 1.36x | 4.42 (4.36-4.60), 1.53x | 4.61 (4.48-4.96), 1.47x |
| 2560x2560 | t4 | 1455 | `7c5be3e97a58e5c8` | 10.44 (10.28-11.73), 0.74x | 7.76 (6.77-8.12), 1.00x | 5.89 (5.50-6.05), 1.32x | 5.11 (4.84-10.03), 1.52x | 4.45 (4.29-4.89), 1.74x | 4.52 (4.34-4.81), 1.72x |
| 2560x2560 | t4 | 2455 | `3bbe30ccca86fd88` | 12.27 (12.15-12.58), 0.56x | 6.88 (6.70-6.98), 1.00x | 5.80 (5.53-5.99), 1.19x | 5.20 (4.88-5.39), 1.32x | 4.59 (4.43-5.54), 1.50x | 4.54 (4.43-4.84), 1.51x |
| 2560x6912 | gauss | 1455 | `3a03fbc084003c74` | 50.49 (43.23-79.48), 0.44x | 22.07 (21.44-22.65), 1.00x | 15.82 (15.18-16.49), 1.39x | 13.81 (13.56-43.92), 1.60x | 12.13 (12.03-12.60), 1.82x | 12.51 (12.34-12.87), 1.76x |
| 2560x6912 | gauss | 2455 | `6bd8cbe2c7421f5d` | 47.49 (44.51-51.53), 0.45x | 21.26 (21.04-22.58), 1.00x | 15.58 (15.39-15.87), 1.37x | 13.72 (13.55-14.48), 1.55x | 12.13 (12.03-12.33), 1.75x | 12.44 (12.28-15.06), 1.71x |
| 2560x6912 | t4 | 1455 | `ad5f36385db92da4` | 55.34 (53.60-63.87), 0.39x | 21.61 (20.97-24.46), 1.00x | 15.67 (15.30-20.78), 1.38x | 13.89 (13.64-14.25), 1.56x | 12.22 (11.97-13.02), 1.77x | 13.31 (12.39-14.50), 1.62x |
| 2560x6912 | t4 | 2455 | `05e5a1b94918c905` | 47.19 (44.39-53.62), 0.45x | 21.17 (20.99-21.64), 1.00x | 15.42 (15.16-15.82), 1.37x | 13.75 (13.55-14.07), 1.54x | 12.23 (12.02-12.37), 1.73x | 12.72 (12.51-13.33), 1.67x |

### B. Bytes and identity

| shape | kind | seed | packed output MB | peak traced MB orig / staged / blocked codes 64 Ki | blocked codes 64 Ki fixed buffers MB | all paths byte-identical to orig |
|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 1455 | 1.64 | 52.43 / 19.66 / 0.0012 | 0.59 | yes |
| 2560x2560 | gauss | 2455 | 1.64 | 52.43 / 19.66 / 0.0012 | 0.59 | yes |
| 2560x2560 | t4 | 1455 | 1.64 | 52.43 / 19.66 / 0.0012 | 0.59 | yes |
| 2560x2560 | t4 | 2455 | 1.64 | 52.43 / 19.66 / 0.0012 | 0.59 | yes |
| 2560x6912 | gauss | 1455 | 4.42 | 141.56 / 53.08 / 0.0015 | 0.59 | yes |
| 2560x6912 | gauss | 2455 | 4.42 | 141.56 / 53.08 / 0.0015 | 0.59 | yes |
| 2560x6912 | t4 | 1455 | 4.42 | 141.56 / 53.08 / 0.0015 | 0.59 | yes |
| 2560x6912 | t4 | 2455 | 4.42 | 141.56 / 53.08 / 0.0015 | 0.59 | yes |

Identity sweep: 792 comparisons, 0 mismatches (both runs).

## Verdict

**Viable on speed and memory.** The blocked-codes pass at 64 Ki elements packs byte-identical I2_S output on all 8 tensors and all 792 sweep comparisons, and is 1.74x (1.50-1.82x) faster than the staged lean path in run 2 and 1.60x (1.51-1.82x) in run 1. Every tensor in both runs clears 1.5x, with the FFN-sized 2560x6912 tensors at 1.63-1.82x. Against the original 02:55 path it is 3.32x (2.35-4.53x) end to end in run 2. It also removes the staged path's full-size scratch (19.7 MB at 2560x2560, 53.1 MB at 2560x6912, from the bool masks and the int8 trit array) and keeps only the packed output plus about 0.59 MB of fixed buffers, so peak transient memory drops by roughly 33x and 90x on a MemoryMax-capped bitnetd host.

- **Where the gain comes from:** keeping each block's masks and codes in L2 saves most of it (blocked with the int8 subtract is 1.53x), and dropping the int8 trit array adds the rest (1.74x). The 64 Ki block is the sweet spot; 16 Ki pays too much Python loop overhead (1.34x) and 256 Ki starts spilling L2 (1.64x).
- **What it does not change:** gamma bits, T, the trits, the 2-bit code map, the JTR1/I2_S byte layout, Trit Pack's and Winsor's keys and the close rules. The split-tree gamma still needs its own pass because a per-block gamma would not keep the split tree's bits.
- **Preconditions:** w must be C-contiguous and its size a multiple of 4 (the I2_S pack already requires that). The code trick depends on numpy bools being stored as bytes 0 or 1, which numpy guarantees for comparison output. Re-run the sweep if numpy is upgraded on the overlay image.
- **Caveat:** synthetic weights and single-thread numpy on this box; time depends on element count, not values. The C path (`absmean.c` plus a C pack) was not measured.

## This wave's pair files

- pair_A (JuniorOS overlay remaining gaps): Trit Pack [BitNet-mlx 1fa67a2 scripts/junioros_kconfig_closure_i2s_key.py](https://github.com/cloudcover95/BitNet-mlx/commit/1fa67a2) (kconfig.junior closure I2_S key). Winsor's 14:20 push was not seen in the local copies.
- pair_B (Live beta step 5, non-quant-error side): no Side-Commit or Handshake push from 14:35 was seen in the local copies (Handshake's 14:35 JuniorLLM clone is still at dc5b02f).

## Next question

The quantize path is now lean on the numpy fallback; does the decode side match it? Is a blocked SWAR unpack of I2_S bytes straight into a bitplane or int8 matvec, block by block in L2, at least 1.5x faster or smaller than unpacking the whole tensor first, with bit-identical matvec output?

## Script

```python
#!/usr/bin/env python3
"""JuniorOS overlay quantize path, end to end: staged lean path (split-tree gamma -> full threshold
ternarize -> full SWAR pack) vs one blocked pass that thresholds and SWAR-packs each L2-sized block
into preallocated buffers. Question: >= 1.5x faster or smaller, with bit-identical packed bytes?
numpy + stdlib only, offline, no sockets, no installs."""
import json, time, tracemalloc, hashlib
import numpy as np

SHAPES = [(2560, 2560), (2560, 6912)]
KINDS = ["gauss", "t4"]
SEEDS = [1455, 2455]
REP = 15
EPS = np.float32(1e-7)
CH = 1 << 16
M = np.uint32((1 << 30) | (1 << 20) | (1 << 10) | 1)
LO = np.uint32(0x01010101)
BLOCKS = [1 << 14, 1 << 16, 1 << 18]

def tstat(f, n=REP):
    f(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts); return {"med": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": n}

def peak(f):
    tracemalloc.start(); tracemalloc.reset_peak(); f(); p = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return p

def make(kind, shape, rng):
    if kind == "gauss": return rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)
    return (rng.standard_t(4, size=shape) * 0.02).astype(np.float32)

def cutoff(g):
    g = np.float32(g); half = np.float32(0.5); t = np.float32(g * half)
    while np.float32(t / g) > half: t = np.nextafter(t, np.float32(0))
    while np.float32(np.nextafter(t, np.float32(np.inf)) / g) <= half: t = np.nextafter(t, np.float32(np.inf))
    return np.float32(t)

def _tree(f, s, n, cbuf):
    if n <= CH:
        c = cbuf[:n]; np.abs(f[s:s + n], out=c); return np.add.reduce(c, dtype=np.float32)
    n2 = n // 2; n2 -= n2 % 8
    return np.float32(_tree(f, s, n2, cbuf) + _tree(f, s + n2, n - n2, cbuf))
def g_tree(f, cbuf): return np.float32(np.float32(_tree(f, 0, f.size, cbuf) / np.float32(f.size)) + EPS)

def pack_swar(q):
    x = q.view(np.uint32); c = (x & LO) + ((x >> 7) & LO); c *= M; c >>= 24; return c.astype(np.uint8)

# A. original path (02:55 note): abs().mean gamma, divide + rint + clip, SWAR pack
def path_orig(w, st):
    g = np.float32(np.float32(np.abs(w).mean(dtype=np.float32)) + EPS)
    q = np.clip(np.rint(w / g), -1, 1).astype(np.int8); return pack_swar(q.reshape(-1))
# B. staged lean path (05:55 + 10:55 notes): split-tree gamma, threshold ternarize, SWAR pack, each full-size
def path_staged(w, st):
    f = w.reshape(-1); T = cutoff(g_tree(f, st["cbuf"]))
    q = (f > T).view(np.int8) - (f < -T).view(np.int8); return pack_swar(q)
# C. blocked: split-tree gamma pass, then one pass that thresholds + packs each block into fixed buffers
def make_state(B, n):
    return {"cbuf": np.empty(CH, np.float32), "B": B, "gt": np.empty(B, np.bool_), "lt": np.empty(B, np.bool_),
            "q": np.empty(B, np.int8), "c1": np.empty(B // 4, np.uint32), "c2": np.empty(B // 4, np.uint32),
            "out": np.empty(n // 4, np.uint8)}
def path_blocked(w, st):
    f = w.reshape(-1); n = f.size; B = st["B"]; T = cutoff(g_tree(f, st["cbuf"])); nT = np.float32(-T)
    out = st["out"]; gt, lt, q, c1, c2 = st["gt"], st["lt"], st["q"], st["c1"], st["c2"]
    for s in range(0, n, B):
        m = min(B, n - s); fb = f[s:s + m]
        np.greater(fb, T, out=gt[:m]); np.less(fb, nT, out=lt[:m])
        np.subtract(gt[:m].view(np.int8), lt[:m].view(np.int8), out=q[:m])
        k = m // 4; x = q[:m].view(np.uint32); a = c1[:k]; b = c2[:k]
        np.bitwise_and(x, LO, out=a); np.right_shift(x, 7, out=b); np.bitwise_and(b, LO, out=b)
        np.add(a, b, out=a); np.multiply(a, M, out=a); np.right_shift(a, 24, out=a)
        out[s // 4:s // 4 + k] = a
    return out
# D. blocked codes: skip the int8 trit array; per byte code = gt + (lt << 1) (0, +1 -> 1, -1 -> 2), the same
#    2-bit codes the SWAR rule (x & 1) + (x >> 7 & 1) gives on int8 trits, then the same multiply-shift pack
def path_blocked_codes(w, st):
    f = w.reshape(-1); n = f.size; B = st["B"]; T = cutoff(g_tree(f, st["cbuf"])); nT = np.float32(-T)
    out = st["out"]; gt, lt, q, c1, c2 = st["gt"], st["lt"], st["q"], st["c1"], st["c2"]
    for s in range(0, n, B):
        m = min(B, n - s); fb = f[s:s + m]
        np.greater(fb, T, out=gt[:m]); np.less(fb, nT, out=lt[:m])
        k = m // 4; a = c1[:k]; b = c2[:k]
        np.left_shift(lt[:m].view(np.uint32), 1, out=b)
        np.add(gt[:m].view(np.uint32), b, out=a); np.multiply(a, M, out=a); np.right_shift(a, 24, out=a)
        out[s // 4:s // 4 + k] = a
    return out

def main():
    out = {"numpy": np.__version__, "rep": REP, "blocks": BLOCKS, "rows": []}
    # identity sweep vs staged path on odd shapes (n % 4 == 0 as the I2_S pack requires)
    rng = np.random.default_rng(14); idn = {"n": 0, "mismatch": 0}
    for n in [int(x) * 4 for x in rng.integers(1, 750_000, 60)] + [4, 8, 128, 4 * CH - 4, 4 * CH, 4 * CH + 4]:
        for kind in KINDS:
            w = make(kind, (n,), rng); ref = path_staged(w, make_state(1 << 16, n)).tobytes()
            for B in BLOCKS:
                for fn in (path_blocked, path_blocked_codes):
                    idn["n"] += 1; idn["mismatch"] += int(fn(w, make_state(B, n)).tobytes() != ref)
    out["identity"] = idn; print("identity", idn, flush=True)
    for shape in SHAPES:
        for kind in KINDS:
            for seed in SEEDS:
                w = make(kind, shape, np.random.default_rng(seed)); n = w.size
                K = [("orig", path_orig, make_state(1 << 16, n)), ("staged", path_staged, make_state(1 << 16, n))]
                for B in BLOCKS:
                    K.append((f"blocked_{B // 1024}k", path_blocked, make_state(B, n)))
                    K.append((f"blocked_codes_{B // 1024}k", path_blocked_codes, make_state(B, n)))
                ref = path_orig(w, K[0][2]).tobytes()
                r = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "n": n,
                     "pack_sha16": hashlib.sha256(ref).hexdigest()[:16], "k": {}}
                for name, fn, st in K:
                    same = fn(w, st).tobytes() == ref
                    fixed = sum(v.nbytes for v in st.values() if isinstance(v, np.ndarray))
                    r["k"][name] = {"t": tstat(lambda: fn(w, st)), "peak": peak(lambda: fn(w, st)), "fixed_bytes": fixed,
                                    "identical": bool(same)}
                for base in ("orig", "staged"):
                    b = r["k"][base]["t"]["med"]
                    for name in r["k"]: r["k"][name]["x_vs_" + base] = b / r["k"][name]["t"]["med"]
                out["rows"].append(r)
                print(r["shape"], kind, seed, " | ".join("%s %.2f(%.2f-%.2f) %.2fx/%.2fx pk=%.2fMB id=%s" % (
                    k, v["t"]["med"], v["t"]["min"], v["t"]["max"], v["x_vs_orig"], v["x_vs_staged"], v["peak"] / 1e6, v["identical"])
                    for k, v in r["k"].items()), flush=True)
    json.dump(out, open("results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
```

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-21_1055_gamma-chunk-split-tree-vs-abs-mean.md.

close: 5537fc899f782a7d
