---
local: true
date: 2026-10-21
wave: "2026-10-21 10:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (gamma step of the native f32 quantize path on overlay hosts under bitnetd MemoryMax)"
chief_assignment: "pair_A Live beta step 5, non-quant-error side; pair_B JuniorOS overlay remaining gaps (Quant Chief 10:16 MT, last_tips.json)"
verdict: viable
tags: [ternary, osai, findings, junioros-overlay, gamma, absmean, pairwise-sum, numpy, quantize, memory]
---

# Chunked abs-mean gamma on numpy's own split tree: same bits, about 100-270x less scratch, 1.5x faster

Previous note: [[2026-10-21_0555_threshold-ternarize-vs-rint-divide]]

## Question

The 05:55 note left gamma (`np.abs(w).mean(dtype=float32)`) as the largest piece of the quantize path once threshold ternarize landed. Is a fused or buffered gamma at least 1.5x faster or smaller than the current one while keeping gamma bit-identical (or at least flipping zero trits under the exact cutoff)?

## Method

- Tensors: 2560x2560 and 2560x6912 (2B4T attention and FFN sizes), Gaussian and Student-t(4) weights at scale 0.02, new seeds 1055 and 2055, f32.
- Current gamma: `np.float32(np.abs(w).mean(dtype=float32)) + 1e-7`. It allocates a full n-element f32 copy of |w| and then runs numpy's pairwise sum over it.
- Variants: (2) abs into a preallocated n-buffer, (3) sign bit masked over a uint32 view into the same buffer, (4) abs in 65,536-element (256 KiB, L2-sized) chunks with a flat f32 sum of the chunk sums, (5) **abs in chunks along numpy's own pairwise split** (new): recurse with numpy's rule `n2 = n // 2; n2 -= n2 % 8` until a node has at most 65,536 elements, abs that node into the 256 KiB buffer, reduce it with `np.add.reduce` (the same pairwise code over the same values), and add left + right in f32, so the summation tree is the current one exactly; (6) `sign(w) . w` through BLAS; (7) block-128 sums via reshape.
- Checks: every variant's gamma in f32 ulps against the current gamma, cutoff T from the 05:55 note's `nextafter` rule, trit flips under the threshold ternarize, relative error against an f64 abs mean. Identity sweep for the split-tree variant: 422 arrays (200 random sizes from 1 to 3M plus 1, 7, 8, 9, 127, 128, 129 and sizes around the chunk), both distributions.
- Timing: 15 repeats after one warm-up, median (min-max) ms with `time.perf_counter`; peak traced allocation with `tracemalloc` (preallocated buffers are outside the trace, so their size is stated separately). numpy 2.5.4, 8-core box, nothing bound, no network, no installs. The script ran twice; tables are run 2, run 1 is summarised in the verdict.

Command (script kept at `/workspace/tbench/r20261021_1055/gamma_fused_bench.py`, reproduced in full below):

```
cd /workspace/tbench/r20261021_1055 && python3 gamma_fused_bench.py
```

## Results

### A. Gamma time (ms, median (min-max), 15 repeats) and speedup over the current gamma

| shape | kind | seed | abs().mean (current) | abs into prealloc n-buffer | u32 sign-mask into n-buffer | 256 KiB chunks, flat partial sum | 256 KiB chunks, numpy split tree (new) | sign(w) . w BLAS | block-128 reshape sums |
|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 1055 | 3.07 (2.81-4.14), 1.00x | 2.79 (2.74-4.11), 1.10x | 2.85 (2.80-3.07), 1.08x | 2.19 (2.06-2.38), 1.40x | 2.32 (2.17-2.52), 1.32x | 4.26 (4.18-4.41), 0.72x | 3.26 (3.21-3.38), 0.94x |
| 2560x2560 | gauss | 2055 | 2.89 (2.77-4.50), 1.00x | 2.92 (2.82-4.25), 0.99x | 2.90 (2.85-3.01), 1.00x | 2.07 (2.01-2.43), 1.40x | 2.17 (2.13-2.45), 1.33x | 4.24 (4.19-4.47), 0.68x | 3.30 (3.20-3.39), 0.88x |
| 2560x2560 | t4 | 1055 | 2.82 (2.78-4.94), 1.00x | 2.94 (2.86-4.24), 0.96x | 2.94 (2.85-3.07), 0.96x | 2.13 (1.99-2.33), 1.32x | 2.25 (2.18-2.44), 1.25x | 4.35 (4.24-4.84), 0.65x | 3.35 (3.23-3.50), 0.84x |
| 2560x2560 | t4 | 2055 | 3.69 (2.89-4.44), 1.00x | 3.04 (2.79-4.52), 1.22x | 2.99 (2.83-3.29), 1.24x | 2.31 (2.11-2.50), 1.60x | 2.41 (2.22-2.51), 1.53x | 4.35 (4.24-5.29), 0.85x | 3.28 (3.19-3.40), 1.12x |
| 2560x6912 | gauss | 1055 | 11.64 (11.29-17.52), 1.00x | 8.20 (8.03-8.67), 1.42x | 8.03 (7.92-8.33), 1.45x | 5.61 (5.39-6.17), 2.07x | 6.42 (6.25-6.86), 1.81x | 11.93 (11.27-14.77), 0.97x | 8.86 (8.73-9.45), 1.31x |
| 2560x6912 | gauss | 2055 | 11.11 (10.09-18.48), 1.00x | 7.85 (7.71-7.99), 1.42x | 7.89 (7.79-8.01), 1.41x | 5.68 (5.39-6.20), 1.96x | 6.38 (6.09-6.72), 1.74x | 11.86 (11.46-15.39), 0.94x | 9.05 (8.81-9.34), 1.23x |
| 2560x6912 | t4 | 1055 | 10.34 (10.02-17.26), 1.00x | 7.99 (7.70-8.20), 1.29x | 8.08 (7.79-8.70), 1.28x | 5.55 (5.42-5.84), 1.86x | 6.75 (6.23-7.02), 1.53x | 11.75 (11.34-14.95), 0.88x | 9.14 (8.89-9.70), 1.13x |
| 2560x6912 | t4 | 2055 | 10.01 (9.91-17.08), 1.00x | 7.56 (7.45-7.70), 1.32x | 7.67 (7.56-7.74), 1.31x | 5.44 (5.33-5.54), 1.84x | 6.23 (6.06-6.38), 1.61x | 11.58 (11.22-13.72), 0.86x | 8.66 (8.51-9.14), 1.16x |

### B. Gamma identity, trit flips and scratch memory

| shape | kind | seed | n | gamma | trit sha16 | split tree ulps / flips | flat chunks ulps / flips | BLAS dot ulps / flips | block-128 ulps / flips | peak traced MB current / split tree (+0.26 MB fixed buffer) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 1055 | 6553600 | 0.01596451 | `4777cffdfa6ec7ca` | +0 / 0 | +0 / 0 | -13 / 2 | +0 / 0 | 26.2 / 0.0012 |
| 2560x2560 | gauss | 2055 | 6553600 | 0.01595205 | `72cb960311e097ff` | +0 / 0 | -1 / 0 | -6 / 1 | +0 / 0 | 26.2 / 0.0012 |
| 2560x2560 | t4 | 1055 | 6553600 | 0.01999951 | `9f4ee1b4dc7a2f51` | +0 / 0 | +0 / 0 | -4 / 2 | +0 / 0 | 26.2 / 0.0012 |
| 2560x2560 | t4 | 2055 | 6553600 | 0.01999492 | `166e4d4c249d8e98` | +0 / 0 | -1 / 0 | -8 / 2 | +0 / 0 | 26.2 / 0.0012 |
| 2560x6912 | gauss | 1055 | 17694720 | 0.0159586 | `90b2bc7b701015b1` | +0 / 0 | -1 / 0 | -46 / 28 | +0 / 0 | 70.8 / 0.0014 |
| 2560x6912 | gauss | 2055 | 17694720 | 0.01595533 | `16390ccc488e8ce0` | +0 / 0 | +0 / 0 | -40 / 15 | +0 / 0 | 70.8 / 0.0014 |
| 2560x6912 | t4 | 1055 | 17694720 | 0.01999875 | `33d23f6fe06df247` | +0 / 0 | +0 / 0 | -70 / 38 | -1 / 0 | 70.8 / 0.0014 |
| 2560x6912 | t4 | 2055 | 17694720 | 0.01999967 | `01f8bebf251dede3` | +0 / 0 | +1 / 1 | -71 / 35 | -1 / 3 | 70.8 / 0.0014 |

Split-tree identity sweep: 0 mismatches over 422 arrays (both runs).

## Verdict

**Viable, mainly on memory.** The split-tree chunked gamma is bit-identical to the current gamma on all 8 tensors and all 422 sweep arrays, so T, the trits, the SWAR-packed bytes and any close key over them do not change. It replaces the current n-element f32 scratch copy (26.2 MB at 2560x2560, 70.8 MB at 2560x6912) with one fixed 0.26 MB buffer, which is 100x and 270x less transient memory on a MemoryMax-capped bitnetd host. Speed is 1.53x (1.25-1.81x) median over the 8 tensors in run 2 and 1.65x (1.21-1.83x) in run 1: clearly over 1.5x on the FFN-sized tensors (1.53-1.83x) and under it on 2560x2560 (1.21-1.53x).

- **The flat chunk sum is faster but not safe:** 1.72x (1.32-2.07x), but it moves gamma by one ulp on 4 of 8 tensors and flipped 1 trit on 2560x6912 t4 seed 2055 in both runs. Its error against f64 is no worse, but it breaks bit-identity with the current rule, so it would change packed bytes and closes.
- **A preallocated n-buffer or the uint32 sign mask** keeps the bits but only gives 1.26x and still needs the full n-element buffer.
- **BLAS `sign(w) . w` is a loss:** 0.86x and 4 to 71 ulps off, with up to 38 trit flips.
- **Block-128 reshape sums** are 1.13x and flip up to 3 trits; they are only right if the I2_S 128-block key's per-block gamma is wanted, which the per-128-block scale finding ([[2026-10-19_1455_block128-opt-scale-vs-per-tensor-opt]], JuniorLLM e9e2772) already called not viable.
- **Preconditions:** the split rule mirrors numpy's float pairwise sum (`pairwise_sum` with an 8-wide unrolled block of 128); the 422-array sweep is the check, and it should be re-run if numpy is upgraded on the overlay image. The array must be C-contiguous (it is reshaped flat).
- **What it does not change:** gamma bits, trits, the JTR1/I2_S format, Trit Pack's and Winsor's keys and the close rules. With this and the 05:55 threshold ternarize, the numpy fallback quantize path never holds more than the weights, the int8 trits and the packed bytes plus 0.26 MB. The C path (`absmean.c`) was not measured.
- **Caveat:** synthetic weights and single-thread numpy on this box; time depends on element count, not values.

## This wave's pair files

- pair_A (Live beta step 5, non-quant-error side): Trit Pack [JuniorLLM 5b35e4d scripts/backfill_archive_attr_i2s_key.py](https://github.com/cloudcover95/JuniorLLM/commit/5b35e4d) (archive-attributes I2_S key). Winsor's 10:20 push was not seen in the local copies.
- pair_B (JuniorOS overlay remaining gaps): Handshake [JuniorLLM 29c8d69 ports/overlay_juniorctl_install_import_closure_handshake.py](https://github.com/cloudcover95/JuniorLLM/commit/29c8d69) (installed juniorctl import-closure handshake). Side-Commit's 10:35 push was not seen in the local copies.

## Next question

With gamma, ternarize and pack all lean, does a single blocked pass that computes gamma on the split tree and then thresholds and SWAR-packs each block while it is still in L2 (two passes over w instead of four) give another 1.5x end to end on the 2560x6912 quantize path, with bit-identical packed bytes?

## Script

```python
#!/usr/bin/env python3
"""f32 abs-mean gamma: current np.abs(w).mean(dtype=f32) vs fused / buffered variants (JuniorOS overlay quantize path).
Question: is any variant >= 1.5x faster or smaller while keeping gamma bit-identical, or flipping zero trits
under the exact threshold cutoff from the 05:55 note? numpy + stdlib only, offline, no sockets, no installs."""
import json, time, tracemalloc, hashlib
import numpy as np

SHAPES = [(2560, 2560), (2560, 6912)]
KINDS = ["gauss", "t4"]
SEEDS = [1055, 2055]
REP = 15
EPS = np.float32(1e-7)
MASK = np.uint32(0x7FFFFFFF)
CH = 1 << 16  # 65536 f32 = 256 KiB chunk (fits L2)

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

# 1. current
def g_cur(w, buf=None): return np.float32(np.float32(np.abs(w).mean(dtype=np.float32)) + EPS)
# 2. reuse a preallocated n-buffer: identical array, identical pairwise sum -> bit-identical
def g_buf(w, buf): np.abs(w, out=buf); return np.float32(np.float32(buf.mean(dtype=np.float32)) + EPS)
# 3. sign-bit mask over uint32 view into the buffer
def g_mask(w, buf):
    np.bitwise_and(w.view(np.uint32), MASK, out=buf.view(np.uint32)); return np.float32(np.float32(buf.mean(dtype=np.float32)) + EPS)
# 4. chunked fused: abs into a 256 KiB L2 buffer, f32 partial sums, final f32 sum of partials
def g_chunk(w, cbuf):
    f = w.reshape(-1); n = f.size; parts = np.empty((n + CH - 1) // CH, np.float32)
    for i, s in enumerate(range(0, n, CH)):
        c = cbuf[: min(CH, n - s)]; np.abs(f[s:s + CH], out=c); parts[i] = np.add.reduce(c, dtype=np.float32)
    return np.float32(np.float32(np.add.reduce(parts, dtype=np.float32) / np.float32(n)) + EPS)
# 5. BLAS: sum|w| = sign(w) . w  (sgemv / sdot, multithreaded)
def g_dot(w, buf):
    f = w.reshape(-1); np.sign(f, out=buf); return np.float32(np.float32(np.dot(buf, f) / np.float32(f.size)) + EPS)
# 6. row-block abs-sum via reshape (block-128 sums, then sum of block sums)
def g_blk(w, buf):
    np.abs(w, out=buf); b = buf.reshape(-1, 128).sum(axis=1, dtype=np.float32)
    return np.float32(np.float32(b.sum(dtype=np.float32) / np.float32(w.size)) + EPS)
# 7. chunked fused that mirrors numpy's pairwise split (n2 = n//2 - (n//2 % 8)) down to nodes <= CH,
#    reduces each leaf with np.add.reduce (same pairwise code on the same values) and adds node results
#    left + right in f32, so the sum is the same tree as np.abs(w).mean -> bit-identical by construction.
def _tree(f, s, n, cbuf):
    if n <= CH:
        c = cbuf[:n]; np.abs(f[s:s + n], out=c); return np.add.reduce(c, dtype=np.float32)
    n2 = n // 2; n2 -= n2 % 8
    return np.float32(_tree(f, s, n2, cbuf) + _tree(f, s + n2, n - n2, cbuf))
def g_tree(w, cbuf):
    f = w.reshape(-1); n = f.size
    return np.float32(np.float32(_tree(f, 0, n, cbuf) / np.float32(n)) + EPS)
# exact reference (f64) for error
def g_ref(w): return float(np.abs(w.astype(np.float64)).mean()) + 1e-7

def cutoff(g):
    g = np.float32(g); half = np.float32(0.5); t = np.float32(g * half)
    while np.float32(t / g) > half: t = np.nextafter(t, np.float32(0))
    while np.float32(np.nextafter(t, np.float32(np.inf)) / g) <= half: t = np.nextafter(t, np.float32(np.inf))
    return np.float32(t)

def tern(w, T): return (w > T).view(np.int8) - (w < -T).view(np.int8)

def ulps(a, b): return int(np.int64(np.float32(a).view(np.int32)) - np.int64(np.float32(b).view(np.int32)))

def main():
    out = {"numpy": np.__version__, "rep": REP, "chunk": CH, "rows": []}
    K = [("current", g_cur, None), ("prealloc_buf", g_buf, "n"), ("u32_mask_buf", g_mask, "n"),
         ("chunk256k_fused", g_chunk, "c"), ("chunk_tree_exact", g_tree, "c"), ("sign_dot_blas", g_dot, "n"), ("block128_reshape", g_blk, "n")]
    # identity sweep: odd and even sizes, f32 normal and t4, against the current gamma bit for bit
    rng = np.random.default_rng(7); cb = np.empty(CH, np.float32); idn = {"n": 0, "mismatch": 0}
    for n in list(rng.integers(1, 3_000_000, 200)) + [1, 7, 8, 9, 127, 128, 129, CH - 1, CH, CH + 1, 2 * CH + 3]:
        for kind in KINDS:
            w = make(kind, (int(n),), rng); idn["n"] += 1; idn["mismatch"] += int(g_cur(w) != g_tree(w, cb))
    out["tree_identity"] = idn; print("tree identity", idn, flush=True)
    for shape in SHAPES:
        for kind in KINDS:
            for seed in SEEDS:
                w = make(kind, shape, np.random.default_rng(seed))
                nbuf = np.empty(w.size, np.float32); cbuf = np.empty(CH, np.float32)
                g0 = g_cur(w); T0 = cutoff(g0); q0 = tern(w, T0); ref = g_ref(w)
                r = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "n": int(w.size), "gamma": float(g0),
                     "cur_rel_err_vs_f64": abs(float(g0) - ref) / ref, "sha16": hashlib.sha256(q0.tobytes()).hexdigest()[:16], "k": {}}
                for name, f, bk in K:
                    b = cbuf if bk == "c" else (nbuf if name == "sign_dot_blas" else nbuf.reshape(shape))
                    g = f(w, b); T = cutoff(g)
                    flips = int((tern(w, T) != q0).sum()) if g != g0 else 0
                    r["k"][name] = {"t": tstat(lambda: f(w, b)), "peak": peak(lambda: f(w, b)), "gamma_ulps": ulps(g, g0),
                                    "T_ulps": ulps(T, T0), "flips": flips, "rel_err_vs_f64": abs(float(g) - ref) / ref}
                base = r["k"]["current"]["t"]["med"]
                for name in r["k"]: r["k"][name]["speedup"] = base / r["k"][name]["t"]["med"]
                out["rows"].append(r)
                print(r["shape"], kind, seed, " | ".join("%s %.2f(%.2f-%.2f) %.2fx pk=%.1fMB du=%d flips=%d re=%.1e" % (
                    k, v["t"]["med"], v["t"]["min"], v["t"]["max"], v["speedup"], v["peak"] / 1e6, v["gamma_ulps"], v["flips"], v["rel_err_vs_f64"])
                    for k, v in r["k"].items()), flush=True)
    json.dump(out, open("results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
```

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-21_0555_threshold-ternarize-vs-rint-divide.md.

close: 38ea5289e7bec4c2
