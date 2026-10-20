---
local: true
date: 2026-10-19
wave: "2026-10-19 18:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 5: `python scripts/backfill_check.py` exits 0 on a fresh clone (gamma format of the step-5 trit packs: power-of-two exponent vs f32/fp16 gamma)"
chief_assignment: "pair_A Live beta step 5 fresh-clone / backfill fidelity; pair_B JuniorOS overlay hardened bitnetd.service"
verdict: not-viable
tags: [ternary, osai, findings, live-beta, step5, backfill, winsor, gamma, power-of-two, e8m0, i2s, numpy]
---

# Power-of-two (E8 exponent) ternary gamma vs f32 / fp16 gamma: dequant speed, bytes, weight error and the step-5 Winsor verdict

Previous note: [[2026-10-19_1455_block128-opt-scale-vs-per-tensor-opt]]

## Question
Both pair_A step-5 files carry a float gamma next to their trits. Trit Pack's JTR1 header stores an f32 absmean gamma, and Winsor's substance gate quantizes log2 file sizes with float absmean, drop and clip gammas. If the gamma is snapped to a power of two (one signed exponent byte, like an OCP MX E8M0 scale), dequant becomes an exponent add (`ldexp`), or a shift on an integer accumulator, instead of a float multiply. The scale also takes 1 byte instead of 4 (f32) or 2 (fp16). Is that at least 1.5x faster or smaller than the f32/fp16 gamma at acceptable error?

The question has two sides:
- (A) BitNet-style absmean ternary weights at 2B4T shapes: weight error, bytes, dequant speed and packed matvec speed.
- (B) Winsor's step-5 gate: does a pow2 gamma keep the same hollow/thin/substantive path trits and the same exit verdict?

Viability bar:
- Speed or bytes must be at least 1.5x better end to end.
- Weight SSE must stay within RATE (5%) of the f32 gamma it replaces.
- The step-5 verdict must agree on at least 1 - RATE = 95% of trees, and be identical on the real tree.

## Method
- python3 3.13.5 + numpy 2.5.4 only, offline, no sockets, no installs, 8 cores, `OMP_NUM_THREADS=8`. Run on the box 2026-10-19 19:05 to 19:07 MT, with the supplement at about 19:09 MT.
- Part A, synthetic float32 weights:
  - Shapes 2560 x 2560 and 2560 x 6912. Kinds `gauss` (sigma 0.02) and `t4` (Student-t df 4). Seeds 1855 and 2855.
  - SSE/E is scale-free, so only the fractional part of log2(gamma) matters to a pow2 snap. Each tensor is rescaled so that log2(absmean) has fraction 0, 0.125, ..., 0.875, and every scheme is scored at each fraction.
  - Schemes:
    - `f32`: BitNet absmean, gamma = mean|w|, q = clip(round(w/gamma), -1, 1), W_hat = gamma q. This is the format Trit Pack's JTR1 and bitnet.cpp I2_S store.
    - `fp16`: the same with gamma cast to fp16.
    - `pow2 keep`: same codes, gamma snapped to 2^round(log2 gamma).
    - `pow2 recode`: codes re-rounded against 2^floor or 2^ceil, keeping the lower SSE.
    - `pow2 best` (supplement): 2^e scanned over floor-1 to floor+2, codes re-rounded. This is the best pow2 scale, since for a fixed scale s, round/clip at s/2 is the optimal code.
    - `f32 exact opt` (supplement): the sorted-prefix optimum from the 10:55 and 14:55 notes.
  - Speed: 31 timed repeats (15 for the full-tensor and matvec rows) after one warm-up. Median, min-max and IQR reported. Paths timed:
    - f32 multiply vs `np.ldexp` on the B=1 output vector, the B=8 output block, an int32 accumulator converted to f32, and the full f32 trit tensor.
    - int32 `>> 6` vs convert-and-multiply (a lossy fixed-point stand-in for a shift dequant).
    - The end-to-end packed I2_S B=1 matvec: 2-bit MSB-first bytes through the 256 x 4 LUT, `@ x`, then the scale.
  - Bytes: 2 bits per weight plus one scale per tensor (4 / 2 / 1 B), and also per 128-trit block.
- Part B, Winsor's gate, loaded from the local copy of JuniorQuant dbafb2e `src/live_beta_step5_backfill_substance_gate.py`:
  - Its own `gate()` is run with its `gammas()` wrapped so that absmean, drop and clip are snapped to 2^round(log2 g). tau is left alone.
  - Compared on three inputs:
    - The real local JuniorLLM tree (`/workspace/JuniorLLM-1819`, HEAD 5032022), sizes from its own `scan()`.
    - The gate's selftest scenarios: substantive, whitespace-only receipt, `#TODO` stub STATE.md, missing file.
    - 20,000 synthetic 12-path trees: log-normal sizes fitted to the real tree (ln mu 7.137, sd 0.817), 15% with one truncated 1 to 12 byte stub and 5% with one hollow path, seed 1855.

Command (run from `/workspace/tbench/r20261019_1855` on the box):

```
OMP_NUM_THREADS=8 python3 pow2_gamma_bench.py > results.json 2> progress.log
OMP_NUM_THREADS=8 python3 supp_exact_opt.py
```

## Results
### A1. Weight error, SSE / ||W||^2 (mean over 2 shapes x 2 seeds; ratio columns min-max over the 4 tensors)

| kind | log2 frac | f32 absmean | fp16 absmean | pow2 keep | pow2 recode | pow2 best | f32 exact opt | keep / f32 | best / f32 | best / f32 opt | y rel_l2 f32 / keep / recode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss | 0.000 | 0.2634 | 0.2634 | 0.2634 | 0.2383 | 0.2299 | 0.1902 | 1.000 | 0.873 | 1.208-1.209 | 0.512 / 0.512 / 0.487 |
| gauss | 0.125 | 0.2634 | 0.2633 | 0.2911 | 0.2077 | 0.2077 | 0.1902 | 1.105 | 0.789 | 1.092 | 0.512 / 0.538 / 0.456 |
| gauss | 0.250 | 0.2634 | 0.2633 | 0.3218 | 0.1947 | 0.1947 | 0.1902 | 1.222 | 0.739 | 1.023-1.024 | 0.512 / 0.567 / 0.448 |
| gauss | 0.375 | 0.2634 | 0.2634 | 0.3545 | 0.1902 | 0.1902 | 0.1902 | 1.346 | 0.722 | 1.000 | 0.512 / 0.595 / 0.439 |
| gauss | 0.500 | 0.2634 | 0.2634 | 0.3450 | 0.1934 | 0.1934 | 0.1902 | 0.818-1.474 | 0.735 | 1.017 | 0.512 / 0.583 / 0.441 |
| gauss | 0.625 | 0.2634 | 0.2634 | 0.2138 | 0.2033 | 0.2033 | 0.1902 | 0.812 | 0.772 | 1.069 | 0.512 / 0.458 / 0.450 |
| gauss | 0.750 | 0.2634 | 0.2634 | 0.2228 | 0.2189 | 0.2189 | 0.1902 | 0.846 | 0.831 | 1.151 | 0.512 / 0.469 / 0.466 |
| gauss | 0.875 | 0.2634 | 0.2634 | 0.2401 | 0.2392 | 0.2392 | 0.1902 | 0.911-0.912 | 0.908 | 1.257-1.258 | 0.512 / 0.487 / 0.487 |
| t4 | 0.000 | 0.4086 | 0.4086 | 0.4086 | 0.3166 | 0.3166 | 0.3137 | 1.000 | 0.775 | 1.009-1.010 | 0.640 / 0.640 / 0.558 |
| t4 | 0.125 | 0.4086 | 0.4085 | 0.4332 | 0.3137 | 0.3137 | 0.3137 | 1.060 | 0.768 | 1.000 | 0.640 / 0.659 / 0.553 |
| t4 | 0.250 | 0.4086 | 0.4085 | 0.4597 | 0.3157 | 0.3157 | 0.3137 | 1.125 | 0.773 | 1.006-1.007 | 0.640 / 0.678 / 0.556 |
| t4 | 0.375 | 0.4086 | 0.4086 | 0.4872 | 0.3224 | 0.3224 | 0.3137 | 1.192-1.193 | 0.789 | 1.028 | 0.640 / 0.699 / 0.564 |
| t4 | 0.500 | 0.4086 | 0.4086 | 0.5152 | 0.3333 | 0.3333 | 0.3137 | 1.260-1.262 | 0.816 | 1.063 | 0.640 / 0.718 / 0.575 |
| t4 | 0.625 | 0.4086 | 0.4086 | 0.3568 | 0.3479 | 0.3479 | 0.3137 | 0.873-0.874 | 0.852 | 1.109-1.110 | 0.640 / 0.598 / 0.591 |
| t4 | 0.750 | 0.4086 | 0.4086 | 0.3691 | 0.3657 | 0.3385 | 0.3137 | 0.903 | 0.829 | 1.078-1.080 | 0.640 / 0.608 / 0.605 |
| t4 | 0.875 | 0.4086 | 0.4086 | 0.3868 | 0.3861 | 0.3249 | 0.3137 | 0.947 | 0.795 | 1.035-1.036 | 0.640 / 0.622 / 0.621 |

Averaged over a uniform log2 fraction:
- `pow2 keep`: 1.069x (gauss) and 1.045x (t4) of the f32 absmean SSE, but within 5% on only 53% / 50% of fractions, and up to 1.47x worse.
- `pow2 recode`: 0.80x / 0.83x of the f32 absmean SSE, under it at every fraction, because absmean itself is not the ternary optimum.
- `pow2 best`: 1.102x / 1.041x of the f32 exact optimum on average, worst 1.258x / 1.110x, and within 5% on only 37.5% / 62.5% of fractions.
- fp16 gamma is indistinguishable from f32 (4th digit).

### A2. Speed (median (min-max) [IQR] over 4 tensors per shape; ratio = multiply time / pow2 time, above 1 means pow2 is faster)

| shape | path | unit | f32 / fp16 gamma multiply | pow2 gamma | ratio |
|---|---|---|---|---|---|
| 2560x2560 | B=1 output scale (f32 vec) | us | 1.01 (0.81-3.59) [0.92-1.12] | ldexp 1.28 (1.18-4.82) [1.27-1.30] | 0.78x (0.73-0.84) |
| 2560x2560 | B=8 output scale | us | 2.91 (2.31-3.38) [2.89-2.94] | ldexp 5.33 (5.01-6.02) [5.28-5.42] | 0.54x (0.45-0.56) |
| 2560x2560 | int32 acc -> f32 -> scale | us | 1.72 (1.34-2.78) [1.70-1.76] | ldexp 1.97 (1.53-2.92) [1.94-2.00] | 0.87x (0.85-0.90) |
| 2560x2560 | int32 acc, cvt*mul vs `>> 6` (lossy) | us | 1.72 (1.34-2.78) [1.70-1.76] | shift 1.10 (0.86-14.27) [1.08-1.12] | 1.56x (1.52-1.59) |
| 2560x2560 | full-tensor dequant of f32 trits | ms | 1.75 (1.49-5.98) [1.69-1.81] | ldexp 1.75 (1.48-1.88) [1.75-1.78] | 1.00x (0.85-1.12) |
| 2560x2560 | packed I2_S B=1 matvec, end to end | ms | 18.21 (16.01-24.77) [17.18-19.17] | ldexp 18.59 (15.93-31.93) [16.99-19.65] | 0.99x (0.97-1.02) |
| 2560x6912 | B=1 output scale (f32 vec) | us | 0.91 (0.73-2.60) [0.86-0.96] | ldexp 1.23 (1.06-16.53) [1.21-1.25] | 0.73x (0.64-0.81) |
| 2560x6912 | B=8 output scale | us | 2.71 (2.12-3.75) [2.70-2.74] | ldexp 5.14 (4.48-5.43) [5.07-5.21] | 0.52x (0.48-0.66) |
| 2560x6912 | int32 acc -> f32 -> scale | us | 1.55 (1.37-2.23) [1.54-1.59] | ldexp 1.82 (1.67-2.28) [1.80-1.87] | 0.85x (0.82-0.89) |
| 2560x6912 | int32 acc, cvt*mul vs `>> 6` (lossy) | us | 1.55 (1.37-2.23) [1.54-1.59] | shift 1.05 (0.92-11.60) [1.04-1.10] | 1.54x (1.41-1.55) |
| 2560x6912 | full-tensor dequant of f32 trits | ms | 9.41 (7.48-15.89) [8.71-10.43] | ldexp 10.54 (8.08-14.97) [10.16-10.87] | 0.93x (0.78-1.01) |
| 2560x6912 | packed I2_S B=1 matvec, end to end | ms | 57.51 (49.93-77.95) [55.06-59.68] | ldexp 59.13 (50.49-82.68) [56.39-60.23] | 0.99x (0.92-1.07) |

The matvec outputs from multiply-by-2^e and from `ldexp` agree to rel_l2 2.9e-8.

### A3. Bytes

| shape | codes only | + f32 gamma | + fp16 gamma | + E8 exponent | E8 vs f32 / fp16 | per-128-block f32 / fp16 / E8 | block E8 vs f32 / fp16 | fp16 weights |
|---|---|---|---|---|---|---|---|---|
| 2560x2560 | 1,638,400 | 1,638,404 | 1,638,402 | 1,638,401 | 1.0000018x / 1.0000006x | 1,843,200 / 1,740,800 / 1,689,600 | 1.091x / 1.030x | 13,107,200 |
| 2560x6912 | 4,423,680 | 4,423,684 | 4,423,682 | 4,423,681 | 1.0000007x / 1.0000002x | 4,976,640 / 4,700,160 / 4,561,920 | 1.091x / 1.030x | 35,389,440 |

### B. Winsor step-5 gate, f32 gammas vs pow2-snapped gammas

| input | f32 gammas (absmean / drop / clip) | pow2 gammas | verdict f32 -> pow2 | path trits equal | q trits equal | quant-error rel (absmean) f32 -> pow2 |
|---|---|---|---|---|---|---|
| real JuniorLLM tree (5032022, 12 paths, 333 to 4,067 non-ws bytes) | 0.977 / 0.977 / 0.970 | 1 / 1 / 1 | +1 -> +1 (exit 0) | yes (12/12 +1) | yes | 0.181 -> 0.173 |
| selftest substantive | 1.205 / 1.205 / 1.194 | 1 / 1 / 1 | +1 -> +1 | yes | yes | 0.095 -> 0.171 |
| selftest whitespace-only receipt | 1.364 / 1.364 / 1.351 | 1 / 1 / 1 | -1 -> -1 | yes | yes | 0.058 -> 0.171 |
| selftest `#TODO` stub STATE.md | 1.930 / 1.930 / 1.244 | 2 / 2 / 1 | 0 -> 0 | yes | no (ports/E.md q -1,-1,-1 -> 0,0,-1) | 0.462 -> 0.454 |
| selftest missing x/f.py | 1.097 / 1.097 / 1.066 | 1 / 1 / 1 | -1 -> -1 | yes | yes | 0.111 -> 0.149 |
| 20,000 synthetic trees | - | - | agree 87.1% (2,573 flip) | 98.48% of path trits | 95.63% of q trits | ratio median 0.966, p95 1.43, max 2.84 |

On the synthetic trees:
- Verdict counts (-1 / 0 / +1) were 969 / 7,635 / 11,396 with f32 gammas and 969 / 7,582 / 11,449 with pow2. All flips are between thin (0) and substantive (+1), in both directions; hollow does not depend on gamma.
- Mean gate time was 205 us per tree (f32) vs 198 us (pow2), pure Python, one pass over 20,000 trees.

## Verdict
**Not viable.** A power-of-two (E8 exponent) gamma never gives a 1.5x end-to-end gain:
- **Speed:** the packed I2_S matvec runs at 0.99x (0.92 to 1.07x). In numpy, `ldexp` is slower than a float multiply: 0.52x to 0.87x on output vectors and 0.93x to 1.00x on full-tensor dequant.
- **The one sub-step that clears 1.5x:** an int32 `>> 6` shift instead of convert-and-multiply is 1.54 to 1.56x faster, but it is lossy fixed point, not the same float result, and it saves about 0.6 us inside an 18 to 58 ms matvec (under 0.01%).
- **Bytes:** about 1.000002x per tensor. Even with one scale per 128-trit block, an E8 scale is only 1.09x smaller than f32 and 1.03x smaller than fp16.
- **Error:** snapping gamma and keeping the absmean codes costs up to 1.47x SSE. Re-rounding the codes against the best pow2 scale actually beats the f32 absmean gamma (0.72x to 1.00x SSE), but that comes from searching a few scales, not from the power of two. Against the f32 exact optimum the best pow2 scale is 1.00x to 1.26x worse (mean 1.10x Gaussian, 1.04x t4), which misses the 5% bar on most log2 fractions.
- **Step 5:** snapping Winsor's gammas leaves the real-tree verdict (+1, exit 0) and all four selftest verdicts unchanged. But it flips the thin/substantive verdict on 12.9% of synthetic 12-path trees, well over RATE 0.05, so the gate needs its float gammas.

Keep the f32 gamma in Trit Pack's JTR1 header and float gammas in Winsor's gate. fp16 is the only lossless-in-practice shrink (identical SSE to the 4th digit), and it only matters with per-block scales.

Caveats:
- Synthetic weights only, with no GGUF on disk.
- numpy has no fused shift-dequant kernel. In a C/NEON kernel an exponent add could be free, but the scale is applied once per output row, so it is still a vanishing share of the matvec.
- The step-5 synthetic tree distribution is fitted to one real tree.

## This wave's pair files
- Pair A (Live beta step 5, fresh-clone / backfill fidelity):
  - Trit Pack JuniorLLM 5032022 [`scripts/backfill_fresh_clone_index_i2s_key.py`](https://github.com/cloudcover95/JuniorLLM/blob/5032022/scripts/backfill_fresh_clone_index_i2s_key.py): offline `.git/index` DIRC read, per-EXPECTED tracked/clean/untracked trits, JTR1 with an f32 absmean gamma plus an I2_S 128-block key. That is the gamma this note tests.
  - Winsor JuniorQuant dbafb2e [`src/live_beta_step5_backfill_substance_gate.py`](https://github.com/cloudcover95/JuniorQuant/blob/dbafb2e/src/live_beta_step5_backfill_substance_gate.py): log2 substance, absmean/drop/clip Winsor gammas, Laplace RATE-tail thin bound, hollow/thin/substantive trits. Its gate is what Part B runs.
- Pair B (JuniorOS overlay, hardened bitnetd.service):
  - Handshake JuniorLLM a6166de [`ports/overlay_seccomp_syscall_closure_handshake.py`](https://github.com/cloudcover95/JuniorLLM/blob/a6166de/ports/overlay_seccomp_syscall_closure_handshake.py).
  - No Side-Commit 18:35 push was found locally.

## Inline script
`pow2_gamma_bench.py`:

```python
#!/usr/bin/env python3
"""Power-of-two (E8 exponent) ternary gamma vs f32 / fp16 gamma.
Part A: BitNet absmean ternary codes at 2B4T shapes; error over the scale's
log2 fraction, bytes, and dequant / packed-matvec speed (multiply vs ldexp).
Part B: Winsor's Live beta step 5 substance gate with pow2-snapped gammas:
verdict and path-trit agreement on the real local JuniorLLM tree, the gate's
selftest scenarios, and 20,000 synthetic 12-path trees.
numpy only, offline, no sockets. Prints one JSON blob."""
import importlib.util, json, math, random, sys, tempfile, time
from pathlib import Path
import numpy as np

SHAPES = [(2560, 2560), (2560, 6912)]
KINDS = ["gauss", "t4"]
SEEDS = [1855, 2855]
FRACS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
REP = 31
B = 128
WINSOR = "/workspace/winsor-draft-20261019-1820/live_beta_step5_backfill_substance_gate.py"
TREE = "/workspace/JuniorLLM-1819"

def tstat(f, n=REP):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return {"med": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75)), "n": n}

def make(kind, shape, rng):
    if kind == "gauss":
        return rng.standard_normal(shape, dtype=np.float32) * 0.02
    return (rng.standard_t(4, size=shape) * 0.02).astype(np.float32)

def absmean_codes(w, s):
    return np.clip(np.rint(w / np.float32(s)), -1, 1).astype(np.int8)

def sse_rel(w64, E, q, s):
    d = w64 - q.astype(np.float64) * float(s)
    return float((d * d).sum() / E)

def pack2(t):
    c = (t.reshape(-1) + 1).astype(np.uint8).reshape(-1, 4)
    return (c[:, 0] << 6 | c[:, 1] << 4 | c[:, 2] << 2 | c[:, 3]).astype(np.uint8)  # I2_S MSB-first

LUT = np.array([[((b >> (6 - 2 * i)) & 3) - 1 for i in range(4)] for b in range(256)], dtype=np.float32)

def part_a():
    out = []
    for shape in SHAPES:
        n = shape[0] * shape[1]
        for kind in KINDS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                w0 = make(kind, shape, rng)
                x = rng.standard_normal(shape[1], dtype=np.float32)
                g0 = float(np.abs(w0).mean(dtype=np.float64))
                fr0 = math.log2(g0) - math.floor(math.log2(g0))
                err = []
                for fr in FRACS:  # shift the tensor so log2(absmean) has fraction fr (SSE/E is scale-free)
                    w = (w0 * np.float32(2.0 ** (fr - fr0))).astype(np.float32)
                    w64 = w.astype(np.float64); E = float((w64 * w64).sum())
                    yref = w64 @ x.astype(np.float64); yn = np.linalg.norm(yref)
                    g = float(np.abs(w).mean(dtype=np.float64))
                    g32 = np.float32(g); g16 = np.float16(g)
                    q = absmean_codes(w, g32)
                    e_near = int(np.rint(math.log2(g)))
                    row = {"frac": fr, "log2g": math.log2(g)}
                    def rec(name, qq, s):
                        row[name] = {"sse": sse_rel(w64, E, qq, s),
                                     "y": float(np.linalg.norm(qq.astype(np.float64) @ x.astype(np.float64) * float(s) - yref) / yn)}
                    rec("f32", q, g32)
                    rec("fp16", absmean_codes(w, g16), g16)
                    rec("pow2_keep", q, 2.0 ** e_near)                      # same codes, gamma snapped
                    best = None
                    for e in (math.floor(math.log2(g)), math.ceil(math.log2(g))):
                        qe = absmean_codes(w, 2.0 ** e)
                        s_ = sse_rel(w64, E, qe, 2.0 ** e)
                        if best is None or s_ < best[0]:
                            best = (s_, e, qe)
                    rec("pow2_recode", best[2], 2.0 ** best[1])             # best of floor/ceil, codes re-rounded
                    qd = q.astype(np.float64).ravel(); lsq = float(w64.ravel() @ qd / (qd @ qd))
                    rec("f32_lsq", q, lsq)                                  # optimal float for absmean codes
                    err.append(row)
                # speed, at frac 0.5 (worst pow2 case; speed is data-independent)
                w = (w0 * np.float32(2.0 ** (0.5 - fr0))).astype(np.float32)
                g32 = np.float32(np.abs(w).mean(dtype=np.float64)); e = int(np.rint(math.log2(float(g32))))
                q = absmean_codes(w, g32); T32 = q.astype(np.float32); pk = pack2(q)
                acc = (T32 @ x).astype(np.float32)
                acc8 = np.ascontiguousarray(np.tile(acc, (8, 1)))
                acci = np.rint(acc * 1000).astype(np.int32)
                ei = np.int32(e)
                assert np.allclose(np.ldexp(acc, ei), acc * np.float32(2.0 ** e))
                sp = {
                    "out_mul": tstat(lambda: acc * g32), "out_ldexp": tstat(lambda: np.ldexp(acc, ei)),
                    "out8_mul": tstat(lambda: acc8 * g32), "out8_ldexp": tstat(lambda: np.ldexp(acc8, ei)),
                    "outi_cvt_mul": tstat(lambda: acci.astype(np.float32) * g32),
                    "outi_cvt_ldexp": tstat(lambda: np.ldexp(acci.astype(np.float32), ei)),
                    "outi_shift": tstat(lambda: acci >> 6),
                    "deq_mul": tstat(lambda: T32 * g32, 15), "deq_ldexp": tstat(lambda: np.ldexp(T32, ei), 15),
                    "mv_mul": tstat(lambda: (LUT[pk].reshape(shape) @ x) * g32, 15),
                    "mv_ldexp": tstat(lambda: np.ldexp(LUT[pk].reshape(shape) @ x, ei), 15),
                }
                ymul = (LUT[pk].reshape(shape) @ x) * g32; yld = np.ldexp(LUT[pk].reshape(shape) @ x, ei)
                by = {"codes": n // 4, "f32": n // 4 + 4, "fp16": n // 4 + 2, "e8": n // 4 + 1,
                      "blk_f32": n // 4 + 4 * (n // B), "blk_fp16": n // 4 + 2 * (n // B), "blk_e8": n // 4 + (n // B),
                      "fp16_weights": 2 * n}
                r = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "frac0": fr0,
                     "err": err, "speed": sp, "bytes": by,
                     "mv_rel_mul_vs_ldexp_pow2scale": float(np.linalg.norm(ymul / g32 * np.float32(2.0 ** e) - yld) / np.linalg.norm(yld))}
                out.append(r)
                print(json.dumps({"A": r["shape"], "kind": kind, "seed": seed,
                                  "worst_keep": max(z["pow2_keep"]["sse"] / z["f32"]["sse"] for z in err)}), file=sys.stderr, flush=True)
    return out

# ---------------- Part B: Winsor step-5 gate with pow2-snapped gammas
def load_winsor():
    spec = importlib.util.spec_from_file_location("wsg", WINSOR)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def snap(v):
    return 2.0 ** round(math.log2(v)) if v > 0 else v

def run_gate(m, sizes, pow2):
    orig = m.gammas
    if pow2:
        def g2(d):
            g = orig(d)
            return {k: (snap(v) if k != "tau" else v) for k, v in g.items()}
        m.gammas = g2
    try:
        return m.gate(sizes)
    finally:
        m.gammas = orig

def summary(res):
    paths = [r for r in res["rows"] if r["kind"] == "path"]
    var = {r["variant"]: r for r in res["rows"] if r["kind"] == "variant"}
    return {"verdict": res["verdict"], "path_trits": [r["trit"] for r in paths],
            "q": [[r["q"][v] for v in ("absmean", "drop", "clip")] for r in paths],
            "rel": {k: v["rel"] for k, v in var.items()}, "flips": {k: v["flips_rate"] for k, v in var.items()},
            "gamma": {k: v["gamma"] for k, v in var.items()}}

def part_b():
    m = load_winsor()
    out = {"real_tree": {}, "scenarios": {}, "synthetic": {}}
    sizes = m.scan(Path(TREE))
    out["real_tree"] = {"sizes": sizes, "f32": summary(run_gate(m, sizes, False)), "pow2": summary(run_gate(m, sizes, True))}
    base = dict(m.SEED_BODY)
    sc = {"substantive": base,
          "ws_receipt": {**base, "grok_bot/LAST_RECEIPT.md": 0},
          "stub_state": {**base, "STATE.md": len(b"#TODO")},
          "missing": {k: v for k, v in base.items() if k != "x/f.py"} | {"x/f.py": 0}}
    for k, s in sc.items():
        out["scenarios"][k] = {"f32": summary(run_gate(m, s, False)), "pow2": summary(run_gate(m, s, True))}
    # synthetic: 20,000 trees of 12 paths, log-normal sizes around the real tree, with stubs/hollows injected
    rnd = random.Random(1855)
    logs = [math.log(max(1, v)) for v in sizes.values() if v > 0]
    mu = sum(logs) / len(logs); sd = (sum((l - mu) ** 2 for l in logs) / len(logs)) ** 0.5
    N = 20000; agree_v = agree_p = agree_q = 0; tot_p = tot_q = 0
    cnt = {"f32": {-1: 0, 0: 0, 1: 0}, "pow2": {-1: 0, 0: 0, 1: 0}}
    rel_ratio = []
    tf = tp = 0.0
    for i in range(N):
        s = {f"p{j}": int(math.exp(rnd.gauss(mu, sd))) for j in range(12)}
        u = rnd.random()
        if u < 0.15: s[f"p{rnd.randrange(12)}"] = rnd.randint(1, 12)       # truncated stub
        elif u < 0.20: s[f"p{rnd.randrange(12)}"] = 0                     # hollow
        t0 = time.perf_counter(); a = summary(run_gate(m, s, False)); tf += time.perf_counter() - t0
        t0 = time.perf_counter(); b = summary(run_gate(m, s, True)); tp += time.perf_counter() - t0
        cnt["f32"][a["verdict"]] += 1; cnt["pow2"][b["verdict"]] += 1
        agree_v += a["verdict"] == b["verdict"]
        agree_p += sum(x == y for x, y in zip(a["path_trits"], b["path_trits"])); tot_p += 12
        for qa, qb in zip(a["q"], b["q"]):
            agree_q += sum(x == y for x, y in zip(qa, qb)); tot_q += 3
        if a["rel"]["absmean"] > 0:
            rel_ratio.append(b["rel"]["absmean"] / a["rel"]["absmean"])
    rr = np.array(rel_ratio)
    out["synthetic"] = {"n": N, "mu_ln": mu, "sd_ln": sd, "verdict_agree": agree_v / N, "path_trit_agree": agree_p / tot_p,
                        "q_trit_agree": agree_q / tot_q, "verdict_counts": cnt,
                        "rel_ratio_absmean_med": float(np.median(rr)), "rel_ratio_absmean_p95": float(np.percentile(rr, 95)),
                        "rel_ratio_absmean_max": float(rr.max()),
                        "gate_us_f32": tf / N * 1e6, "gate_us_pow2": tp / N * 1e6}
    return out

if __name__ == "__main__":
    t0 = time.time()
    res = {"numpy": np.__version__, "partB": part_b()}
    print(json.dumps({"B_done": res["partB"]["synthetic"]["verdict_agree"]}), file=sys.stderr, flush=True)
    res["partA"] = part_a()
    res["wall_s"] = time.time() - t0
    print(json.dumps(res))
```

`supp_exact_opt.py` (imports the file above):

```python
#!/usr/bin/env python3
"""Supplement: exact per-tensor f32 ternary optimum (sorted-prefix rule) on the same tensors,
and a check that floor/ceil exponents are the best pow2 choice (e-1 .. e+2 scanned)."""
import json, math
import numpy as np
from pow2_gamma_bench import SHAPES, KINDS, SEEDS, FRACS, make, absmean_codes, sse_rel
out = []
for shape in SHAPES:
    for kind in KINDS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed); w0 = make(kind, shape, rng)
            w64 = w0.astype(np.float64); E = float((w64 * w64).sum())
            s = np.sort(np.abs(w64).ravel())[::-1]; P = np.cumsum(s); k = np.arange(1, s.size + 1)
            sc = P * P / k; kb = int(np.argmax(sc)); opt = (E - sc[kb]) / E; a_opt = P[kb] / (kb + 1)
            g0 = float(np.abs(w0).mean(dtype=np.float64)); fr0 = math.log2(g0) - math.floor(math.log2(g0))
            rows = []
            for fr in FRACS:
                w = (w0 * np.float32(2.0 ** (fr - fr0))).astype(np.float32); wd = w.astype(np.float64)
                Ew = float((wd * wd).sum()); g = float(np.abs(w).mean(dtype=np.float64)); f = math.floor(math.log2(g))
                sse = {e - f: sse_rel(wd, Ew, absmean_codes(w, 2.0 ** e), 2.0 ** e) for e in range(f - 1, f + 3)}
                best = min(sse, key=sse.get)
                rows.append({"frac": fr, "best_off": best, "pow2_best": sse[best], "ratio_vs_opt": sse[best] / opt})
            out.append({"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "f32_opt": opt,
                        "opt_scale_over_absmean": a_opt / g0, "rows": rows})
            print(out[-1]["shape"], kind, seed, round(opt, 4), round(a_opt / g0, 3),
                  [(r["frac"], r["best_off"], round(r["ratio_vs_opt"], 3)) for r in rows], flush=True)
json.dump(out, open("supp_exact_opt.json", "w"))
```

## Next question
Step 5's thin/substantive split moved on 12.9% of synthetic trees under a gamma change of at most 2^0.5. How often does it flip under the much smaller gamma x (1 +/- RATE) wobble the gate already budgets for? Would a hysteresis trit (thin only if certified at gamma(1 + RATE), substantive only if clear at gamma(1 - RATE), 0 = undecided in between) cut verdict flips under RATE on the same 20,000 trees without adding bytes to the I2_S pack?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-19_1455_block128-opt-scale-vs-per-tensor-opt.md.


close: 272e84c88549468e
