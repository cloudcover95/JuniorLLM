---
local: true
date: 2026-10-20
wave: "2026-10-20 02:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH; I2_S 2B4T only if GGUF already on disk (native f32 vs stdlib f64 quantize path on overlay hosts)"
chief_assignment: "pair_A JuniorOS overlay remaining gaps (from Trit Pack 86b112a and Winsor 61872f3 commit text, slot 02:05); pair_B Live beta step 5 (from Handshake c93a5ec)"
verdict: not-viable
tags: [ternary, osai, findings, junioros-overlay, absmean, winsor, f32, f64, gamma, i2s, numpy]
---

# f32 vs f64 ternary quantize path (absmean / Winsor gamma, ternarize, I2_S pack): speed, peak memory and trit parity

Previous note: [[2026-10-19_1855_pow2-e8-gamma-vs-f32-fp16-gamma]]

## Question
Winsor's 02:20 file (JuniorStock 61872f3) shows that an overlay host with `rails/linux/absmean.c` and `winsor.c` compiled quantizes in f32 (4 SIMD f32 lanes for absmean, a scalar f32 running sum for the Winsor clip gamma, `roundf` half-away), while a host without the `.so` falls back to the stdlib engines (f64 sum, `round()` half-even). Is doing the whole quantize step in f32 (gamma, x/gamma, round, clip, I2_S pack) at least 1.5x faster or 1.5x smaller in peak memory than the f64 arithmetic, with trit flips and SSE change within RATE 0.05? And which f32 accumulation keeps the I2_S pack, and so the close key, identical to the stdlib path?

Viability bar:
- End-to-end quantize time or peak memory at least 1.5x better.
- Trit flips at most RATE (5%) and SSE within 5% of the f64 path.

## Method
- python3 3.13.5 + numpy 2.5.4 only, offline, no sockets, no installs, 8 cores, `OMP_NUM_THREADS=8`. Run on the box 2026-10-20 03:07 to 03:13 MT (main) and about 03:15 MT (supplement).
- Synthetic float32 weights at 2B4T shapes 2560 x 2560 and 2560 x 6912 (the FFN shape, n = 17,694,720). Kinds `gauss` (sigma 0.02) and `t4` (Student-t df 4, x 0.02). Seeds 255 and 1255.
- Paths, each gamma + ternarize + I2_S pack (0->0, 1->1, -1->2, four per byte, MSB first, the mapping in Winsor's close rule), eps 1e-7 as in both engines:
  - `f64 stdlib arithmetic`: gamma = f64 mean |w|, q = clip(rint(w64 / gamma)) (half-even, like `round()`).
  - `f32 pairwise + rint`: gamma = numpy f32 pairwise mean, x = w / gamma in f32, rint, clip. Not a C engine; the cheapest accurate f32 path.
  - `absmean.c emulation`: 4 sequential f32 lanes (`reshape(-1, 4).sum(axis=0, dtype=f32)`, which numpy reduces row by row), lanes added in float, divided by n in double, cast to float; x / gamma in f32; exact half-away rounding (`roundf`).
  - `winsor.c-style scalar running sum`: f32 `cumsum` (sequential) of |w|, divided by (float)n; half-away rounding. Applied to both the absmean gamma and the Winsor clip gamma mean(min(|w|, P95)).
- Speed: 15 timed repeats after one warm-up; median, min-max and IQR. Peak memory: `tracemalloc` peak of one call (numpy reports its buffers to it). Error: trit flips against the f64 path, SSE / ||W||^2 against the f64 SSE, I2_S pack sha256[:16].
- Supplement: the shared pack rebuilt to make uint8 codes directly (no int64 `np.where` temp), end-to-end f64 vs f32 pairwise re-timed, and the signed gamma error of the two C emulations.
- Context: the pure-Python stdlib arithmetic (`sum(abs)`, `round`) timed on a 65,536-weight slice, 5 repeats.

Commands (run from `/workspace/tbench/r20261020_0255` on the box):

```
OMP_NUM_THREADS=8 python3 f32_gamma_path_bench.py > results.json 2> progress.log
OMP_NUM_THREADS=8 python3 supp_u8pack.py
```

## Results

### A. End to end: gamma + ternarize + I2_S pack (ms, median (min-max) [IQR], 15 repeats)

| shape | kind | seed | f64 stdlib arithmetic | f32 pairwise + rint | speedup | u8-pack rerun f64 / f32 | rerun speedup | peak f64 / f32 MB | memory ratio |
|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 255 | 74.0 (61.0-100.6) [69.8-78.5] | 44.8 (41.4-48.8) [43.1-48.1] | 1.65x | 84.1 / 57.7 | 1.46x | 104.9 / 85.2 | 1.23x |
| 2560x2560 | gauss | 1255 | 75.7 (65.7-84.1) [71.1-79.5] | 51.3 (44.8-56.3) [48.1-53.6] | 1.48x | 62.5 / 43.4 | 1.44x | 104.9 / 85.2 | 1.23x |
| 2560x2560 | t4 | 255 | 72.0 (66.5-76.4) [68.6-74.5] | 46.1 (42.1-50.9) [44.9-47.9] | 1.56x | 70.0 / 45.8 | 1.53x | 104.9 / 85.2 | 1.23x |
| 2560x2560 | t4 | 1255 | 67.4 (64.6-72.9) [66.2-68.7] | 46.9 (43.6-50.9) [46.0-47.7] | 1.44x | 64.0 / 38.8 | 1.65x | 104.9 / 85.2 | 1.23x |
| 2560x6912 | gauss | 255 | 232.2 (227.2-277.0) [228.9-233.4] | 157.6 (154.0-177.1) [155.0-167.4] | 1.47x | 218.3 / 155.4 | 1.40x | 283.1 / 230.0 | 1.23x |
| 2560x6912 | gauss | 1255 | 226.1 (218.9-270.4) [224.1-230.4] | 159.5 (154.6-172.2) [158.4-160.5] | 1.42x | 229.9 / 190.1 | 1.21x | 283.1 / 230.0 | 1.23x |
| 2560x6912 | t4 | 255 | 225.1 (212.7-248.4) [221.8-228.1] | 154.3 (147.9-161.3) [150.9-156.4] | 1.46x | 226.3 / 161.3 | 1.40x | 283.1 / 230.0 | 1.23x |
| 2560x6912 | t4 | 1255 | 229.8 (215.3-296.7) [223.3-238.5] | 159.6 (149.8-175.3) [157.8-161.1] | 1.44x | 223.1 / 155.2 | 1.44x | 283.1 / 230.0 | 1.23x |

### B. Steps on their own (ms, median (min-max))

| shape | kind | seed | gamma f64 | gamma f32 pairwise | gamma ratio | ternarize f64 | ternarize f32 rint | ternarize ratio | ternarize f32 half-away (numpy emulation) |
|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 255 | 4.2 (3.9-7.0) | 2.9 (2.6-3.1) | 1.43x | 35.0 (28.1-111.3) | 10.0 (9.1-14.7) | 3.50x | 46.0 (36.7-50.0) |
| 2560x2560 | gauss | 1255 | 4.2 (4.0-6.8) | 3.0 (2.7-3.1) | 1.42x | 27.1 (24.6-33.2) | 11.3 (10.5-15.7) | 2.40x | 45.6 (37.0-50.9) |
| 2560x2560 | t4 | 255 | 4.1 (3.9-6.4) | 2.9 (2.8-3.1) | 1.42x | 29.0 (26.5-60.4) | 8.7 (8.1-13.0) | 3.33x | 32.7 (30.0-45.5) |
| 2560x2560 | t4 | 1255 | 5.0 (4.2-6.5) | 2.8 (2.7-3.0) | 1.80x | 24.7 (22.6-32.4) | 9.8 (9.5-14.5) | 2.52x | 35.6 (33.0-39.7) |
| 2560x6912 | gauss | 255 | 17.7 (15.5-25.4) | 12.2 (11.7-13.0) | 1.45x | 120.2 (110.6-162.7) | 56.1 (51.7-60.2) | 2.14x | 183.0 (173.5-215.7) |
| 2560x6912 | gauss | 1255 | 14.3 (13.5-24.4) | 10.4 (10.0-12.6) | 1.38x | 105.0 (87.3-147.9) | 41.1 (37.7-55.2) | 2.55x | 166.9 (158.9-174.7) |
| 2560x6912 | t4 | 255 | 16.2 (14.5-26.4) | 10.7 (10.0-11.0) | 1.51x | 108.0 (94.6-121.9) | 45.5 (42.6-50.4) | 2.38x | 170.4 (167.5-180.3) |
| 2560x6912 | t4 | 1255 | 16.5 (14.2-22.1) | 11.4 (10.6-12.3) | 1.45x | 114.9 (100.5-126.8) | 50.1 (43.8-75.0) | 2.29x | 172.1 (164.7-235.8) |

### C. Error and trit agreement against the f64 stdlib arithmetic

| shape | kind | seed | path | gamma rel err (signed for C emulations) | trit flips | flip rate | SSE / f64 SSE | I2_S pack sha16 |
|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 255 | f64 stdlib arithmetic | 1.1e-15 | 0 | 0.00e+00 | 1.000000 | `f6654efdaf4b7390` |
| 2560x2560 | gauss | 255 | f32 pairwise + rint | 2.0e-08 | 0 | 0.00e+00 | 1.000000 | `f6654efdaf4b7390` |
| 2560x2560 | gauss | 255 | absmean.c emulation (4 f32 lanes, roundf) | -1.83e-04 | 367 | 5.60e-05 | 1.000206 | `8e8d444951bd8add` |
| 2560x2560 | gauss | 255 | winsor.c-style scalar f32 running sum, roundf | -3.03e-03 | 5765 | 8.80e-04 | 1.003433 | `395ee54764d99824` |
| 2560x2560 | gauss | 1255 | f64 stdlib arithmetic | 0.0e+00 | 0 | 0.00e+00 | 1.000000 | `eccbe8299b58aa36` |
| 2560x2560 | gauss | 1255 | f32 pairwise + rint | 1.2e-08 | 0 | 0.00e+00 | 1.000000 | `eccbe8299b58aa36` |
| 2560x2560 | gauss | 1255 | absmean.c emulation (4 f32 lanes, roundf) | -1.88e-04 | 400 | 6.10e-05 | 1.000213 | `734ee3a1d9bee2c2` |
| 2560x2560 | gauss | 1255 | winsor.c-style scalar f32 running sum, roundf | -3.00e-03 | 5789 | 8.83e-04 | 1.003396 | `7c74d3605603d6d6` |
| 2560x2560 | t4 | 255 | f64 stdlib arithmetic | 1.7e-16 | 0 | 0.00e+00 | 1.000000 | `18a0b78d882917d0` |
| 2560x2560 | t4 | 255 | f32 pairwise + rint | 3.4e-08 | 0 | 0.00e+00 | 1.000000 | `18a0b78d882917d0` |
| 2560x2560 | t4 | 255 | absmean.c emulation (4 f32 lanes, roundf) | -1.73e-04 | 370 | 5.65e-05 | 1.000114 | `bbef78100959fc3e` |
| 2560x2560 | t4 | 255 | winsor.c-style scalar f32 running sum, roundf | -2.71e-03 | 5846 | 8.92e-04 | 1.001792 | `82d4eab31e65613a` |
| 2560x2560 | t4 | 1255 | f64 stdlib arithmetic | 3.5e-16 | 0 | 0.00e+00 | 1.000000 | `560d33e7e89f3c58` |
| 2560x2560 | t4 | 1255 | f32 pairwise + rint | 3.1e-08 | 0 | 0.00e+00 | 1.000000 | `560d33e7e89f3c58` |
| 2560x2560 | t4 | 1255 | absmean.c emulation (4 f32 lanes, roundf) | -1.76e-04 | 342 | 5.22e-05 | 1.000116 | `d63090cbb4af9c81` |
| 2560x2560 | t4 | 1255 | winsor.c-style scalar f32 running sum, roundf | -2.80e-03 | 5733 | 8.75e-04 | 1.001845 | `6bbc2bf5734a60c5` |
| 2560x6912 | gauss | 255 | f64 stdlib arithmetic | 4.3e-16 | 0 | 0.00e+00 | 1.000000 | `6143baefeb47b423` |
| 2560x6912 | gauss | 255 | f32 pairwise + rint | 6.0e-08 | 0 | 0.00e+00 | 1.000000 | `6143baefeb47b423` |
| 2560x6912 | gauss | 255 | absmean.c emulation (4 f32 lanes, roundf) | -1.29e-03 | 6671 | 3.77e-04 | 1.001462 | `45b0704c423da42b` |
| 2560x6912 | gauss | 255 | winsor.c-style scalar f32 running sum, roundf | -2.03e-02 | 106432 | 6.01e-03 | 1.023528 | `070e76aae8990d06` |
| 2560x6912 | gauss | 1255 | f64 stdlib arithmetic | 6.5e-16 | 0 | 0.00e+00 | 1.000000 | `d95f7caadbd509e9` |
| 2560x6912 | gauss | 1255 | f32 pairwise + rint | 2.5e-08 | 0 | 0.00e+00 | 1.000000 | `d95f7caadbd509e9` |
| 2560x6912 | gauss | 1255 | absmean.c emulation (4 f32 lanes, roundf) | -1.30e-03 | 6725 | 3.80e-04 | 1.001473 | `05b387739a9ca02e` |
| 2560x6912 | gauss | 1255 | winsor.c-style scalar f32 running sum, roundf | -2.05e-02 | 106416 | 6.01e-03 | 1.023671 | `720d5a25aad429d2` |
| 2560x6912 | t4 | 255 | f64 stdlib arithmetic | 1.4e-15 | 0 | 0.00e+00 | 1.000000 | `06fbbc08081c0c58` |
| 2560x6912 | t4 | 255 | f32 pairwise + rint | 1.6e-08 | 1 | 5.65e-08 | 1.000000 | `653d9099221533e2` |
| 2560x6912 | t4 | 255 | absmean.c emulation (4 f32 lanes, roundf) | -1.79e-03 | 10285 | 5.81e-04 | 1.001180 | `c270e626b67a2c51` |
| 2560x6912 | t4 | 255 | winsor.c-style scalar f32 running sum, roundf | -3.00e-02 | 171742 | 9.71e-03 | 1.020301 | `83afddde43f56397` |
| 2560x6912 | t4 | 1255 | f64 stdlib arithmetic | 1.0e-15 | 0 | 0.00e+00 | 1.000000 | `2feaf79af08ab56c` |
| 2560x6912 | t4 | 1255 | f32 pairwise + rint | 3.7e-08 | 1 | 5.65e-08 | 1.000000 | `645bea375023e794` |
| 2560x6912 | t4 | 1255 | absmean.c emulation (4 f32 lanes, roundf) | -1.76e-03 | 10062 | 5.69e-04 | 1.001164 | `0ae9b06c8dc74509` |
| 2560x6912 | t4 | 1255 | winsor.c-style scalar f32 running sum, roundf | -2.98e-02 | 170949 | 9.66e-03 | 1.020217 | `73609f7d87077585` |

### D. Winsor clip gamma (mean of min(|w|, P95)), f32 accumulation vs f64

| shape | kind | seed | P95 | rel err f32 pairwise / scalar running sum | flips f32 pairwise / scalar running sum | SSE ratio scalar running sum | ms f64 / f32 pairwise / scalar |
|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 255 | 0.0392 | 2.7e-08 / 2.95e-03 | 0 / 5647 | 1.0034 | 9.5 / 7.8 / 22.4 |
| 2560x2560 | gauss | 1255 | 0.0392 | 1.3e-08 / 3.02e-03 | 0 / 5736 | 1.0034 | 8.9 / 7.6 / 24.2 |
| 2560x2560 | t4 | 255 | 0.0555 | 3.0e-09 / 4.48e-03 | 0 / 9162 | 1.0030 | 7.0 / 4.6 / 18.7 |
| 2560x2560 | t4 | 1255 | 0.0555 | 3.5e-08 / 4.54e-03 | 0 / 9160 | 1.0030 | 9.0 / 6.5 / 21.9 |
| 2560x6912 | gauss | 255 | 0.0392 | 1.7e-08 / 1.15e-02 | 0 / 58814 | 1.0132 | 36.8 / 31.9 / 78.8 |
| 2560x6912 | gauss | 1255 | 0.0392 | 2.3e-08 / 1.16e-02 | 1 / 59073 | 1.0133 | 37.6 / 30.6 / 77.4 |
| 2560x6912 | t4 | 255 | 0.0555 | 1.3e-08 / 1.86e-02 | 1 / 101111 | 1.0125 | 42.5 / 38.3 / 84.7 |
| 2560x6912 | t4 | 1255 | 0.0555 | 1.4e-08 / 1.85e-02 | 1 / 100600 | 1.0124 | 33.7 / 36.4 / 71.1 |

Exact half-way ties in f32 w/gamma per tensor: 1, 1, 1, 0, 1, 2, 2, 2. Pure-Python stdlib engine arithmetic (65,536-weight slice, 5 repeats): 105, 105, 104, 109, 110, 118, 108, 104 ns per weight.

## Verdict
**Not viable.** Doing the quantize step in f32 is a real but sub-1.5x gain:
- **End to end** (gamma + ternarize + I2_S pack), f32 pairwise is 1.42x to 1.65x faster, median 1.47x over 8 tensors; the uint8-pack rerun gives 1.21x to 1.65x, median 1.44x. Only 4 of 16 timings clear 1.5x, all at 2560 x 2560; none at the FFN shape.
- **Peak memory** is 1.23x lower (85.2 vs 104.9 MB, 230.0 vs 283.1 MB), because both paths still hold f32 or f64 temporaries the size of the tensor.
- **The one step that clears the bar:** ternarize alone (x / gamma, rint, clip to int8) is 2.14x to 3.50x faster in f32. But the I2_S pack, identical in both paths, is now the largest share of the f32 path (about 32 of 45 ms at 2560 x 2560), so the end-to-end gain stays under 1.5x.
- **Error:** f32 pairwise accumulation is free. Gamma is within 6e-8 of f64, the I2_S pack is byte-identical on 6 of 8 tensors and 1 trit off (5.7e-8) on the two t4 FFN tensors, and SSE is unchanged to 6 digits.

What it does show for the overlay item (the parity side Winsor certifies):
- The **absmean.c 4-lane f32 arithmetic** puts gamma 1.7e-4 to 1.8e-3 low, flips 5.2e-5 to 5.8e-4 of trits and raises SSE by at most 0.15%. That is well inside RATE, but the I2_S pack, and so any close key over it, differs on every tensor between hosts with and without the `.so`.
- The **scalar f32 running sum** (winsor.c style) is the real hazard. At the FFN shape its gamma is 2.0% to 3.0% low for absmean and 1.2% to 1.9% low for the Winsor clip gamma. It flips 0.60% to 0.97% of trits and raises SSE by 2.0% to 2.4%. That is consistent with the f32 stall Winsor's file predicts for n > 2^24, inside RATE today but growing with n.
- `roundf` half-away vs `rint` half-even changed at most 1 trit per tensor (0 to 2 exact ties per tensor).
- A pairwise (or blocked) f32 sum gives the C speed path the stdlib's trits. A native engine that wants the same pack as the fallback should use one instead of a running sum.

Caveats:
- Synthetic weights only, with no GGUF on disk.
- numpy, not the C kernels: the C emulations are timed only as numpy code (slow), so their speed columns say nothing about `absmean.c`.
- `tracemalloc` peak covers numpy buffers only.

## This wave's pair files
- Pair A (JuniorOS overlay, remaining gaps):
  - Trit Pack BitNet-mlx 86b112a [`scripts/bitnetd_second_tier_hardening_i2s_key.py`](https://github.com/cloudcover95/BitNet-mlx/blob/86b112a/scripts/bitnetd_second_tier_hardening_i2s_key.py): second-tier systemd.exec sandboxing and surface-narrowing trits for bitnetd.service, JTR1 + I2_S 128-block key.
  - Winsor JuniorStock 61872f3 [`quant/calibration/overlay_native_f32_gamma_parity.py`](https://github.com/cloudcover95/JuniorStock/blob/61872f3/quant/calibration/overlay_native_f32_gamma_parity.py): certified native-f32 vs stdlib gamma parity (SIMD lanes, winsor.c f32 sum, roundf vs half-even, f32 stall cap). This note measures the speed side and the observed flips of the same two backends.
- Pair B (Live beta step 5):
  - Handshake JuniorLLM c93a5ec [`ports/live_beta_backfill_fresh_clone_index_handshake.py`](https://github.com/cloudcover95/JuniorLLM/blob/c93a5ec/ports/live_beta_backfill_fresh_clone_index_handshake.py).
  - No Side-Commit 02:35 push was found locally.

## Next question
The I2_S pack is now the largest cost in the f32 quantize path (about 32 ms of 45 ms at 2560 x 2560). Is a pack built as one uint8 matmul against [64, 16, 4, 1] (or a 3^4 -> byte LUT on base-3 indices) at least 1.5x faster than the current shift-or chain over a (n/4, 4) view, with byte-identical output?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-19_1855_pow2-e8-gamma-vs-f32-fp16-gamma.md.

## Inline script
`f32_gamma_path_bench.py`:

```python
#!/usr/bin/env python3
"""f32 vs f64 absmean / Winsor-clip ternary quantize path (JuniorOS overlay native-vs-stdlib).
Question: is quantizing in f32 end to end (gamma accumulation + x/g + round + I2_S pack, the
arithmetic of rails/linux absmean.c and winsor.c) at least 1.5x faster or smaller in peak memory
than the f64 path (the arithmetic of the stdlib engines junior_bitnet.math.absmean and
junior_bitnet.winsor.pack), with trit flips and SSE change within RATE 0.05?
numpy only, offline, no sockets, no installs. Prints one JSON blob."""
import hashlib, json, sys, time, tracemalloc
import numpy as np

SHAPES = [(2560, 2560), (2560, 6912)]
KINDS = ["gauss", "t4"]
SEEDS = [255, 1255]
REP = 15
RATE = 0.05
EPS = 1e-7

def tstat(f, n=REP):
    f()
    ts = []
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

def pack2(q):  # I2_S MSB-first, 0->0, 1->1, -1->2 (Winsor close-rule mapping)
    c = np.where(q < 0, 2, q).astype(np.uint8).reshape(-1, 4)
    return (c[:, 0] << 6 | c[:, 1] << 4 | c[:, 2] << 2 | c[:, 3]).astype(np.uint8)

def half_away(x):  # roundf semantics, exact on f32 (no x+0.5 double rounding)
    t = np.trunc(x); r = np.rint(x)
    tie = np.abs(x - t) == x.dtype.type(0.5)
    return np.where(tie, t + np.sign(x), r)

# ---- gammas
def g_f64(w):             # stdlib engines: f64 sum(|w|)/n
    return float(np.abs(w).mean(dtype=np.float64))
def g_f32_pairwise(w):    # numpy f32 pairwise (not a C engine; cheapest accurate f32)
    return np.float32(np.abs(w).mean(dtype=np.float32))
def g_f32_lanes4(w):      # absmean.c SSE2/NEON: 4 sequential f32 lanes, lanes summed in float, /n in double
    lanes = np.abs(w).reshape(-1, 4).sum(axis=0, dtype=np.float32)  # axis-0 reduce is sequential
    s = np.float32(np.float32(lanes[0] + lanes[1]) + np.float32(lanes[2] + lanes[3]))
    return np.float32(float(s) / w.size)
def g_f32_seq(w):         # winsor.c-style scalar running f32 sum, acc/(float)n
    return np.float32(np.cumsum(np.abs(w).ravel(), dtype=np.float32)[-1] / np.float32(w.size))

def p95(w):
    a = np.abs(w).ravel(); k = int(0.95 * (a.size - 1))
    return float(np.partition(a, k)[k])
def gw_f64(w, P):
    return float(np.minimum(np.abs(w), np.float32(P)).mean(dtype=np.float64))
def gw_f32_seq(w, P):
    return np.float32(np.cumsum(np.minimum(np.abs(w).ravel(), np.float32(P)), dtype=np.float32)[-1] / np.float32(w.size))
def gw_f32_pairwise(w, P):
    return np.float32(np.minimum(np.abs(w), np.float32(P)).mean(dtype=np.float32))

# ---- full quantize paths: gamma + ternarize + pack
def q64(w, g):
    return np.clip(np.rint(w.astype(np.float64) / (g + EPS)), -1, 1).astype(np.int8)
def q32(w, g, away=False):
    x = w / np.float32(g + np.float32(EPS))
    r = half_away(x) if away else np.rint(x)
    return np.clip(r, -1, 1).astype(np.int8)

def path_f64(w):        g = g_f64(w); q = q64(w, g); return g, q, pack2(q)
def path_f32_pw(w):     g = g_f32_pairwise(w); q = q32(w, g); return g, q, pack2(q)
def path_f32_lanes(w):  g = g_f32_lanes4(w); q = q32(w, g, away=True); return g, q, pack2(q)
def path_f32_lanes_rint(w): g = g_f32_lanes4(w); q = q32(w, g); return g, q, pack2(q)
def path_f32_seq(w):    g = g_f32_seq(w); q = q32(w, g, away=True); return g, q, pack2(q)

PATHS = {"f64_stdlib_arith": path_f64, "f32_pairwise_rint": path_f32_pw,
         "f32_lanes4_rint": path_f32_lanes_rint, "f32_lanes4_away (absmean.c)": path_f32_lanes,
         "f32_seq_away (scalar running sum)": path_f32_seq}

def sse_rel(w64, E, q, g):
    d = w64 - q.astype(np.float64) * float(g); return float((d * d).sum() / E)

def stdlib_rate(w, m=65536):  # pure-Python engine arithmetic, per weight, on a slice
    xs = [float(v) for v in w.ravel()[:m]]
    def run():
        g = sum(abs(v) for v in xs) / len(xs)
        return [max(-1, min(1, round(v / (g + EPS)))) for v in xs]
    t = tstat(run, 5); return t["med"] / m * 1e6  # ns per weight

def main():
    out = []
    for shape in SHAPES:
        for kind in KINDS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                w = make(kind, shape, rng); n = w.size
                w64 = w.astype(np.float64); E = float((w64 * w64).sum())
                exact = float(np.abs(w64).sum()) / n  # f64 pairwise reference
                gref, qref, pref = path_f64(w)
                row = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "n": n,
                       "fp32_bytes": 4 * n, "pack_bytes": n // 4, "paths": {}}
                for name, fn in PATHS.items():
                    g, q, p = fn(w)
                    flips = int((q != qref).sum())
                    row["paths"][name] = {
                        "gamma": float(g), "gamma_rel_err": abs(float(g) - exact) / exact,
                        "flips": flips, "flip_rate": flips / n,
                        "sse": sse_rel(w64, E, q, g), "sse_ratio": sse_rel(w64, E, q, g) / sse_rel(w64, E, qref, gref),
                        "pack_equal": bool(np.array_equal(p, pref)),
                        "pack_sha16": hashlib.sha256(p.tobytes()).hexdigest()[:16],
                        "time_ms": tstat(lambda: fn(w)), "peak_bytes": peak(lambda: fn(w)),
                    }
                # gamma-only timings
                row["gamma_ms"] = {k: tstat(f) for k, f in {
                    "f64": lambda: g_f64(w), "f32_pairwise": lambda: g_f32_pairwise(w),
                    "f32_lanes4": lambda: g_f32_lanes4(w), "f32_seq": lambda: g_f32_seq(w)}.items()}
                # ternarize-only (given gamma)
                g32 = g_f32_pairwise(w)
                row["tern_ms"] = {"f64": tstat(lambda: q64(w, float(g32))), "f32_rint": tstat(lambda: q32(w, g32)),
                                  "f32_away": tstat(lambda: q32(w, g32, True))}
                row["tern_peak"] = {"f64": peak(lambda: q64(w, float(g32))), "f32_rint": peak(lambda: q32(w, g32))}
                # Winsor clip gamma at P95: accumulation bias and flips
                P = p95(w); ex_w = float(np.minimum(np.abs(w64), P).sum()) / n
                gw = {"f64": gw_f64(w, P), "f32_pairwise": float(gw_f32_pairwise(w, P)), "f32_seq": float(gw_f32_seq(w, P))}
                qw_ref = q64(w, gw["f64"])
                row["winsor"] = {"P95": P, "exact": ex_w,
                                 "gamma_rel_err": {k: abs(v - ex_w) / ex_w for k, v in gw.items()},
                                 "flips_vs_f64": {k: int((q32(w, np.float32(v), k == "f32_seq") != qw_ref).sum()) for k, v in gw.items() if k != "f64"},
                                 "sse_ratio_vs_f64": {k: sse_rel(w64, E, q32(w, np.float32(v), k == "f32_seq"), v) / sse_rel(w64, E, qw_ref, gw["f64"]) for k, v in gw.items() if k != "f64"},
                                 "ms": {"f64": tstat(lambda: gw_f64(w, P), 7), "f32_seq": tstat(lambda: gw_f32_seq(w, P), 7),
                                        "f32_pairwise": tstat(lambda: gw_f32_pairwise(w, P), 7)}}
                row["exact_ties_f32"] = int((np.abs(w / g32 - np.trunc(w / g32)) == np.float32(0.5)).sum())
                row["stdlib_python_ns_per_weight"] = stdlib_rate(w)
                out.append(row)
                print(json.dumps({"done": row["shape"], "kind": kind, "seed": seed,
                                  "t": {k: round(v["time_ms"]["med"], 2) for k, v in row["paths"].items()},
                                  "flips": {k: v["flips"] for k, v in row["paths"].items()}}), file=sys.stderr, flush=True)
    print(json.dumps({"numpy": np.__version__, "python": sys.version.split()[0], "rows": out}))

if __name__ == "__main__":
    main()
```

`supp_u8pack.py` (imports the file above):

```python
#!/usr/bin/env python3
"""Supplement: same f64 vs f32-pairwise quantize paths, but the shared I2_S pack builds uint8 codes
(no int64 np.where temp), so the pack cost common to both paths is smaller. Also the sign of the
f32 lanes4 / scalar-running-sum gamma error. Imports the main bench."""
import json, numpy as np
from f32_gamma_path_bench import (SHAPES, KINDS, SEEDS, make, g_f64, g_f32_pairwise, g_f32_lanes4,
                                  g_f32_seq, q64, q32, tstat, peak)
def pack2u(q):
    c = np.where(q < 0, np.uint8(2), q.view(np.uint8)).reshape(-1, 4)
    return (c[:, 0] << 6 | c[:, 1] << 4 | c[:, 2] << 2 | c[:, 3])
def p64(w): g = g_f64(w); q = q64(w, g); return pack2u(q)
def p32(w): g = g_f32_pairwise(w); q = q32(w, g); return pack2u(q)
out = []
for shape in SHAPES:
    for kind in KINDS:
        for seed in SEEDS:
            w = make(kind, shape, np.random.default_rng(seed)); n = w.size
            ex = float(np.abs(w.astype(np.float64)).sum()) / n
            a, b = p64(w), p32(w); assert a.dtype == np.uint8
            r = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed,
                 "t64": tstat(lambda: p64(w)), "t32": tstat(lambda: p32(w)),
                 "peak64": peak(lambda: p64(w)), "peak32": peak(lambda: p32(w)),
                 "pack_bytes_diff": int((a != b).sum()),
                 "lanes4_signed_rel": (float(g_f32_lanes4(w)) - ex) / ex, "seq_signed_rel": (float(g_f32_seq(w)) - ex) / ex}
            r["speedup"] = r["t64"]["med"] / r["t32"]["med"]; r["mem_ratio"] = r["peak64"] / r["peak32"]
            out.append(r)
            print(r["shape"], kind, seed, "f64 %.2f (%.2f-%.2f)[%.2f-%.2f]" % tuple(r["t64"][k] for k in ("med","min","max","q1","q3")),
                  "f32 %.2f (%.2f-%.2f)[%.2f-%.2f]" % tuple(r["t32"][k] for k in ("med","min","max","q1","q3")),
                  "spd %.2fx peak %.1f/%.1f MB (%.2fx) packdiff %d lanes4 %+.2e seq %+.2e" % (r["speedup"], r["peak64"]/1e6, r["peak32"]/1e6, r["mem_ratio"], r["pack_bytes_diff"], r["lanes4_signed_rel"], r["seq_signed_rel"]), flush=True)
s = np.array([r["speedup"] for r in out]); print("speedup median %.2f min %.2f max %.2f" % (np.median(s), s.min(), s.max()))
json.dump(out, open("supp_u8pack.json", "w"))
```

close: 39a6003c306869c1
