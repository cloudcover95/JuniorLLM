---
local: true
date: 2026-10-17T14:55:00-06:00
wave: "2026-10-17 14:05"
roadmap_item: "JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH. I2_S 2B4T only if GGUF already on disk."
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md ## JuniorOS overlay (pair_A item this wave, per Trit Pack 7d40533 and Winsor e82b55b)"
verdict: viable
tags: [osai, vault, findings, ternary, i2s, junioros-overlay, reproducibility, determinism, int8, winsor, benchmark]
---

# Bit-reproducible matvec: exact int8 x trit accumulation vs deterministic float

- Prev: [[2026-10-17_1055_ue5-watts-gate-trit-vs-float-certified]]
- Roadmap: JuniorOS overlay, Debian/Alpine + hardened bitnetd.service (JuniorLLM docs/BETA_TO_OS.md)
- Why this question: Winsor's pair file this wave (e82b55b) bounds how many weight trits can flip between Debian (glibc) and Alpine (musl) when gamma is summed in a different order. The same worry applies one step later, to the matvec outputs bitnetd computes on each base. Ternary weights times int8 activations are small integers, so their sums can be exact. This note measures whether that exact path is also cheaper than the cheapest float path that gives the same bits every time. It also moves the bench off #5 and #3.

## Question
For B=1 and B=8 matvecs at BitNet 2B4T-like shapes, is the exact ternary int path (per-token absmax int8 activations x {-1,0,+1} weights) at least 1.5x faster or smaller than the cheapest float path that is bit-reproducible across summation order, thread count and batch size, at acceptable error against a float64 reference? And how often does an order change actually flip a downstream trit, RoundClip(y / mean|y|)?

## Method
- Weights: T in {-1,0,1} with p = 0.3/0.4/0.3. Shapes (rows out x cols in): 2560x2560, 6912x2560, 2560x6912. Activations: 64 tokens of N(0,1) fp32. A second set of 64 tokens with 0.5% outlier channels (x20 lognormal, as in the 05:55 note) is used for error only. Seed 20261017.
- Int path: s = max|x| / 127, q = clip(rint(x / s), -127, 127), y = (T @ q) * s. Every partial sum is an integer with |sum| <= 127 x 6912 = 877,824 < 2^24, so fp32 holds it exactly in any order. Four ways to run it, all bit-identical:
  - **int_f32blas**: T stored as fp32, q as integer-valued fp32, plain BLAS sgemv/sgemm.
  - **int_i8tile**: T stored as int8. For each 64-row tile, cast to fp32 into a reused buffer, then sgemv.
  - **int_p2tile**: T stored 2-bit packed (I2_S codes 0, 1, 2, four per byte). For each 64-row tile, decode with a 256x4 fp32 LUT, then sgemv.
  - **int_np_int32**: numpy integer matmul (int8 x int8, int32 accumulate, no BLAS).
- Float paths, y = T @ x on float activations:
  - **f32_blas**: fp32 BLAS. This is the normal fast path, with no guarantee.
  - **f32_einsum**: numpy's C einsum. It's single-threaded and fixed-order for a given build and layout.
  - **f64_blas_round**: T as float64, x cast to float64, BLAS dgemv, round to fp32. The fp32 products are exact in float64, so this is very likely order-proof, but not provably.
  - **f64_einsum_round**: the same with einsum.
  - **fsum_round**: math.fsum per output row, then fp32. It's correctly rounded, so it's order-proof on any machine.
- Order tests, on all 64 tokens:
  - 1 vs 8 BLAS threads (separate processes).
  - The same matvec with a shuffled column order (T[:, p] and x[p]). This stands in for a different BLAS blocking or libc build.
  - B=1 calls vs B=8 calls.
  - Cross-implementation: BLAS vs einsum vs fsum.
  For each test I count output elements whose fp32 bits differ, and trit flips after RoundClip(y / mean|y|). fsum uses 8 tokens because it's slow.
- Error: relative L2 per token against float64 T @ x (median and max), and against float64 (T @ q) * s. The second one is the W1.58A8 forward that BitNet b1.58 defines (activations are int8 per token by design).
- Timing: 15 repeats after a warm-up (fsum: 9), median (min-max), at OPENBLAS_NUM_THREADS=1 and 8. Box: 8 cores, numpy 2.5.4, OpenBLAS 0.3.34 (SkylakeX kernel). Peak RSS 795 MB at 1 thread, 816 MB at 8.
- JSON and npz results are in `/workspace/tbench/r1455/d20261017/`. They're box-local and not pushed.

```
cd /workspace/tbench
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_int_vs_det_float_bench.py run threads1
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 repro_int_vs_det_float_bench.py run threads8
python3 repro_int_vs_det_float_bench.py compare
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 repro_int_vs_det_float_8t_recheck.py
```

Core of the script (full file at `/workspace/tbench/repro_int_vs_det_float_bench.py`):

```python
def quant(X):                                   # per-token absmax int8, plain fp32 IEEE ops (no libm)
    s = (np.abs(X).max(axis=0) / np.float32(127)).astype(np.float32)
    return np.clip(np.rint(X / s), -127, 127).astype(np.float32), s
def trits(Y):                                   # RoundClip(y / mean|y|)
    return np.clip(np.rint(Y / np.abs(Y.astype(np.float64)).mean(axis=0)), -1, 1).astype(np.int8)
def int_f32blas(X):  q, s = quant(X); return (Wf @ q) * s            # integer sums < 2^24: exact in fp32
def int_i8tile(X):
    q, s = quant(X); Y = np.empty((R, X.shape[1]), np.float32)
    for r in range(0, R, 64):
        b = buf[:min(64, R - r)]; np.copyto(b, T[r:r + 64]); np.matmul(b, q, out=Y[r:r + 64])
    return Y * s
def int_p2tile(X):                              # 2-bit I2_S codes, 256x4 fp32 LUT decode per 64-row tile
    q, s = quant(X); Y = np.empty((R, X.shape[1]), np.float32)
    for r in range(0, R, 64):
        n = min(64, R - r); b = buf3[:n]
        np.take(LUT, P[r:r + n], axis=0, out=b); np.matmul(b.reshape(n, C), q, out=Y[r:r + n])
    return Y * s
def f64_blas_round(X): return (Wd @ X.astype(np.float64)).astype(np.float32)
def fsum_round(X): ...  # Y[i, b] = math.fsum((Wf[i] * X[:, b]).tolist())
# order tests: same fn on T[:, perm] / x[perm]; B=1 vs B=8; threads 1 vs 8 (separate runs); compare fp32 bits
```

## Results: order sensitivity
Each cell gives outputs whose fp32 bits differ / trit flips, per shape in the order 2560x2560 ; 6912x2560 ; 2560x6912. Outputs compared: 163,840 ; 442,368 ; 163,840 (64 tokens). fsum: 20,480 ; 55,296 ; 20,480 (8 tokens).

| path | 1 vs 8 threads | shuffled column order | B=1 vs B=8 batch |
|---|---|---|---|
| f32_blas | 0 / 0 ; 0 / 0 ; 0 / 0 | 149,282 / 0 ; 403,038 / 0 ; 151,585 / 0 | 150,664 / 0 ; 407,177 / 0 ; 152,554 / 0 |
| f32_einsum | 0 / 0 ; 0 / 0 ; 0 / 0 | 153,179 / 0 ; 413,867 / 0 ; 157,295 / 0 | 157,728 / 0 ; 425,639 / 2 ; 160,012 / 0 |
| f64_blas_round | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 |
| f64_einsum_round | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 |
| fsum_round | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 | - |
| int_f32blas | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 |
| int_i8tile | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 |
| int_p2tile | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 |
| int_np_int32 | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 ; 0 / 0 |

Cross-implementation (the same sum computed by different code), same cell format:

| pair | 2560x2560 | 6912x2560 | 2560x6912 |
|---|---|---|---|
| f32_blas vs f32_einsum | 150,156 / 0 | 406,012 / 0 | 155,494 / 0 |
| f32_blas vs f64_blas_round | 146,173 / 0 | 394,915 / 0 | 148,877 / 0 |
| f32_blas vs fsum_round | 18,255 / 0 | 49,410 / 0 | 18,568 / 0 |
| f64_blas_round vs f64_einsum_round | 0 / 0 | 0 / 0 | 0 / 0 |
| f64_blas_round vs fsum_round | 0 / 0 | 0 / 0 | 0 / 0 |
| int_f32blas vs int_i8tile / int_p2tile / int_np_int32 | 0 / 0 | 0 / 0 | 0 / 0 |

- **Thread count changed nothing for any path on this box.** OpenBLAS appears to split gemv rows across threads here, so each output keeps its summation order. That holds for this build and kernel only. It's not a guarantee.
- **Column order, batch size and implementation all change fp32 bits.** 91% to 96% of outputs change under a shuffle, and 92% to 98% between a B=1 and a B=8 call. The drift is small: the largest shuffle difference was 7.6e-5 to 1.2e-4, on outputs with a typical size of 39 to 64. Only 2 of 442,368 trits flipped (einsum, 6912x2560, B=1 vs B=8), and none flipped for BLAS.
- **float64 accumulation rounded to fp32 gave the same bits in every test** and matched fsum exactly. The four int paths gave 0 differences everywhere and match each other bit for bit.
- **W8A8 contrast.** With int8 weights in fp32 BLAS, the worst-case sum (127² x K = 41.3M to 111.5M) can pass 2^24, so exactness isn't guaranteed. In these runs the largest |sum| was 293k and nothing differed. The 2^24 headroom for any K up to 132,104 is what ternary specifically adds.

## Results: speed and bytes
Times are ms per call, median (min-max). Weight bytes include the tile buffer and LUT where used.

**2560x2560**

| path | weight bytes | 1 thr B=1 | 1 thr B=8 | 8 thr B=1 | 8 thr B=8 |
|---|---|---|---|---|---|
| f32_blas | 26.21 MB | 0.863 (0.802-1.561) | 1.737 (1.719-2.121) | 0.137 (0.130-0.246) | 0.257 (0.232-0.379) |
| f32_einsum | 26.21 MB | 0.922 (0.852-1.046) | 23.689 (22.459-37.734) | 1.121 (1.089-1.584) | 26.054 (23.169-29.176) |
| f64_blas_round | 52.43 MB | 1.777 (1.719-3.271) | 3.448 (3.430-3.654) | 0.271 (0.263-0.519) | 0.707 (0.693-1.423) |
| f64_einsum_round | 52.43 MB | 1.969 (1.777-2.206) | 25.565 (24.596-29.221) | 2.106 (2.063-2.253) | 29.476 (26.443-32.837) |
| fsum_round | 26.21 MB | 85.515 (83.274-90.645) | - | 94.179 (86.946-99.593) | - |
| int_f32blas | 26.21 MB | 0.820 (0.812-0.901) | 1.883 (1.860-2.095) | 0.136 (0.133-0.285) | 0.398 (0.361-4.319) |
| int_i8tile | 7.21 MB | 0.897 (0.882-1.505) | 2.403 (2.285-3.007) | 1.149 (0.906-1.728) | 2.829 (2.673-2.922) |
| int_p2tile | 2.30 MB | 3.364 (2.969-9.047) | 5.097 (4.805-6.036) | 4.303 (3.900-4.625) | 5.678 (5.355-6.236) |
| int_np_int32 | 6.55 MB | 3.728 (3.426-7.053) | 21.986 (20.930-27.328) | 3.809 (3.497-8.172) | 24.830 (21.980-26.954) |

**6912x2560**

| path | weight bytes | 1 thr B=1 | 1 thr B=8 | 8 thr B=1 | 8 thr B=8 |
|---|---|---|---|---|---|
| f32_blas | 70.78 MB | 2.320 (2.205-4.522) | 4.654 (4.574-5.898) | 0.364 (0.348-0.909) | 1.222 (1.175-2.830) |
| f32_einsum | 70.78 MB | 2.519 (2.243-3.142) | 69.751 (62.640-73.868) | 2.943 (2.842-3.389) | 73.097 (66.089-76.996) |
| f64_blas_round | 141.56 MB | 4.950 (4.495-12.389) | 10.617 (9.825-12.597) | 0.642 (0.601-2.935) | 2.118 (1.649-9.979) |
| f64_einsum_round | 141.56 MB | 5.830 (4.773-13.134) | 81.174 (73.522-86.227) | 5.636 (5.281-7.514) | 78.038 (71.328-84.534) |
| fsum_round | 70.78 MB | 244.374 (237.337-257.300) | - | 246.820 (241.956-251.721) | - |
| int_f32blas | 70.78 MB | 2.178 (2.118-4.448) | 5.884 (4.786-6.140) | 0.642 (0.621-1.165) | 1.785 (1.584-4.990) |
| int_i8tile | 18.35 MB | 2.517 (2.351-4.040) | 6.314 (5.962-8.091) | 2.520 (2.374-3.966) | 7.087 (6.740-7.806) |
| int_p2tile | 5.08 MB | 9.031 (8.109-10.647) | 14.437 (12.919-16.737) | 9.604 (8.428-11.019) | 14.987 (13.850-16.263) |
| int_np_int32 | 17.69 MB | 19.161 (16.309-27.172) | 78.291 (71.530-86.562) | 19.866 (16.897-27.538) | 76.468 (73.570-84.457) |

**2560x6912**

| path | weight bytes | 1 thr B=1 | 1 thr B=8 | 8 thr B=1 | 8 thr B=8 |
|---|---|---|---|---|---|
| f32_blas | 70.78 MB | 2.586 (2.544-4.656) | 5.502 (4.990-7.286) | 0.406 (0.352-0.612) | 0.916 (0.851-0.979) |
| f32_einsum | 70.78 MB | 2.617 (2.397-2.892) | 84.672 (81.894-89.213) | 2.356 (2.281-2.652) | 84.320 (80.941-87.301) |
| f64_blas_round | 141.56 MB | 4.936 (4.421-11.274) | 11.155 (9.660-17.695) | 0.758 (0.741-1.756) | 2.013 (1.913-3.192) |
| f64_einsum_round | 141.56 MB | 5.208 (4.731-5.810) | 73.279 (70.312-81.817) | 5.327 (4.766-7.631) | 77.832 (71.770-83.310) |
| fsum_round | 70.78 MB | 269.096 (262.315-274.510) | - | 265.249 (257.696-269.335) | - |
| int_f32blas | 70.78 MB | 2.356 (2.283-4.241) | 5.399 (4.991-6.273) | 0.378 (0.363-0.738) | 1.113 (0.917-1.448) |
| int_i8tile | 19.46 MB | 3.590 (3.212-4.921) | 7.169 (6.810-8.714) | 3.130 (3.086-4.654) | 6.900 (6.317-9.577) |
| int_p2tile | 6.20 MB | 16.773 (15.230-18.061) | 19.883 (18.701-22.633) | 15.368 (14.495-16.783) | 15.995 (15.294-17.921) |
| int_np_int32 | 17.69 MB | 17.380 (15.059-27.329) | 78.358 (69.601-85.382) | 16.718 (13.169-25.581) | 70.622 (67.628-76.879) |

Ratios against f64_blas_round, the cheapest float path with zero observed order sensitivity. Speed ratio = f64 time / int time, so above 1 means the int path is faster.

| shape | int path | 1 thr B=1 | 1 thr B=8 | 8 thr B=1 | 8 thr B=8 | bytes smaller |
|---|---|---|---|---|---|---|
| 2560x2560 | int_f32blas | 2.17x | 1.83x | 2.00x | 1.78x | 2.00x |
| 2560x2560 | int_i8tile | 1.98x | 1.43x | 0.24x | 0.25x | 7.27x |
| 2560x2560 | int_p2tile | 0.53x | 0.68x | 0.06x | 0.12x | 22.82x |
| 6912x2560 | int_f32blas | 2.27x | 1.80x | 1.00x | 1.19x | 2.00x |
| 6912x2560 | int_i8tile | 1.97x | 1.68x | 0.25x | 0.30x | 7.71x |
| 6912x2560 | int_p2tile | 0.55x | 0.74x | 0.07x | 0.14x | 27.85x |
| 2560x6912 | int_f32blas | 2.10x | 2.07x | 2.01x | 1.81x | 2.00x |
| 2560x6912 | int_i8tile | 1.38x | 1.56x | 0.24x | 0.29x | 7.27x |
| 2560x6912 | int_p2tile | 0.29x | 0.56x | 0.05x | 0.13x | 22.84x |

The 8-thread B=1 ratio of 1.00x at 6912x2560 looked like a noisy run: int_f32blas took 0.642 ms there against 0.364 ms for plain f32_blas. An interleaved recheck (31 reps, 8 threads, B=1, `r1455/d20261017/recheck8.json`) gave f64_blas_round / int_f32blas = 0.262 / 0.153, 0.713 / 0.358 and 0.696 / 0.381 ms. That's 1.72x, 1.99x and 1.83x. In the same recheck plain f32_blas ran 0.118, 0.318 and 0.337 ms, so the int8 quantize step adds about 0.04 ms per token.

## Results: error
Relative L2 over 64 Gaussian tokens, median / max. The last column is the share of output trits that differ from the float64 T @ x reference trits. The int path's residual against the A8 forward is only the fp32 rounding of the output. A separate check found all four int paths bit-equal to the float64 (T @ q) * s result rounded to fp32, with 0 of 163,840 / 442,368 / 163,840 elements differing.

| shape | path | vs float64 T @ x | vs A8 forward (T @ q) * s | outlier tokens vs T @ x | trits off the float reference |
|---|---|---|---|---|---|
| 2560x2560 | f32_blas | 2.5e-7 / 2.7e-7 | 0.83% / 1.06% | 2.5e-7 / 3.1e-7 | 0.000% |
| 2560x2560 | f64_blas_round | 2.5e-8 / 2.6e-8 | 0.83% / 1.06% | 2.5e-8 / 2.6e-8 | 0.000% |
| 2560x2560 | int (all four) | 0.83% / 1.06% | 2.7e-8 / 2.7e-8 (bit-exact) | 5.27% / 8.83% | 0.461% |
| 6912x2560 | f32_blas | 2.6e-7 / 2.6e-7 | 0.82% / 0.97% | 2.6e-7 / 3.1e-7 | 0.000% |
| 6912x2560 | f64_blas_round | 2.5e-8 / 2.6e-8 | 0.82% / 0.97% | 2.5e-8 / 2.6e-8 | 0.000% |
| 6912x2560 | int (all four) | 0.82% / 0.97% | 2.6e-8 / 2.6e-8 (bit-exact) | 5.55% / 8.88% | 0.476% |
| 2560x6912 | f32_blas | 3.0e-7 / 3.1e-7 | 0.90% / 1.15% | 3.0e-7 / 3.4e-7 | 0.000% |
| 2560x6912 | f64_blas_round | 2.5e-8 / 2.6e-8 | 0.90% / 1.15% | 2.5e-8 / 2.6e-8 | 0.000% |
| 2560x6912 | int (all four) | 0.90% / 1.15% | 2.6e-8 / 2.6e-8 (bit-exact) | 6.75% / 9.86% | 0.536% |

## Verdict
**Viable, for the W1.58A8 forward that bitnetd runs.** If the overlay needs Debian and Alpine (or 1 and 8 threads, or B=1 and B=8) to produce identical matvec bits, the exact int path is the cheapest way to get them. It's also the only fast path that's order-proof by arithmetic rather than by observation.

- **int_f32blas against f64_blas_round:**
  - At 1 thread it's 2.10x to 2.27x faster for B=1 and 1.80x to 2.07x faster for B=8, with half the weight bytes (fp32 vs float64).
  - At 8 threads it's 1.72x to 2.00x faster for B=1 (main run plus recheck; one noisy 1.00x cell above) and 1.19x to 1.81x for B=8.
  - It runs at the same speed as plain fp32 BLAS, so exactness costs only the int8 quantize step.
- **int_i8tile** holds 7.3x to 7.7x fewer bytes than f64_blas_round and is still 1.38x to 1.98x faster at 1 thread for B=1. My tile loop doesn't use threads, though, so at 8 threads it's 3.3x to 4.2x slower.
- **int_p2tile** is 23x to 28x smaller but 1.8x to 3.4x slower at 1 thread, because of the LUT decode. This matches the earlier packed-tile notes.
- **Against fsum,** the only float path that's order-proof on any machine by construction, the int paths are 104x to 114x faster at 1 thread for B=1: 0.82 to 2.36 ms against 85.5 to 269 ms.
- **Error.** The int path is bit-exact (the correctly rounded fp32 value) against the A8 forward BitNet b1.58 defines. Against float activations it's 0.82% to 0.90% median relative L2 (max 0.97% to 1.15%), which is the same int8 cost the earlier notes measured.

Limits:

- **Exactness isn't accuracy.** On this box, fp32 order noise barely reached the trit level: 0 to 2 flips per test. Int8 activation quantization moves far more trits away from the float reference: 0.46% to 0.54% of outputs, which is about 755 to 2,106 per test. The int path is the right choice when the overlay needs identical bits on both bases (keys, close hashes, cached trit streams). It isn't the right choice when it needs to match a float-activation model.
- **Outliers.** With 0.5% outlier channels, plain absmax is 5.3% to 6.7% median relative L2. Outlier-heavy layers still need the split from the 02:55 and 05:55 notes. Its fp32 side matvec should then run in float64, or on integer-scaled columns, to stay reproducible.
- **Scope of the zero-diff results.** The int path's order-proofness is a math fact (integer sums under 2^24). The f64 path's zero differences are a measurement on one box and one OpenBLAS build. Alpine musl wheels, other CPUs and the Asahi MLX path weren't tested.

For the overlay: run the bitnetd matvec as integer-valued fp32, or as int8-resident tiles where memory matters. Keep the activation scale step to absmax, divide, rint and clip. These are correctly rounded IEEE ops that need no libm, so they match across glibc and musl. Then Winsor's gamma flip margin only has to cover the weight trits, not the matvec outputs.

## Ops check
In local `/workspace/JuniorLLM` (tip 5a3eaec; offline, blender off), `python3 scripts/terraform_lean_prod.py code` and `... llm` both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false` and `"launch": false`.

## This wave's pair files (2026-10-17 14:05)
- Trit Pack & I2_S: [JuniorLLM scripts/junioros_overlay_jtr1_key.py @ 7d40533](https://github.com/cloudcover95/JuniorLLM/blob/7d40533/scripts/junioros_overlay_jtr1_key.py) (BETA_TO_OS JuniorOS overlay, Chief pair A per its commit)
- Winsor & Quant Error: [JuniorStock quant/calibration/junioros_overlay_trit_repro.py @ e82b55b](https://github.com/cloudcover95/JuniorStock/blob/e82b55b/quant/calibration/junioros_overlay_trit_repro.py) (JuniorOS overlay, quant-error side, buddy of 7d40533)
- Side-Commit & Rate Lock: [web3node asahi_mlx_probe_receipt_ledger.py @ 76c2cd0](https://github.com/cloudcover95/web3node/blob/76c2cd0/asahi_mlx_probe_receipt_ledger.py) (ROADMAP #4 Asahi MLX)
- Handshake & Ports: [JuniorLLM ports/asahi_mlx_lane_port_map.py @ 008e4ab](https://github.com/cloudcover95/JuniorLLM/blob/008e4ab/ports/asahi_mlx_lane_port_map.py) (ROADMAP #4 Asahi MLX)

## Next question
With the matvec exact, which of the remaining float steps in a BitNet block break bit-reproducibility, and what does fixing them cost? The candidates are RMSNorm's sum of squares (a reduction), softmax's exp (libm, which can differ between glibc and musl), and the ReLU²/SiLU activation. For a 2-layer numpy block, perturb only those steps: shuffle the reduction order, and nudge exp by 1 ulp to stand in for a different libm. Count changed output bits, trits and argmax tokens, then time float64-accumulate and integer-LUT replacements against the plain fp32 versions.

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-17_1055_ue5-watts-gate-trit-vs-float-certified.md.

close: 9f94d260246ba291
