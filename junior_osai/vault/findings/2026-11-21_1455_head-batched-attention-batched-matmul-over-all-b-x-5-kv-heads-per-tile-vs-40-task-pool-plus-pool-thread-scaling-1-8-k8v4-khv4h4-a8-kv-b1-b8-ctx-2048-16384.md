---
local: true
date: 2026-11-21
wave: "2026-11-21 14:55 (ran ~14:45–15:00 MT; no pair reports reached chat this wave, quant derivation lane paused)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, attention, head-batched, batched-matmul, kv-cache, k8v4, khv4h4, a8-g64, threads, numpy, viable]
---

# Head-batched attention: one batched call over all 40 heads is 0.06–0.38x (slower), but 8 pool tasks of 5 batched heads each give 1.65–1.69x (K8V4) and 1.76–1.90x (khv4h4) at B = 8 / ctx 8192, bit-exact; the pool scales ~2.8–4.0x from 1 to 8 threads

The 10:55 note asked whether head-batched attention (one batched np.matmul per TK tile over all B × 5 KV heads) beats the 40-task 8-thread pool by ≥ 1.5x, and whether the pool gives any real parallelism. Answers:
- **One batched call over all B × 5 heads (hb, no pool) is much slower**: 0.06–0.38x of the pool at B = 8 and 0.26–0.54x at B = 1. It runs on one thread, and at B = 8 each tile's dequantized K/V block is (40, 2048, 128) f32 = 40 MiB, so it is memory-bound; at ctx 8192 its timing is also very noisy (up to 0.78 s).
- **The same batched tile code, cut into 8 head groups (5 heads each at B = 8) with one pool task per group (hbP8), clears 1.5x at B = 8 on both seeds**: ctx 8192 K8V4 **1.69x / 1.65x**, khv4h4 **1.90x / 1.76x**, a8 1.54x / 1.52x; ctx 2048 K8V4 1.99x / 2.22x, khv4h4 2.26x / 2.21x, a8 1.63x / 1.61x (medians, run1 / run2). Per-round ranges are wide (table below); a few rounds at ctx 8192 drop to 1.04–1.4x.
- **B = 1 has no gain**: with only 5 heads, hbP8 makes 5 one-head tasks, the same as the baseline (0.86–1.11x).
- **The pool does give real parallelism, correcting the 10:55 reading**: pool1 → pool8 is 2.82–3.99x at B = 8 / ctx 8192 and 2.41–3.61x at B = 1 / ctx 8192–16384 (at most 5 tasks). At B = 8, 4 threads already come close to 8 (pool8 time is 0.82–0.99 of pool4 at ctx 8192, and pool4 is slightly faster than pool8 for K8V4 / khv4h4 at ctx 2048). So the 10:55 split-K loss came from per-task overhead outrunning the gain, not from a pool that ran serially. hbP8 wins because it cuts B = 8 from 40 tasks to 8 larger ones while keeping all 8 workers busy.
- **Exact**: every variant matches the pool8 output bit for bit (max abs diff 0.0e+00, min cos 1.0000000) in every cell of both runs, so the vs-f32 error is unchanged per format (min cos at ctx 8192 on these draws: K8V4 0.981–0.982, khv4h4 0.992–0.993, a8 0.987–0.989). KV bytes are unchanged: the stacked (B·5, n, ...) layout holds the same codes and scales.

Verdict: **viable** for B = 8 attention, as the grouped head-batched form (hbP8): ≥ 1.5x median on both seeds for K8V4 and khv4h4 at ctx 2048 and 8192 with bit-exact output. The literal single-call form from the question is not viable (0.06–0.38x at B = 8). Projected B = 8 decode step from the 05:55 numbers is in the table below. Nothing changes at B = 1.

## Question

[[2026-11-21_1055_flash-decoding-split-k-history-partition-across-8-workers-merge-partial-m-l-acc-k8v4-khv4h4-a8-kv-b1-b8-ctx-2048-16384-vs-per-stream-head-tasks]] found that task count, not history length, drives numpy attention cost here. Its next question:
- At B = 8 / ctx 8192, does **head-batched attention** (one batched np.matmul per TK tile over all B × 5 KV heads, i.e. a (40, 4, 128) @ (40, 128, TK) score GEMM and one batched P @ V, with K8V4 (and khv4h4) dequant done per tile for all heads at once, online softmax across tiles) beat the 40-task 8-thread pool by ≥ 1.5x? (B = 1 and ctx 2048 / 16384 also checked.)
- How does the baseline scale from a 1-thread to an 8-thread pool (is the pool giving real parallelism)?

## Method

- 'hb.py' (below) loads '/workspace/bench0255_1121/iq.py' without its 'main()' (same KV formats, synthetic KV generator with planted outlier K channels, 'iq.tile' online softmax, 8-thread 'iq.POOL').
- Baseline pool8 is the unchanged 05:55 / 10:55 path 'iq.attn': one task per (stream, KV head), TK = 2048 old tiles plus the A8 1024 window.
- Thread scaling: the same 'iq.attn' with 'iq.POOL' swapped for a 1-, 2-, or 4-thread ThreadPoolExecutor (pool1 / pool2 / pool4).
- hb: every stream's codes and scales are stacked once into (B·5, n, ...) arrays (not timed; same bytes). Per tile, K is dequantized for all heads in one dq8 call, scores come from one batched matmul (BH, 4, 128) @ (BH, 128, T) (khv4h4 adds one batched (BH, 4, 4) @ (BH, 4, T) for the 4 hot f16 channels), V (A4 g32 old / A8 g64 window) is dequantized for all heads in one call, and P @ V is one batched matmul. Online softmax runs on (BH, 4, 1) m / l. No pool.
- hbP8: hb's tile code run on 8 contiguous head groups (B = 8: 8 × 5 heads; B = 1: 5 × 1 head), one 'iq.POOL' task per group.
- Formats: k8v4 (i2sd default), khv4h4, a8. Grid: ctx {2048, 8192} × B {1, 8}, plus ctx 16384 at B = 1. 5 KV heads × 4 query heads, HD 128.
- Timing: attention call only, 2 warmup passes over all 18 configs per cell, then 7 rounds with config order rotated each round, 5 reps per config per round (n = 35). Median (min–max), IQR, and the per-round speedup range (round medians vs the same round's pool8) are reported.
- Error: max abs diff and min-over-heads cos vs the pool8 output, and min cos / max abs vs exact f32 softmax attention over the f32 history ('iq.ref').
- Two runs: SEED 11211455 (new) and SEED 11211055 (the 10:55 seed). Queries are fresh standard-normal draws × 2 × 1/√HD.
- Python 3 + numpy, OPENBLAS_NUM_THREADS=1, 8 cores. No network, nothing bound, no installs. Wall time ~2.0 + 1.6 min.

Command:

    cd /workspace/bench1455_1121
    OPENBLAS_NUM_THREADS=1 SEED=11211455 python3 hb.py results_run1.json > run1.txt
    OPENBLAS_NUM_THREADS=1 SEED=11211055 python3 hb.py results_run2.json > run2.txt

## Results

### Summary (both runs; speedup = pool8 median / variant median)

| fmt | ctx | B | pool8 ms (run1 / run2) | hb (one batched call) speedup (run1 / run2) | hbP8 (8 head-group tasks) speedup med (run1 / run2) | hbP8 per-round range | hbP8 ms (run1 / run2) | max abs vs pool8 | min cos vs f32 | KV MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| k8v4 | 2048 | 1 | 1.40 / 1.34 | 0.523x / 0.508x | **1.111x / 1.059x** | 0.83–1.24x | 1.26 / 1.27 | 0.0e+00 | 0.992654 | 2.38 |
| khv4h4 | 2048 | 1 | 1.54 / 1.49 | 0.541x / 0.525x | **0.889x / 0.859x** | 0.74–1.01x | 1.73 / 1.73 | 0.0e+00 | 0.994334 | 2.32 |
| a8 | 2048 | 1 | 1.00 / 1.00 | 0.426x / 0.429x | **0.928x / 0.993x** | 0.84–1.14x | 1.07 / 1.01 | 0.0e+00 | 0.996499 | 2.66 |
| k8v4 | 2048 | 8 | 10.31 / 9.92 | 0.249x / 0.325x | **1.988x / 2.223x** | 1.34–3.00x | 5.19 / 4.46 | 0.0e+00 | 0.986626 | 19.06 |
| khv4h4 | 2048 | 8 | 11.78 / 12.00 | 0.255x / 0.381x | **2.264x / 2.211x** | 1.97–2.41x | 5.20 / 5.43 | 0.0e+00 | 0.992255 | 18.59 |
| a8 | 2048 | 8 | 6.50 / 6.56 | 0.063x / 0.233x | **1.634x / 1.610x** | 1.38–2.23x | 3.98 / 4.08 | 0.0e+00 | 0.992906 | 21.25 |
| k8v4 | 8192 | 1 | 4.01 / 4.05 | 0.325x / 0.332x | **1.033x / 0.976x** | 0.83–1.37x | 3.88 / 4.15 | 0.0e+00 | 0.981289 | 8.71 |
| khv4h4 | 8192 | 1 | 4.81 / 5.01 | 0.366x / 0.390x | **0.900x / 0.931x** | 0.74–1.07x | 5.35 / 5.39 | 0.0e+00 | 0.992911 | 8.42 |
| a8 | 8192 | 1 | 3.58 / 3.56 | 0.342x / 0.342x | **1.048x / 0.942x** | 0.89–1.27x | 3.42 / 3.78 | 0.0e+00 | 0.986900 | 10.62 |
| k8v4 | 8192 | 8 | 29.37 / 26.32 | 0.061x / 0.101x | **1.686x / 1.652x** | 1.04–1.97x | 17.42 / 15.93 | 0.0e+00 | 0.981873 | 69.69 |
| khv4h4 | 8192 | 8 | 35.18 / 31.10 | 0.093x / 0.158x | **1.898x / 1.756x** | 1.38–2.11x | 18.53 / 17.71 | 0.0e+00 | 0.992279 | 67.34 |
| a8 | 8192 | 8 | 23.75 / 20.56 | 0.065x / 0.142x | **1.543x / 1.520x** | 1.31–1.91x | 15.39 / 13.52 | 0.0e+00 | 0.989232 | 85.00 |
| k8v4 | 16384 | 1 | 7.76 / 6.81 | 0.305x / 0.260x | **1.051x / 0.995x** | 0.78–1.30x | 7.38 / 6.85 | 0.0e+00 | 0.990719 | 17.15 |
| khv4h4 | 16384 | 1 | 8.05 / 8.03 | 0.303x / 0.298x | **0.896x / 0.923x** | 0.76–1.17x | 8.99 / 8.71 | 0.0e+00 | 0.993476 | 16.54 |
| a8 | 16384 | 1 | 5.99 / 5.73 | 0.291x / 0.278x | **0.996x / 0.990x** | 0.73–1.17x | 6.01 / 5.79 | 0.0e+00 | 0.997340 | 21.25 |

### Pool thread scaling of the baseline (attn ms median, run1 / run2)

| fmt | ctx | B | pool1 ms (run1 / run2) | pool2 | pool4 | pool8 | pool8 / pool1 speedup (run1 / run2) | best pool |
|---|---|---|---|---|---|---|---|---|
| k8v4 | 2048 | 1 | 2.46 / 2.52 | 1.72 / 1.70 | 1.59 / 1.52 | 1.40 / 1.34 | 1.76x / 1.87x | 8 / 8 |
| khv4h4 | 2048 | 1 | 2.72 / 2.64 | 1.88 / 1.84 | 1.89 / 1.84 | 1.54 / 1.49 | 1.77x / 1.78x | 8 / 8 |
| a8 | 2048 | 1 | 2.60 / 2.49 | 1.58 / 1.60 | 1.36 / 1.29 | 1.00 / 1.00 | 2.60x / 2.49x | 8 / 8 |
| k8v4 | 2048 | 8 | 20.52 / 21.01 | 12.17 / 12.07 | 9.29 / 9.27 | 10.31 / 9.92 | 1.99x / 2.12x | 4 / 4 |
| khv4h4 | 2048 | 8 | 22.15 / 22.99 | 13.22 / 13.88 | 10.92 / 11.23 | 11.78 / 12.00 | 1.88x / 1.92x | 4 / 4 |
| a8 | 2048 | 8 | 20.75 / 21.63 | 11.22 / 11.57 | 7.12 / 7.44 | 6.50 / 6.56 | 3.19x / 3.30x | 8 / 8 |
| k8v4 | 8192 | 1 | 11.69 / 11.14 | 7.20 / 7.29 | 5.66 / 5.50 | 4.01 / 4.05 | 2.92x / 2.75x | 8 / 8 |
| khv4h4 | 8192 | 1 | 12.57 / 12.06 | 7.84 / 7.80 | 6.22 / 6.16 | 4.81 / 5.01 | 2.61x / 2.41x | 8 / 8 |
| a8 | 8192 | 1 | 10.40 / 9.96 | 6.15 / 6.17 | 4.76 / 4.48 | 3.58 / 3.56 | 2.91x / 2.80x | 8 / 8 |
| k8v4 | 8192 | 8 | 93.24 / 93.10 | 49.73 / 50.75 | 30.97 / 30.25 | 29.37 / 26.32 | 3.17x / 3.54x | 8 / 8 |
| khv4h4 | 8192 | 8 | 99.35 / 98.58 | 54.70 / 53.62 | 35.64 / 33.39 | 35.18 / 31.10 | 2.82x / 3.17x | 8 / 8 |
| a8 | 8192 | 8 | 82.19 / 82.10 | 44.18 / 43.96 | 26.16 / 24.97 | 23.75 / 20.56 | 3.46x / 3.99x | 8 / 8 |
| k8v4 | 16384 | 1 | 22.76 / 23.31 | 14.37 / 14.54 | 11.07 / 10.46 | 7.76 / 6.81 | 2.93x / 3.42x | 8 / 8 |
| khv4h4 | 16384 | 1 | 25.06 / 25.29 | 15.37 / 15.25 | 11.55 / 11.61 | 8.05 / 8.03 | 3.11x / 3.15x | 8 / 8 |
| a8 | 16384 | 1 | 20.36 / 20.69 | 11.93 / 12.23 | 9.00 / 8.66 | 5.99 / 5.73 | 3.40x / 3.61x | 8 / 8 |

### End-to-end projection, B = 8 (not re-timed)

Applies the smaller of this run's two hbP8 speedups to the 05:55 run1 attention time; weights and the rest of the step stay as measured at 05:55.

| cfg (05:55 run1 step) | 05:55 step ms | 05:55 attn ms | hbP8 speedup (min of runs) | projected step ms | projected step speedup |
|---|---|---|---|---|---|
| u32+k8v4_ctx2048_B8 | 26.05 | 10.83 | 1.988x | 20.67 | 1.260x |
| u32+khv4h4_ctx2048_B8 | 28.88 | 12.94 | 2.211x | 21.79 | 1.325x |
| u32+a8_ctx2048_B8 | 22.65 | 7.31 | 1.610x | 19.88 | 1.139x |
| u32+k8v4_ctx8192_B8 | 44.33 | 27.75 | 1.652x | 33.38 | 1.328x |
| u32+khv4h4_ctx8192_B8 | 49.12 | 33.50 | 1.756x | 34.70 | 1.415x |
| u32+a8_ctx8192_B8 | 40.35 | 23.26 | 1.520x | 32.39 | 1.246x |

### Run 1 (SEED 11211455)

| cfg | attn ms med (min–max) | IQR ms | speedup vs pool8 (per-round range) | cos vs pool8 | max abs vs pool8 | vs f32 cos | max abs vs f32 | KV MiB |
|---|---|---|---|---|---|---|---|---|
| k8v4_pool1_ctx2048_B1 | 2.46 (2.34–2.77) | 0.16 | 0.568x (0.50–0.58) | 1.0000000 | 0.0e+00 | 0.992654 | 2.93e-01 | 2.38 |
| k8v4_pool2_ctx2048_B1 | 1.72 (1.53–2.11) | 0.19 | 0.812x (0.69–0.87) | 1.0000000 | 0.0e+00 | 0.992654 | 2.93e-01 | 2.38 |
| k8v4_pool4_ctx2048_B1 | 1.59 (1.42–2.21) | 0.24 | 0.879x (0.75–0.92) | 1.0000000 | 0.0e+00 | 0.992654 | 2.93e-01 | 2.38 |
| k8v4_pool8_ctx2048_B1 | 1.40 (1.14–2.06) | 0.25 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.992654 | 2.93e-01 | 2.38 |
| k8v4_hb_ctx2048_B1 | 2.67 (2.52–2.98) | 0.15 | 0.523x (0.44–0.55) | 1.0000000 | 0.0e+00 | 0.992654 | 2.93e-01 | 2.38 |
| k8v4_hbP8_ctx2048_B1 | 1.26 (1.05–2.25) | 0.56 | 1.111x (0.83–1.18) | 1.0000000 | 0.0e+00 | 0.992654 | 2.93e-01 | 2.38 |
| khv4h4_pool1_ctx2048_B1 | 2.72 (2.45–3.08) | 0.25 | 0.565x (0.54–0.59) | 1.0000000 | 0.0e+00 | 0.994334 | 2.20e-01 | 2.32 |
| khv4h4_pool2_ctx2048_B1 | 1.88 (1.75–2.52) | 0.20 | 0.819x (0.79–0.90) | 1.0000000 | 0.0e+00 | 0.994334 | 2.20e-01 | 2.32 |
| khv4h4_pool4_ctx2048_B1 | 1.89 (1.54–2.40) | 0.23 | 0.816x (0.76–0.99) | 1.0000000 | 0.0e+00 | 0.994334 | 2.20e-01 | 2.32 |
| khv4h4_pool8_ctx2048_B1 | 1.54 (1.38–2.21) | 0.26 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.994334 | 2.20e-01 | 2.32 |
| khv4h4_hb_ctx2048_B1 | 2.84 (2.68–3.74) | 0.22 | 0.541x (0.52–0.60) | 1.0000000 | 0.0e+00 | 0.994334 | 2.20e-01 | 2.32 |
| khv4h4_hbP8_ctx2048_B1 | 1.73 (1.47–2.59) | 0.45 | 0.889x (0.74–1.01) | 1.0000000 | 0.0e+00 | 0.994334 | 2.20e-01 | 2.32 |
| a8_pool1_ctx2048_B1 | 2.60 (2.36–3.49) | 0.33 | 0.384x (0.36–0.42) | 1.0000000 | 0.0e+00 | 0.996499 | 2.07e-01 | 2.66 |
| a8_pool2_ctx2048_B1 | 1.58 (1.48–2.20) | 0.29 | 0.630x (0.57–0.70) | 1.0000000 | 0.0e+00 | 0.996499 | 2.07e-01 | 2.66 |
| a8_pool4_ctx2048_B1 | 1.36 (1.18–2.28) | 0.21 | 0.735x (0.65–0.82) | 1.0000000 | 0.0e+00 | 0.996499 | 2.07e-01 | 2.66 |
| a8_pool8_ctx2048_B1 | 1.00 (0.81–1.76) | 0.28 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.996499 | 2.07e-01 | 2.66 |
| a8_hb_ctx2048_B1 | 2.34 (2.22–3.32) | 0.19 | 0.426x (0.40–0.46) | 1.0000000 | 0.0e+00 | 0.996499 | 2.07e-01 | 2.66 |
| a8_hbP8_ctx2048_B1 | 1.07 (0.82–1.71) | 0.27 | 0.928x (0.88–1.14) | 1.0000000 | 0.0e+00 | 0.996499 | 2.07e-01 | 2.66 |
| k8v4_pool1_ctx2048_B8 | 20.52 (19.00–22.20) | 1.69 | 0.503x (0.45–0.57) | 1.0000000 | 0.0e+00 | 0.990865 | 4.48e-01 | 19.06 |
| k8v4_pool2_ctx2048_B8 | 12.17 (11.17–19.06) | 1.47 | 0.848x (0.55–0.96) | 1.0000000 | 0.0e+00 | 0.990865 | 4.48e-01 | 19.06 |
| k8v4_pool4_ctx2048_B8 | 9.29 (8.67–12.01) | 1.29 | 1.110x (0.87–1.17) | 1.0000000 | 0.0e+00 | 0.990865 | 4.48e-01 | 19.06 |
| k8v4_pool8_ctx2048_B8 | 10.31 (9.00–12.26) | 1.03 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.990865 | 4.48e-01 | 19.06 |
| k8v4_hb_ctx2048_B8 | 41.50 (32.41–70.79) | 10.17 | 0.249x (0.21–0.29) | 1.0000000 | 0.0e+00 | 0.990865 | 4.48e-01 | 19.06 |
| k8v4_hbP8_ctx2048_B8 | 5.19 (3.78–7.40) | 1.10 | 1.988x (1.91–2.30) | 1.0000000 | 0.0e+00 | 0.990865 | 4.48e-01 | 19.06 |
| khv4h4_pool1_ctx2048_B8 | 22.15 (20.06–33.03) | 2.41 | 0.532x (0.49–0.59) | 1.0000000 | 0.0e+00 | 0.992255 | 4.31e-01 | 18.59 |
| khv4h4_pool2_ctx2048_B8 | 13.22 (12.05–18.60) | 0.85 | 0.891x (0.84–0.96) | 1.0000000 | 0.0e+00 | 0.992255 | 4.31e-01 | 18.59 |
| khv4h4_pool4_ctx2048_B8 | 10.92 (10.13–12.49) | 0.88 | 1.078x (1.02–1.17) | 1.0000000 | 0.0e+00 | 0.992255 | 4.31e-01 | 18.59 |
| khv4h4_pool8_ctx2048_B8 | 11.78 (10.60–17.05) | 1.10 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.992255 | 4.31e-01 | 18.59 |
| khv4h4_hb_ctx2048_B8 | 46.24 (37.01–77.94) | 13.35 | 0.255x (0.24–0.29) | 1.0000000 | 0.0e+00 | 0.992255 | 4.31e-01 | 18.59 |
| khv4h4_hbP8_ctx2048_B8 | 5.20 (4.79–7.70) | 0.96 | 2.264x (2.07–2.41) | 1.0000000 | 0.0e+00 | 0.992255 | 4.31e-01 | 18.59 |
| a8_pool1_ctx2048_B8 | 20.75 (18.71–24.36) | 1.31 | 0.313x (0.29–0.33) | 1.0000000 | 0.0e+00 | 0.996019 | 4.64e-01 | 21.25 |
| a8_pool2_ctx2048_B8 | 11.22 (10.19–12.69) | 1.01 | 0.580x (0.53–0.61) | 1.0000000 | 0.0e+00 | 0.996019 | 4.64e-01 | 21.25 |
| a8_pool4_ctx2048_B8 | 7.12 (6.52–8.37) | 0.61 | 0.914x (0.85–0.98) | 1.0000000 | 0.0e+00 | 0.996019 | 4.64e-01 | 21.25 |
| a8_pool8_ctx2048_B8 | 6.50 (5.90–7.32) | 0.55 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.996019 | 4.64e-01 | 21.25 |
| a8_hb_ctx2048_B8 | 102.63 (71.52–163.56) | 28.35 | 0.063x (0.04–0.07) | 1.0000000 | 0.0e+00 | 0.996019 | 4.64e-01 | 21.25 |
| a8_hbP8_ctx2048_B8 | 3.98 (3.33–10.20) | 1.42 | 1.634x (1.38–1.73) | 1.0000000 | 0.0e+00 | 0.996019 | 4.64e-01 | 21.25 |
| k8v4_pool1_ctx8192_B1 | 11.69 (11.04–15.63) | 0.68 | 0.343x (0.31–0.41) | 1.0000000 | 0.0e+00 | 0.981289 | 3.64e-01 | 8.71 |
| k8v4_pool2_ctx8192_B1 | 7.20 (6.90–8.88) | 0.34 | 0.557x (0.53–0.65) | 1.0000000 | 0.0e+00 | 0.981289 | 3.64e-01 | 8.71 |
| k8v4_pool4_ctx8192_B1 | 5.66 (5.20–6.76) | 0.27 | 0.709x (0.68–0.82) | 1.0000000 | 0.0e+00 | 0.981289 | 3.64e-01 | 8.71 |
| k8v4_pool8_ctx8192_B1 | 4.01 (3.50–5.53) | 0.53 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.981289 | 3.64e-01 | 8.71 |
| k8v4_hb_ctx8192_B1 | 12.34 (11.53–16.03) | 0.73 | 0.325x (0.31–0.37) | 1.0000000 | 0.0e+00 | 0.981289 | 3.64e-01 | 8.71 |
| k8v4_hbP8_ctx8192_B1 | 3.88 (3.65–5.57) | 0.58 | 1.033x (0.83–1.20) | 1.0000000 | 0.0e+00 | 0.981289 | 3.64e-01 | 8.71 |
| khv4h4_pool1_ctx8192_B1 | 12.57 (11.77–13.65) | 0.71 | 0.383x (0.34–0.44) | 1.0000000 | 0.0e+00 | 0.992911 | 1.72e-01 | 8.42 |
| khv4h4_pool2_ctx8192_B1 | 7.84 (7.44–9.74) | 0.33 | 0.614x (0.54–0.71) | 1.0000000 | 0.0e+00 | 0.992911 | 1.72e-01 | 8.42 |
| khv4h4_pool4_ctx8192_B1 | 6.22 (5.82–7.98) | 0.55 | 0.774x (0.69–0.85) | 1.0000000 | 0.0e+00 | 0.992911 | 1.72e-01 | 8.42 |
| khv4h4_pool8_ctx8192_B1 | 4.81 (4.12–7.36) | 0.64 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.992911 | 1.72e-01 | 8.42 |
| khv4h4_hb_ctx8192_B1 | 13.14 (12.46–17.58) | 0.73 | 0.366x (0.34–0.41) | 1.0000000 | 0.0e+00 | 0.992911 | 1.72e-01 | 8.42 |
| khv4h4_hbP8_ctx8192_B1 | 5.35 (4.53–11.45) | 0.95 | 0.900x (0.74–1.04) | 1.0000000 | 0.0e+00 | 0.992911 | 1.72e-01 | 8.42 |
| a8_pool1_ctx8192_B1 | 10.40 (9.54–12.20) | 0.85 | 0.344x (0.32–0.38) | 1.0000000 | 0.0e+00 | 0.986900 | 3.57e-01 | 10.62 |
| a8_pool2_ctx8192_B1 | 6.15 (5.74–7.13) | 0.29 | 0.581x (0.56–0.65) | 1.0000000 | 0.0e+00 | 0.986900 | 3.57e-01 | 10.62 |
| a8_pool4_ctx8192_B1 | 4.76 (4.29–6.38) | 0.51 | 0.752x (0.70–0.81) | 1.0000000 | 0.0e+00 | 0.986900 | 3.57e-01 | 10.62 |
| a8_pool8_ctx8192_B1 | 3.58 (3.13–4.51) | 0.44 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.986900 | 3.57e-01 | 10.62 |
| a8_hb_ctx8192_B1 | 10.45 (9.86–16.22) | 0.77 | 0.342x (0.32–0.39) | 1.0000000 | 0.0e+00 | 0.986900 | 3.57e-01 | 10.62 |
| a8_hbP8_ctx8192_B1 | 3.42 (2.86–4.94) | 0.75 | 1.048x (0.89–1.27) | 1.0000000 | 0.0e+00 | 0.986900 | 3.57e-01 | 10.62 |
| k8v4_pool1_ctx8192_B8 | 93.24 (91.10–99.23) | 2.27 | 0.315x (0.27–0.36) | 1.0000000 | 0.0e+00 | 0.981873 | 3.17e-01 | 69.69 |
| k8v4_pool2_ctx8192_B8 | 49.73 (48.05–57.67) | 1.69 | 0.591x (0.52–0.69) | 1.0000000 | 0.0e+00 | 0.981873 | 3.17e-01 | 69.69 |
| k8v4_pool4_ctx8192_B8 | 30.97 (28.74–36.81) | 1.84 | 0.949x (0.83–1.09) | 1.0000000 | 0.0e+00 | 0.981873 | 3.17e-01 | 69.69 |
| k8v4_pool8_ctx8192_B8 | 29.37 (25.10–35.46) | 5.14 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.981873 | 3.17e-01 | 69.69 |
| k8v4_hb_ctx8192_B8 | 482.99 (255.24–781.22) | 204.56 | 0.061x (0.05–0.10) | 1.0000000 | 0.0e+00 | 0.981873 | 3.17e-01 | 69.69 |
| k8v4_hbP8_ctx8192_B8 | 17.42 (13.93–41.98) | 6.39 | 1.686x (1.04–1.97) | 1.0000000 | 0.0e+00 | 0.981873 | 3.17e-01 | 69.69 |
| khv4h4_pool1_ctx8192_B8 | 99.35 (96.58–113.32) | 3.69 | 0.354x (0.32–0.39) | 1.0000000 | 0.0e+00 | 0.992742 | 2.81e-01 | 67.34 |
| khv4h4_pool2_ctx8192_B8 | 54.70 (51.92–67.03) | 2.91 | 0.643x (0.53–0.71) | 1.0000000 | 0.0e+00 | 0.992742 | 2.81e-01 | 67.34 |
| khv4h4_pool4_ctx8192_B8 | 35.64 (33.47–43.97) | 2.21 | 0.987x (0.89–1.12) | 1.0000000 | 0.0e+00 | 0.992742 | 2.81e-01 | 67.34 |
| khv4h4_pool8_ctx8192_B8 | 35.18 (29.96–45.88) | 5.54 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.992742 | 2.81e-01 | 67.34 |
| khv4h4_hb_ctx8192_B8 | 379.02 (279.77–628.19) | 143.68 | 0.093x (0.07–0.11) | 1.0000000 | 0.0e+00 | 0.992742 | 2.81e-01 | 67.34 |
| khv4h4_hbP8_ctx8192_B8 | 18.53 (15.82–26.53) | 2.99 | 1.898x (1.69–2.11) | 1.0000000 | 0.0e+00 | 0.992742 | 2.81e-01 | 67.34 |
| a8_pool1_ctx8192_B8 | 82.19 (78.06–96.47) | 2.92 | 0.289x (0.24–0.32) | 1.0000000 | 0.0e+00 | 0.989232 | 3.23e-01 | 85.00 |
| a8_pool2_ctx8192_B8 | 44.18 (41.28–50.52) | 1.86 | 0.538x (0.41–0.60) | 1.0000000 | 0.0e+00 | 0.989232 | 3.23e-01 | 85.00 |
| a8_pool4_ctx8192_B8 | 26.16 (23.18–30.25) | 1.92 | 0.908x (0.73–0.95) | 1.0000000 | 0.0e+00 | 0.989232 | 3.23e-01 | 85.00 |
| a8_pool8_ctx8192_B8 | 23.75 (19.61–27.40) | 3.67 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.989232 | 3.23e-01 | 85.00 |
| a8_hb_ctx8192_B8 | 367.21 (279.31–648.45) | 179.04 | 0.065x (0.04–0.07) | 1.0000000 | 0.0e+00 | 0.989232 | 3.23e-01 | 85.00 |
| a8_hbP8_ctx8192_B8 | 15.39 (11.88–24.87) | 3.97 | 1.543x (1.31–1.91) | 1.0000000 | 0.0e+00 | 0.989232 | 3.23e-01 | 85.00 |
| k8v4_pool1_ctx16384_B1 | 22.76 (22.00–26.09) | 0.82 | 0.341x (0.30–0.41) | 1.0000000 | 0.0e+00 | 0.990719 | 2.54e-01 | 17.15 |
| k8v4_pool2_ctx16384_B1 | 14.37 (13.66–23.82) | 0.42 | 0.540x (0.48–0.67) | 1.0000000 | 0.0e+00 | 0.990719 | 2.54e-01 | 17.15 |
| k8v4_pool4_ctx16384_B1 | 11.07 (10.33–12.75) | 0.72 | 0.701x (0.63–0.85) | 1.0000000 | 0.0e+00 | 0.990719 | 2.54e-01 | 17.15 |
| k8v4_pool8_ctx16384_B1 | 7.76 (6.38–10.59) | 2.07 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.990719 | 2.54e-01 | 17.15 |
| k8v4_hb_ctx16384_B1 | 25.41 (23.94–30.36) | 2.98 | 0.305x (0.25–0.39) | 1.0000000 | 0.0e+00 | 0.990719 | 2.54e-01 | 17.15 |
| k8v4_hbP8_ctx16384_B1 | 7.38 (6.34–10.52) | 2.75 | 1.051x (0.78–1.30) | 1.0000000 | 0.0e+00 | 0.990719 | 2.54e-01 | 17.15 |
| khv4h4_pool1_ctx16384_B1 | 25.06 (23.51–27.81) | 1.40 | 0.321x (0.30–0.43) | 1.0000000 | 0.0e+00 | 0.993476 | 2.52e-01 | 16.54 |
| khv4h4_pool2_ctx16384_B1 | 15.37 (14.36–16.33) | 0.60 | 0.524x (0.49–0.68) | 1.0000000 | 0.0e+00 | 0.993476 | 2.52e-01 | 16.54 |
| khv4h4_pool4_ctx16384_B1 | 11.55 (10.70–12.76) | 0.72 | 0.697x (0.66–0.87) | 1.0000000 | 0.0e+00 | 0.993476 | 2.52e-01 | 16.54 |
| khv4h4_pool8_ctx16384_B1 | 8.05 (7.06–11.09) | 1.04 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.993476 | 2.52e-01 | 16.54 |
| khv4h4_hb_ctx16384_B1 | 26.54 (25.36–30.60) | 2.68 | 0.303x (0.28–0.39) | 1.0000000 | 0.0e+00 | 0.993476 | 2.52e-01 | 16.54 |
| khv4h4_hbP8_ctx16384_B1 | 8.99 (7.96–11.82) | 0.99 | 0.896x (0.76–1.17) | 1.0000000 | 0.0e+00 | 0.993476 | 2.52e-01 | 16.54 |
| a8_pool1_ctx16384_B1 | 20.36 (18.74–23.03) | 1.41 | 0.294x (0.28–0.45) | 1.0000000 | 0.0e+00 | 0.997340 | 1.20e-01 | 21.25 |
| a8_pool2_ctx16384_B1 | 11.93 (11.47–13.00) | 0.79 | 0.502x (0.47–0.74) | 1.0000000 | 0.0e+00 | 0.997340 | 1.20e-01 | 21.25 |
| a8_pool4_ctx16384_B1 | 9.00 (8.28–11.15) | 0.61 | 0.665x (0.62–0.98) | 1.0000000 | 0.0e+00 | 0.997340 | 1.20e-01 | 21.25 |
| a8_pool8_ctx16384_B1 | 5.99 (5.11–9.88) | 0.69 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.997340 | 1.20e-01 | 21.25 |
| a8_hb_ctx16384_B1 | 20.57 (19.91–24.24) | 1.29 | 0.291x (0.27–0.41) | 1.0000000 | 0.0e+00 | 0.997340 | 1.20e-01 | 21.25 |
| a8_hbP8_ctx16384_B1 | 6.01 (5.31–8.86) | 0.90 | 0.996x (0.73–1.17) | 1.0000000 | 0.0e+00 | 0.997340 | 1.20e-01 | 21.25 |

### Run 2 (SEED 11211055)

| cfg | attn ms med (min–max) | IQR ms | speedup vs pool8 (per-round range) | cos vs pool8 | max abs vs pool8 | vs f32 cos | max abs vs f32 | KV MiB |
|---|---|---|---|---|---|---|---|---|
| k8v4_pool1_ctx2048_B1 | 2.52 (2.37–3.16) | 0.14 | 0.534x (0.46–0.61) | 1.0000000 | 0.0e+00 | 0.994646 | 1.89e-01 | 2.38 |
| k8v4_pool2_ctx2048_B1 | 1.70 (1.53–2.26) | 0.20 | 0.792x (0.68–0.89) | 1.0000000 | 0.0e+00 | 0.994646 | 1.89e-01 | 2.38 |
| k8v4_pool4_ctx2048_B1 | 1.52 (1.37–2.26) | 0.21 | 0.883x (0.77–0.98) | 1.0000000 | 0.0e+00 | 0.994646 | 1.89e-01 | 2.38 |
| k8v4_pool8_ctx2048_B1 | 1.34 (1.05–1.83) | 0.31 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.994646 | 1.89e-01 | 2.38 |
| k8v4_hb_ctx2048_B1 | 2.65 (2.55–2.84) | 0.09 | 0.508x (0.43–0.59) | 1.0000000 | 0.0e+00 | 0.994646 | 1.89e-01 | 2.38 |
| k8v4_hbP8_ctx2048_B1 | 1.27 (1.04–2.05) | 0.39 | 1.059x (0.85–1.24) | 1.0000000 | 0.0e+00 | 0.994646 | 1.89e-01 | 2.38 |
| khv4h4_pool1_ctx2048_B1 | 2.64 (2.52–3.36) | 0.16 | 0.563x (0.50–0.64) | 1.0000000 | 0.0e+00 | 0.996194 | 1.56e-01 | 2.32 |
| khv4h4_pool2_ctx2048_B1 | 1.84 (1.72–2.51) | 0.23 | 0.810x (0.78–0.87) | 1.0000000 | 0.0e+00 | 0.996194 | 1.56e-01 | 2.32 |
| khv4h4_pool4_ctx2048_B1 | 1.84 (1.55–2.55) | 0.25 | 0.809x (0.76–0.97) | 1.0000000 | 0.0e+00 | 0.996194 | 1.56e-01 | 2.32 |
| khv4h4_pool8_ctx2048_B1 | 1.49 (1.32–2.25) | 0.24 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.996194 | 1.56e-01 | 2.32 |
| khv4h4_hb_ctx2048_B1 | 2.83 (2.71–3.35) | 0.19 | 0.525x (0.50–0.60) | 1.0000000 | 0.0e+00 | 0.996194 | 1.56e-01 | 2.32 |
| khv4h4_hbP8_ctx2048_B1 | 1.73 (1.41–2.76) | 0.27 | 0.859x (0.78–1.01) | 1.0000000 | 0.0e+00 | 0.996194 | 1.56e-01 | 2.32 |
| a8_pool1_ctx2048_B1 | 2.49 (2.40–3.05) | 0.12 | 0.401x (0.36–0.43) | 1.0000000 | 0.0e+00 | 0.998341 | 1.23e-01 | 2.66 |
| a8_pool2_ctx2048_B1 | 1.60 (1.45–2.25) | 0.21 | 0.625x (0.56–0.66) | 1.0000000 | 0.0e+00 | 0.998341 | 1.23e-01 | 2.66 |
| a8_pool4_ctx2048_B1 | 1.29 (1.10–1.78) | 0.18 | 0.775x (0.66–0.88) | 1.0000000 | 0.0e+00 | 0.998341 | 1.23e-01 | 2.66 |
| a8_pool8_ctx2048_B1 | 1.00 (0.84–1.59) | 0.17 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.998341 | 1.23e-01 | 2.66 |
| a8_hb_ctx2048_B1 | 2.33 (2.25–2.89) | 0.13 | 0.429x (0.38–0.46) | 1.0000000 | 0.0e+00 | 0.998341 | 1.23e-01 | 2.66 |
| a8_hbP8_ctx2048_B1 | 1.01 (0.82–1.81) | 0.46 | 0.993x (0.84–1.11) | 1.0000000 | 0.0e+00 | 0.998341 | 1.23e-01 | 2.66 |
| k8v4_pool1_ctx2048_B8 | 21.01 (19.31–24.70) | 1.92 | 0.472x (0.44–0.62) | 1.0000000 | 0.0e+00 | 0.986626 | 4.98e-01 | 19.06 |
| k8v4_pool2_ctx2048_B8 | 12.07 (11.23–27.13) | 0.68 | 0.822x (0.76–0.94) | 1.0000000 | 0.0e+00 | 0.986626 | 4.98e-01 | 19.06 |
| k8v4_pool4_ctx2048_B8 | 9.27 (8.39–12.49) | 1.00 | 1.070x (1.04–1.41) | 1.0000000 | 0.0e+00 | 0.986626 | 4.98e-01 | 19.06 |
| k8v4_pool8_ctx2048_B8 | 9.92 (9.04–21.24) | 1.68 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.986626 | 4.98e-01 | 19.06 |
| k8v4_hb_ctx2048_B8 | 30.48 (27.93–49.67) | 5.84 | 0.325x (0.29–0.45) | 1.0000000 | 0.0e+00 | 0.986626 | 4.98e-01 | 19.06 |
| k8v4_hbP8_ctx2048_B8 | 4.46 (3.85–8.89) | 1.51 | 2.223x (1.34–3.00) | 1.0000000 | 0.0e+00 | 0.986626 | 4.98e-01 | 19.06 |
| khv4h4_pool1_ctx2048_B8 | 22.99 (20.62–29.66) | 2.82 | 0.522x (0.44–0.59) | 1.0000000 | 0.0e+00 | 0.993546 | 2.60e-01 | 18.59 |
| khv4h4_pool2_ctx2048_B8 | 13.88 (12.75–15.25) | 1.17 | 0.864x (0.82–0.94) | 1.0000000 | 0.0e+00 | 0.993546 | 2.60e-01 | 18.59 |
| khv4h4_pool4_ctx2048_B8 | 11.23 (10.04–12.99) | 1.16 | 1.069x (1.03–1.14) | 1.0000000 | 0.0e+00 | 0.993546 | 2.60e-01 | 18.59 |
| khv4h4_pool8_ctx2048_B8 | 12.00 (10.87–14.04) | 0.70 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.993546 | 2.60e-01 | 18.59 |
| khv4h4_hb_ctx2048_B8 | 31.53 (28.82–63.91) | 4.25 | 0.381x (0.33–0.43) | 1.0000000 | 0.0e+00 | 0.993546 | 2.60e-01 | 18.59 |
| khv4h4_hbP8_ctx2048_B8 | 5.43 (4.78–7.98) | 0.97 | 2.211x (1.97–2.31) | 1.0000000 | 0.0e+00 | 0.993546 | 2.60e-01 | 18.59 |
| a8_pool1_ctx2048_B8 | 21.63 (19.42–29.63) | 1.57 | 0.303x (0.28–0.32) | 1.0000000 | 0.0e+00 | 0.992906 | 4.99e-01 | 21.25 |
| a8_pool2_ctx2048_B8 | 11.57 (10.39–14.36) | 1.56 | 0.567x (0.51–0.76) | 1.0000000 | 0.0e+00 | 0.992906 | 4.99e-01 | 21.25 |
| a8_pool4_ctx2048_B8 | 7.44 (6.71–10.14) | 1.02 | 0.882x (0.83–1.02) | 1.0000000 | 0.0e+00 | 0.992906 | 4.99e-01 | 21.25 |
| a8_pool8_ctx2048_B8 | 6.56 (5.96–10.84) | 0.99 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.992906 | 4.99e-01 | 21.25 |
| a8_hb_ctx2048_B8 | 28.14 (25.48–123.22) | 7.72 | 0.233x (0.21–0.30) | 1.0000000 | 0.0e+00 | 0.992906 | 4.99e-01 | 21.25 |
| a8_hbP8_ctx2048_B8 | 4.08 (3.36–7.96) | 0.77 | 1.610x (1.48–2.23) | 1.0000000 | 0.0e+00 | 0.992906 | 4.99e-01 | 21.25 |
| k8v4_pool1_ctx8192_B1 | 11.14 (10.82–12.20) | 0.30 | 0.364x (0.34–0.45) | 1.0000000 | 0.0e+00 | 0.992187 | 3.47e-01 | 8.71 |
| k8v4_pool2_ctx8192_B1 | 7.29 (6.75–8.28) | 0.56 | 0.556x (0.49–0.70) | 1.0000000 | 0.0e+00 | 0.992187 | 3.47e-01 | 8.71 |
| k8v4_pool4_ctx8192_B1 | 5.50 (5.08–6.13) | 0.27 | 0.737x (0.69–0.92) | 1.0000000 | 0.0e+00 | 0.992187 | 3.47e-01 | 8.71 |
| k8v4_pool8_ctx8192_B1 | 4.05 (3.52–5.76) | 1.04 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.992187 | 3.47e-01 | 8.71 |
| k8v4_hb_ctx8192_B1 | 12.22 (11.78–15.65) | 0.62 | 0.332x (0.30–0.41) | 1.0000000 | 0.0e+00 | 0.992187 | 3.47e-01 | 8.71 |
| k8v4_hbP8_ctx8192_B1 | 4.15 (3.43–9.55) | 1.15 | 0.976x (0.86–1.37) | 1.0000000 | 0.0e+00 | 0.992187 | 3.47e-01 | 8.71 |
| khv4h4_pool1_ctx8192_B1 | 12.06 (11.48–13.41) | 0.86 | 0.416x (0.37–0.46) | 1.0000000 | 0.0e+00 | 0.993451 | 3.46e-01 | 8.42 |
| khv4h4_pool2_ctx8192_B1 | 7.80 (7.45–8.83) | 0.47 | 0.642x (0.56–0.74) | 1.0000000 | 0.0e+00 | 0.993451 | 3.46e-01 | 8.42 |
| khv4h4_pool4_ctx8192_B1 | 6.16 (5.73–7.15) | 0.48 | 0.813x (0.73–0.91) | 1.0000000 | 0.0e+00 | 0.993451 | 3.46e-01 | 8.42 |
| khv4h4_pool8_ctx8192_B1 | 5.01 (4.24–6.24) | 0.90 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.993451 | 3.46e-01 | 8.42 |
| khv4h4_hb_ctx8192_B1 | 12.85 (12.31–15.96) | 0.59 | 0.390x (0.34–0.44) | 1.0000000 | 0.0e+00 | 0.993451 | 3.46e-01 | 8.42 |
| khv4h4_hbP8_ctx8192_B1 | 5.39 (4.64–6.55) | 0.92 | 0.931x (0.85–1.07) | 1.0000000 | 0.0e+00 | 0.993451 | 3.46e-01 | 8.42 |
| a8_pool1_ctx8192_B1 | 9.96 (9.37–11.03) | 0.75 | 0.357x (0.31–0.43) | 1.0000000 | 0.0e+00 | 0.996973 | 1.66e-01 | 10.62 |
| a8_pool2_ctx8192_B1 | 6.17 (5.77–8.44) | 0.70 | 0.577x (0.50–0.70) | 1.0000000 | 0.0e+00 | 0.996973 | 1.66e-01 | 10.62 |
| a8_pool4_ctx8192_B1 | 4.48 (4.05–5.56) | 0.46 | 0.794x (0.66–0.97) | 1.0000000 | 0.0e+00 | 0.996973 | 1.66e-01 | 10.62 |
| a8_pool8_ctx8192_B1 | 3.56 (2.96–5.35) | 0.93 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.996973 | 1.66e-01 | 10.62 |
| a8_hb_ctx8192_B1 | 10.40 (9.81–12.76) | 0.62 | 0.342x (0.30–0.41) | 1.0000000 | 0.0e+00 | 0.996973 | 1.66e-01 | 10.62 |
| a8_hbP8_ctx8192_B1 | 3.78 (2.75–4.74) | 1.10 | 0.942x (0.92–1.16) | 1.0000000 | 0.0e+00 | 0.996973 | 1.66e-01 | 10.62 |
| k8v4_pool1_ctx8192_B8 | 93.10 (89.77–110.57) | 2.62 | 0.283x (0.27–0.35) | 1.0000000 | 0.0e+00 | 0.983993 | 6.17e-01 | 69.69 |
| k8v4_pool2_ctx8192_B8 | 50.75 (48.29–59.50) | 2.55 | 0.519x (0.50–0.64) | 1.0000000 | 0.0e+00 | 0.983993 | 6.17e-01 | 69.69 |
| k8v4_pool4_ctx8192_B8 | 30.25 (28.10–42.36) | 1.52 | 0.870x (0.78–1.06) | 1.0000000 | 0.0e+00 | 0.983993 | 6.17e-01 | 69.69 |
| k8v4_pool8_ctx8192_B8 | 26.32 (23.96–35.52) | 5.77 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.983993 | 6.17e-01 | 69.69 |
| k8v4_hb_ctx8192_B8 | 260.48 (123.69–775.02) | 294.85 | 0.101x (0.05–0.23) | 1.0000000 | 0.0e+00 | 0.983993 | 6.17e-01 | 69.69 |
| k8v4_hbP8_ctx8192_B8 | 15.93 (13.73–23.68) | 4.01 | 1.652x (1.23–1.95) | 1.0000000 | 0.0e+00 | 0.983993 | 6.17e-01 | 69.69 |
| khv4h4_pool1_ctx8192_B8 | 98.58 (94.85–105.11) | 4.16 | 0.315x (0.31–0.36) | 1.0000000 | 0.0e+00 | 0.992279 | 2.82e-01 | 67.34 |
| khv4h4_pool2_ctx8192_B8 | 53.62 (51.81–65.18) | 1.52 | 0.580x (0.56–0.65) | 1.0000000 | 0.0e+00 | 0.992279 | 2.82e-01 | 67.34 |
| khv4h4_pool4_ctx8192_B8 | 33.39 (32.23–36.86) | 0.97 | 0.931x (0.87–1.04) | 1.0000000 | 0.0e+00 | 0.992279 | 2.82e-01 | 67.34 |
| khv4h4_pool8_ctx8192_B8 | 31.10 (28.88–44.69) | 1.88 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.992279 | 2.82e-01 | 67.34 |
| khv4h4_hb_ctx8192_B8 | 196.74 (128.52–487.13) | 265.67 | 0.158x (0.07–0.24) | 1.0000000 | 0.0e+00 | 0.992279 | 2.82e-01 | 67.34 |
| khv4h4_hbP8_ctx8192_B8 | 17.71 (15.83–26.51) | 3.83 | 1.756x (1.38–1.88) | 1.0000000 | 0.0e+00 | 0.992279 | 2.82e-01 | 67.34 |
| a8_pool1_ctx8192_B8 | 82.10 (78.59–115.72) | 3.92 | 0.250x (0.24–0.26) | 1.0000000 | 0.0e+00 | 0.990746 | 6.16e-01 | 85.00 |
| a8_pool2_ctx8192_B8 | 43.96 (41.44–52.43) | 1.65 | 0.468x (0.44–0.48) | 1.0000000 | 0.0e+00 | 0.990746 | 6.16e-01 | 85.00 |
| a8_pool4_ctx8192_B8 | 24.97 (23.47–28.15) | 1.14 | 0.823x (0.80–0.85) | 1.0000000 | 0.0e+00 | 0.990746 | 6.16e-01 | 85.00 |
| a8_pool8_ctx8192_B8 | 20.56 (18.97–23.20) | 0.88 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.990746 | 6.16e-01 | 85.00 |
| a8_hb_ctx8192_B8 | 144.43 (108.34–489.84) | 274.95 | 0.142x (0.05–0.19) | 1.0000000 | 0.0e+00 | 0.990746 | 6.16e-01 | 85.00 |
| a8_hbP8_ctx8192_B8 | 13.52 (11.41–18.15) | 3.16 | 1.520x (1.32–1.73) | 1.0000000 | 0.0e+00 | 0.990746 | 6.16e-01 | 85.00 |
| k8v4_pool1_ctx16384_B1 | 23.31 (22.20–28.62) | 0.93 | 0.292x (0.28–0.32) | 1.0000000 | 0.0e+00 | 0.993407 | 2.11e-01 | 17.15 |
| k8v4_pool2_ctx16384_B1 | 14.54 (13.96–18.05) | 0.65 | 0.468x (0.44–0.52) | 1.0000000 | 0.0e+00 | 0.993407 | 2.11e-01 | 17.15 |
| k8v4_pool4_ctx16384_B1 | 10.46 (9.90–12.95) | 0.67 | 0.651x (0.61–0.72) | 1.0000000 | 0.0e+00 | 0.993407 | 2.11e-01 | 17.15 |
| k8v4_pool8_ctx16384_B1 | 6.81 (6.20–11.10) | 0.77 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.993407 | 2.11e-01 | 17.15 |
| k8v4_hb_ctx16384_B1 | 26.15 (24.59–31.41) | 1.98 | 0.260x (0.24–0.30) | 1.0000000 | 0.0e+00 | 0.993407 | 2.11e-01 | 17.15 |
| k8v4_hbP8_ctx16384_B1 | 6.85 (6.24–8.74) | 0.69 | 0.995x (0.90–1.11) | 1.0000000 | 0.0e+00 | 0.993407 | 2.11e-01 | 17.15 |
| khv4h4_pool1_ctx16384_B1 | 25.29 (24.45–27.97) | 1.13 | 0.318x (0.29–0.34) | 1.0000000 | 0.0e+00 | 0.994811 | 1.74e-01 | 16.54 |
| khv4h4_pool2_ctx16384_B1 | 15.25 (14.68–20.77) | 0.88 | 0.527x (0.46–0.57) | 1.0000000 | 0.0e+00 | 0.994811 | 1.74e-01 | 16.54 |
| khv4h4_pool4_ctx16384_B1 | 11.61 (11.07–13.76) | 0.54 | 0.692x (0.66–0.74) | 1.0000000 | 0.0e+00 | 0.994811 | 1.74e-01 | 16.54 |
| khv4h4_pool8_ctx16384_B1 | 8.03 (7.48–9.40) | 0.66 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.994811 | 1.74e-01 | 16.54 |
| khv4h4_hb_ctx16384_B1 | 26.99 (26.33–36.87) | 2.04 | 0.298x (0.29–0.32) | 1.0000000 | 0.0e+00 | 0.994811 | 1.74e-01 | 16.54 |
| khv4h4_hbP8_ctx16384_B1 | 8.71 (8.20–10.59) | 0.65 | 0.923x (0.90–0.98) | 1.0000000 | 0.0e+00 | 0.994811 | 1.74e-01 | 16.54 |
| a8_pool1_ctx16384_B1 | 20.69 (19.30–22.36) | 1.24 | 0.277x (0.26–0.31) | 1.0000000 | 0.0e+00 | 0.998406 | 1.82e-01 | 21.25 |
| a8_pool2_ctx16384_B1 | 12.23 (11.60–13.32) | 0.58 | 0.469x (0.45–0.53) | 1.0000000 | 0.0e+00 | 0.998406 | 1.82e-01 | 21.25 |
| a8_pool4_ctx16384_B1 | 8.66 (7.76–11.29) | 0.53 | 0.662x (0.64–0.72) | 1.0000000 | 0.0e+00 | 0.998406 | 1.82e-01 | 21.25 |
| a8_pool8_ctx16384_B1 | 5.73 (5.23–7.14) | 0.85 | 1.000x (1.00–1.00) | 1.0000000 | 0.0e+00 | 0.998406 | 1.82e-01 | 21.25 |
| a8_hb_ctx16384_B1 | 20.60 (19.85–24.31) | 1.28 | 0.278x (0.26–0.31) | 1.0000000 | 0.0e+00 | 0.998406 | 1.82e-01 | 21.25 |
| a8_hbP8_ctx16384_B1 | 5.79 (5.06–8.24) | 0.57 | 0.990x (0.94–1.11) | 1.0000000 | 0.0e+00 | 0.998406 | 1.82e-01 | 21.25 |

## Verdict

**viable** (B = 8, grouped head-batched hbP8). Median speedup vs the 40-task pool is 1.65–1.90x for K8V4 / khv4h4 at ctx 8192 and 1.99–2.26x at ctx 2048 on both seeds, bit-exact (max abs diff 0.0e+00), same KV bytes. The literal one-call-over-40-heads form is **not viable** (0.06–0.38x at B = 8): it gives up the pool's real ~3x thread parallelism and is memory-bound on 40 MiB tiles. B = 1 is unchanged (~1x). For i2sd: keep K8V4 and, at B ≥ 8, run attention as 8 head-group tasks over a stacked (stream·head, n, ...) cache instead of one task per (stream, head).

## This wave's pair files

No quant pair reports reached chat this wave (the 5:30 quant derivation lane is paused), so there are no pair files to link. Previous note: [[2026-11-21_1055_flash-decoding-split-k-history-partition-across-8-workers-merge-partial-m-l-acc-k8v4-khv4h4-a8-kv-b1-b8-ctx-2048-16384-vs-per-stream-head-tasks]].

## Script

hb.py:

    #!/usr/bin/env python3
    """Head-batched attention for i2sd decode, 14:55 11/21. Derived from iq.py (02:55 11/21, loaded without main()).
    Baseline (05:55 / 10:55 11/21 path): iq.attn, one pool task per (stream, KV head), online softmax over TK tiles.
    Thread scaling: the same baseline on a 1 / 2 / 4 / 8-thread pool (iq.POOL swapped).
    hb:  no pool; per TK tile ONE batched np.matmul over all B x 5 KV heads: (BH, 4, 128) @ (BH, 128, T) scores and one
         batched (BH, 4, T) @ (BH, T, 128) P @ V; K/V dequant done per tile for all heads at once (codes/scales stacked
         (BH, n, ...) once at cache build, same bytes); online softmax across tiles on (BH, 4, 1) m / l.
    hbP: hb with the BH heads cut into 8 contiguous groups, one pool task per group (bigger numpy ops per task).
    Formats: k8v4 (i2sd default), khv4h4, a8. Grid: ctx {2048, 8192} x B {1, 8}, plus ctx 16384 at B = 1.
    Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 hb.py out.json"""
    import json, sys, time, os, types
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    src = open("/workspace/bench0255_1121/iq.py").read().rstrip()
    assert src.endswith("main()"); iq = types.ModuleType("iq"); exec(compile(src[:-6], "iq.py", "exec"), iq.__dict__)
    KVH, G, HD, SC, WIN = iq.KVH, iq.G, iq.HD, iq.SC, iq.WIN
    FMTS = ("k8v4", "khv4h4", "a8")
    GRID = [(2048, 1), (2048, 8), (8192, 1), (8192, 8), (16384, 1)]
    THREADS = (1, 2, 4, 8); POOLS = {w: (iq.POOL if w == iq.W else ThreadPoolExecutor(w)) for w in THREADS}
    TK = 2048; ROUNDS, REPS = 7, 5
    class BKV:
        """All streams' codes/scales stacked (B*KVH, n, ...) for one format; built once, not timed."""
        def __init__(s, kvs, fmt):
            st = lambda nm: np.ascontiguousarray(np.concatenate([getattr(kv, nm) for kv in kvs], 0))
            s.n = kvs[0].n; s.fmt = fmt; s.BH = len(kvs) * KVH
            if fmt == "khv4h4":
                s.k, s.ks = st("kz48h"), st("kzs48h"); s.khot = st("khot4"); hidx = st("hidx4")   # (BH, 4)
                s.hidx = hidx[:, None, :].repeat(G, 1)                                           # (BH, G, 4) for take_along_axis
                s.vo, s.vos = st("v4h"), st("v4hs"); s.vw, s.vws = st("v8h"), st("vs8h")
            else:
                s.k, s.ks = st("k8"), st("ks8"); s.vw, s.vws = st("v8"), st("vs8")
                if fmt == "k8v4": s.vo, s.vos = st("v4"), st("v4s")
        def S(s, Q, a, b, h0, h1):
            K = iq.dq8(s.k[h0:h1, a:b], s.ks[h0:h1, a:b])                                     # (bh, T, HD) f32
            Sx = np.matmul(Q, K.transpose(0, 2, 1))                                            # (bh, G, T)
            if s.fmt == "khv4h4":
                Qh = np.take_along_axis(Q, s.hidx[h0:h1], 2)                                   # (bh, G, 4)
                Sx += np.matmul(Qh, s.khot[h0:h1, a:b].astype(np.float32).transpose(0, 2, 1))
            return Sx
        def V(s, a, b, old, h0, h1):
            if old and s.fmt == "k8v4": return iq.dq4(s.vo[h0:h1, a:b], s.vos[h0:h1, a:b])
            if old and s.fmt == "khv4h4": return iq.dq4(s.vo[h0:h1, a:b], s.vos[h0:h1, a:b].astype(np.float32))
            return iq.dq8(s.vw[h0:h1, a:b], s.vws[h0:h1, a:b])
        def tiles(s):
            if s.fmt in iq.A8FAM: return [(a, min(a + TK, s.n), True) for a in range(0, s.n, TK)]
            wb = max(0, s.n - WIN)
            return [(a, min(a + TK, wb), True) for a in range(0, wb, TK)] + [(a, min(a + WIN, s.n), False) for a in range(wb, s.n, WIN)]
    def hb_range(bk, Q, h0, h1):
        bh = h1 - h0; m = np.full((bh, G, 1), -np.inf, np.float32); l = np.zeros((bh, G, 1), np.float32); acc = np.zeros((bh, G, HD), np.float32)
        for a, b, old in bk.tiles():
            Sx = bk.S(Q, a, b, h0, h1); mn = np.maximum(m, Sx.max(2, keepdims=True)); al = np.exp(m - mn)
            Sx -= mn; np.exp(Sx, out=Sx); l *= al; l += Sx.sum(2, keepdims=True); acc *= al; acc += np.matmul(Sx, bk.V(a, b, old, h0, h1)); m = mn
        return acc / l
    def hb(bk, Qb): return hb_range(bk, Qb, 0, bk.BH)
    def hbP(bk, Qb, W=8):
        cuts = [round(i * bk.BH / W) for i in range(W + 1)]; cuts = sorted(set(cuts))
        futs = [iq.POOL.submit(hb_range, bk, Qb[h0:h1], h0, h1) for h0, h1 in zip(cuts[:-1], cuts[1:])]
        return np.concatenate([f.result() for f in futs], 0)
    def base_w(kvs, q, fmt, w):
        iq.POOL = POOLS[w]
        try: return iq.attn(kvs, q, fmt, TK)
        finally: iq.POOL = POOLS[8]
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    def main():
        rng = iq.rng; out = {"seed": iq.SEED, "cfg": {}}
        print(f"seed {iq.SEED} | TK {TK} | rounds {ROUNDS} x reps {REPS} | pools {THREADS}")
        print(f"{'cfg':28s} {'attn ms med (min-max)':>24s} {'iqr':>6s}  vs-pool8  vs-base-cos  vs-base-maxabs  vs-f32-cos  KV MiB")
        for n in sorted({g[0] for g in GRID}):
            Bmax = max(B for m, B in GRID if m == n)
            kvs8 = [iq.KV(*iq.kvhist(n)) for _ in range(Bmax)]
            for B in [B for m, B in GRID if m == n]:
                kvs = kvs8[:B]; q = (rng.standard_normal((B, KVH, G, HD)).astype(np.float32) * 2.0) * SC
                Qb = np.ascontiguousarray(q.reshape(B * KVH, G, HD))
                ref = np.stack([np.stack([iq.ref(kvs[s], q[s, h], h) for h in range(KVH)]) for s in range(B)]).reshape(B * KVH, G, HD)
                cfgs = {}
                for fmt in FMTS:
                    base = iq.attn(kvs, q, fmt, TK).reshape(B * KVH, G, HD); bk = BKV(kvs, fmt)
                    variants = [(f"pool{w}", (lambda fmt=fmt, w=w: base_w(kvs, q, fmt, w))) for w in THREADS]
                    variants += [("hb", (lambda bk=bk: hb(bk, Qb))), ("hbP8", (lambda bk=bk: hbP(bk, Qb)))]
                    for nm, f in variants:
                        o = np.asarray(f()).reshape(B * KVH, G, HD)
                        cfgs[(fmt, nm)] = dict(f=f, t=[], cb=min(cos(o[i], base[i]) for i in range(B * KVH)), mab=float(np.abs(o - base).max()),
                                               cr=min(cos(o[i], ref[i]) for i in range(B * KVH)), mar=float(np.abs(o - ref).max()), kv=B * kvs[0].bytes(fmt) / 2**20)
                order = list(cfgs)
                for _ in range(2):
                    for k in order: cfgs[k]["f"]()
                for rd in range(ROUNDS):
                    for k in order[rd % len(order):] + order[:rd % len(order)]:
                        for _ in range(REPS):
                            t0 = time.perf_counter(); cfgs[k]["f"](); cfgs[k]["t"].append((time.perf_counter() - t0) * 1e3)
                for (fmt, nm), c in cfgs.items():
                    T = np.array(c["t"]); med = float(np.median(T)); b8 = float(np.median(cfgs[(fmt, "pool8")]["t"]))
                    # per-round speedup spread vs pool8 (round medians)
                    rr = [float(np.median(T[i * REPS:(i + 1) * REPS])) for i in range(ROUNDS)]
                    rb = [float(np.median(np.array(cfgs[(fmt, "pool8")]["t"][i * REPS:(i + 1) * REPS]))) for i in range(ROUNDS)]
                    sp_r = [x / y for x, y in zip(rb, rr)]
                    key = f"{fmt}_{nm}_ctx{n}_B{B}"
                    d = dict(fmt=fmt, var=nm, ctx=n, B=B, attn=med, lo=float(T.min()), hi=float(T.max()), iqr=float(np.percentile(T, 75) - np.percentile(T, 25)),
                             speedup=b8 / med, sp_lo=min(sp_r), sp_hi=max(sp_r), cos_base=c["cb"], maxabs_base=c["mab"], cos_f32=c["cr"], maxabs_f32=c["mar"], kv_mib=c["kv"], n=len(T))
                    out["cfg"][key] = d
                    print(f"{key:28s} {med:8.2f} ({d['lo']:.2f}-{d['hi']:.2f}) {d['iqr']:6.2f}  {d['speedup']:6.3f}x ({d['sp_lo']:.2f}-{d['sp_hi']:.2f})  {d['cos_base']:.7f}  {d['maxabs_base']:.2e}  {d['cos_f32']:.6f}  {d['kv_mib']:6.2f}", flush=True)
            del kvs8
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Next question

hbP8 wins by cutting the task count to the thread count while keeping per-task tiles cache-sized. At B = 8 / ctx 8192 (and B = 16 / 32, where the pool would have 80 / 160 one-head tasks), what are the best group count (4 / 8 / 16 / 32 tasks) and tile size TK (512 / 1024 / 2048) for grouped head-batched K8V4 attention? Does the gain over the one-task-per-head pool grow with B (target ≥ 1.5x over today's hbP8 at B = 32)?

close: c63a11c62be2571f
