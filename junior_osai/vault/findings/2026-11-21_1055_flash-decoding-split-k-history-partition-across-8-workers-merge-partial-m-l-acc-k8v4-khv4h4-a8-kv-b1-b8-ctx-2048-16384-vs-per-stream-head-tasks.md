---
local: true
date: 2026-11-21
wave: "2026-11-21 10:55 (ran ~10:45–11:05 MT; no pair reports reached chat this wave, quant derivation lane paused)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, attention, flash-decoding, split-k, kv-cache, k8v4, khv4h4, a8-g64, threads, numpy, not-viable]
---

# Flash-decoding split-K does not speed up i2sd attention in numpy: best 1.04x, mostly 0.3–1.0x, because each extra task costs ~88–91 µs of serialized overhead

The 05:55 note found attention is 30–68% of each B = 8 decode step and asked whether partitioning each stream's history across the 8 workers (flash-decoding: per-chunk partial (m, l, acc), merged by log-sum-exp) would lift attention throughput ≥ 1.5x. It does not:
- **Best split speedup anywhere in the grid is 1.045x** (two runs, 15 format × ctx × B cells). No cell reaches 1.5x; most splits are slower.
- **B = 1** (5 tasks unsplit, 3 idle threads): at ctx 8192–16384, S = 2–8 lands at **0.61–1.05x**; S = 16 drops to 0.37–0.66x. At ctx 2048 every split is slower (0.17–0.90x).
- **B = 8** (40 tasks unsplit): S = 2 is **0.57–0.93x** and S = 4 is **0.35–0.88x** at both ctx 2048 and 8192.
- **Exactness holds**: split vs unsplit cos ≥ 0.9999999 in every cell, so merge math is not the problem; vs-f32 cos is unchanged per format (K8V4 0.989–0.994 at ctx 8192 on this query draw, khv4h4 0.993–0.996, a8 0.995–0.999).
- **Why**: attention time grows roughly linearly with the task count (a8, B = 1, ctx 2048: slope ≈ 88 / 91 µs per extra task). Per-tile numpy work at G = 4 query rows is small and mostly holds the GIL, so the 8-thread pool runs these tasks close to serially. The 05:55 "3 idle threads at B = 1" were not doing real parallel work to begin with.

Verdict: **not-viable**. In this numpy/thread-pool path, split-K only adds Python-level task and merge overhead. K8V4 with one task per (stream, KV head) stays the i2sd attention path, and the end-to-end step share from 05:55 is unchanged (projection below). A native kernel (C/Metal) with real parallel workers could behave differently, but that is outside this numpy-only bench.

## Question

[[2026-11-21_0555_full-bitnetd-decode-step-attention-share-u32-lut-2-bit-weights-with-khv4h4-k8v4-a8-kv-vs-int8-a8-b1-b8-ctx-1024-16384]] showed attention dominates the B = 8 decode step past ctx ≈ 3k (0.68 of the step at ctx 8192 with khv4h4, 0.64 with K8V4). Its next question:
- At B = 8 / ctx 8192 (and B = 1, where only 5 of 8 threads get a task), does a split-K partition of each (stream, KV head) history into S chunks, each a separate pool task, merged by rescaling the partial (m, l, acc), raise attention throughput ≥ 1.5x for K8V4 and khv4h4?
- What does that do to the end-to-end step?

## Method

- 'split.py' (below) loads '/workspace/bench0255_1121/iq.py' without its 'main()' (same KV formats, synthetic KV generator with planted outlier K channels, 'iq.tile' online-softmax tile, 8-thread 'iq.POOL').
- Baseline S = 1 is the unchanged 05:55 path 'iq.attn' (one task per (stream, KV head), TK = 2048 tiles, A8 1024 window).
- Split S ∈ {2, 4, 8, 16} at B = 1 and {2, 4} at B = 8: the history [0, n) is cut into S contiguous chunks, with the old/window boundary always a cut (if that would make S + 1 chunks, the smallest non-boundary neighbours merge, so chunks can be uneven). Each chunk is its own pool task; partials are merged with m = max(m_i), w_i = exp(m_i − m), out = Σ w_i acc_i / Σ w_i l_i.
- Formats: k8v4 (i2sd default), khv4h4, a8. Grid: ctx {2048, 8192} × B {1, 8}, plus ctx 16384 at B = 1. 5 KV heads × 4 query heads, HD 128.
- Timing: attention call only, 2 warmup passes over all configs, then 7 rounds with config order rotated each round, 5 reps per config per round, so n = 35 per config. Median (min–max) and IQR reported.
- Error: min over streams of cos(split, unsplit) and cos vs exact f32 softmax attention over the f32 history ('iq.ref').
- Two runs: SEED 11211055 (new) and SEED 11210555 (the 05:55 seed). Queries are fresh standard-normal draws × 2 × 1/√HD.
- Python 3 + numpy, OPENBLAS_NUM_THREADS=1, 8 cores. No network, nothing bound, no installs.

Command:

    cd /workspace/bench1055_1121
    OPENBLAS_NUM_THREADS=1 SEED=11211055 python3 split.py results_run1.json > run1.txt
    OPENBLAS_NUM_THREADS=1 SEED=11210555 python3 split.py results_run2.json > run2.txt

## Results

### Summary (both runs)

| fmt | ctx | B | unsplit attn ms (run1 / run2) | best S (run1 / run2) | best split speedup (run1 / run2) | S = 8 speedup (B=1) or S = 4 (B=8) | split vs unsplit cos (min) | vs f32 cos |
|---|---|---|---|---|---|---|---|---|
| k8v4 | 2048 | 1 | 1.52 / 1.56 | 2 / 2 | 0.814x / 0.846x | 0.308x / 0.312x | 1.0000000 | 0.994699 |
| khv4h4 | 2048 | 1 | 1.87 / 1.88 | 2 / 2 | 0.896x / 0.854x | 0.316x / 0.308x | 1.0000000 | 0.995228 |
| a8 | 2048 | 1 | 1.33 / 1.39 | 2 / 2 | 0.750x / 0.807x | 0.293x / 0.295x | 1.0000000 | 0.999057 |
| k8v4 | 2048 | 8 | 10.80 / 10.76 | 2 / 2 | 0.816x / 0.844x | 0.498x / 0.491x | 1.0000000 | 0.992416 |
| khv4h4 | 2048 | 8 | 13.57 / 13.39 | 2 / 2 | 0.896x / 0.858x | 0.510x / 0.487x | 1.0000000 | 0.995329 |
| a8 | 2048 | 8 | 7.34 / 7.25 | 2 / 2 | 0.596x / 0.573x | 0.373x / 0.351x | 1.0000000 | 0.996301 |
| k8v4 | 8192 | 1 | 4.44 / 4.88 | 2 / 2 | 0.940x / 0.985x | 0.643x / 0.727x | 1.0000000 | 0.993369 |
| khv4h4 | 8192 | 1 | 5.20 / 5.28 | 2 / 2 | 0.979x / 0.976x | 0.653x / 0.661x | 1.0000000 | 0.993836 |
| a8 | 8192 | 1 | 3.82 / 3.71 | 4 / 4 | 0.994x / 0.951x | 0.614x / 0.614x | 1.0000000 | 0.998464 |
| k8v4 | 8192 | 8 | 28.38 / 34.52 | 2 / 2 | 0.914x / 0.928x | 0.858x / 0.879x | 1.0000000 | 0.989290 |
| khv4h4 | 8192 | 8 | 34.60 / 40.12 | 2 / 2 | 0.933x / 0.901x | 0.884x / 0.841x | 1.0000000 | 0.993365 |
| a8 | 8192 | 8 | 21.74 / 26.43 | 2 / 2 | 0.882x / 0.908x | 0.795x / 0.810x | 1.0000000 | 0.995226 |
| k8v4 | 16384 | 1 | 8.66 / 7.27 | 2 / 2 | 1.045x / 0.934x | 0.981x / 0.772x | 1.0000000 | 0.993642 |
| khv4h4 | 16384 | 1 | 9.43 / 8.82 | 2 / 2 | 0.994x / 0.940x | 0.901x / 0.772x | 1.0000000 | 0.994580 |
| a8 | 16384 | 1 | 7.31 / 6.80 | 2 / 4 | 1.019x / 0.937x | 0.994x / 0.869x | 1.0000000 | 0.998619 |

### End-to-end projection (not re-timed)

Applies the smaller of this run's two best-split speedups to the 05:55 run1 attention time; the weights and the rest of the step stay as measured at 05:55. All of these min-of-runs best-split speedups are below 1x, so these rows show the step if split-K were forced on: 0.81–0.99x. With the unsplit path kept, the step stays as measured at 05:55.

| cfg (05:55 run1 step) | 05:55 step ms | 05:55 attn ms | best split speedup here (min of runs) | projected step ms | projected step speedup |
|---|---|---|---|---|---|
| u32+k8v4_ctx2048_B1 | 22.91 | 1.84 | 0.814x | 23.33 | 0.982x |
| u32+khv4h4_ctx2048_B1 | 23.65 | 2.22 | 0.854x | 24.03 | 0.984x |
| u32+a8_ctx2048_B1 | 22.27 | 1.74 | 0.750x | 22.85 | 0.975x |
| u32+k8v4_ctx2048_B8 | 26.05 | 10.83 | 0.816x | 28.50 | 0.914x |
| u32+khv4h4_ctx2048_B8 | 28.88 | 12.94 | 0.858x | 31.02 | 0.931x |
| u32+a8_ctx2048_B8 | 22.65 | 7.31 | 0.573x | 28.10 | 0.806x |
| u32+k8v4_ctx8192_B1 | 27.05 | 5.18 | 0.940x | 27.38 | 0.988x |
| u32+khv4h4_ctx8192_B1 | 27.32 | 5.69 | 0.976x | 27.46 | 0.995x |
| u32+a8_ctx8192_B1 | 25.86 | 4.13 | 0.951x | 26.07 | 0.992x |
| u32+k8v4_ctx8192_B8 | 44.33 | 27.75 | 0.914x | 46.96 | 0.944x |
| u32+khv4h4_ctx8192_B8 | 49.12 | 33.50 | 0.901x | 52.80 | 0.930x |
| u32+a8_ctx8192_B8 | 40.35 | 23.26 | 0.882x | 43.47 | 0.928x |
| u32+k8v4_ctx16384_B1 | 33.34 | 10.96 | 0.934x | 34.11 | 0.977x |
| u32+khv4h4_ctx16384_B1 | 33.45 | 10.67 | 0.940x | 34.13 | 0.980x |
| u32+a8_ctx16384_B1 | 30.49 | 7.76 | 0.937x | 31.01 | 0.983x |

### Run 1 (SEED 11211055)

| cfg | attn ms med (min–max) | IQR ms | tasks | speedup vs S=1 | split vs unsplit cos | vs f32 cos | KV MiB |
|---|---|---|---|---|---|---|---|
| k8v4_S1_ctx2048_B1 | 1.52 (1.32–1.92) | 0.25 | 5 | 1.000x | 1.0000000 | 0.997245 | 2.38 |
| k8v4_S2_ctx2048_B1 | 1.86 (1.56–2.16) | 0.17 | 10 | 0.814x | 1.0000000 | 0.997245 | 2.38 |
| k8v4_S4_ctx2048_B1 | 2.80 (2.56–3.86) | 0.16 | 20 | 0.541x | 1.0000000 | 0.997245 | 2.38 |
| k8v4_S8_ctx2048_B1 | 4.92 (4.50–6.48) | 0.43 | 40 | 0.308x | 1.0000000 | 0.997245 | 2.38 |
| k8v4_S16_ctx2048_B1 | 8.46 (7.63–11.04) | 0.90 | 80 | 0.179x | 1.0000000 | 0.997245 | 2.38 |
| khv4h4_S1_ctx2048_B1 | 1.87 (1.52–2.30) | 0.22 | 5 | 1.000x | 1.0000000 | 0.998048 | 2.32 |
| khv4h4_S2_ctx2048_B1 | 2.08 (1.85–2.34) | 0.22 | 10 | 0.896x | 1.0000000 | 0.998048 | 2.32 |
| khv4h4_S4_ctx2048_B1 | 3.40 (3.04–3.91) | 0.21 | 20 | 0.550x | 1.0000000 | 0.998048 | 2.32 |
| khv4h4_S8_ctx2048_B1 | 5.92 (5.37–9.11) | 0.59 | 40 | 0.316x | 1.0000000 | 0.998048 | 2.32 |
| khv4h4_S16_ctx2048_B1 | 10.09 (9.35–11.22) | 0.61 | 80 | 0.185x | 1.0000000 | 0.998048 | 2.32 |
| a8_S1_ctx2048_B1 | 1.33 (0.94–1.80) | 0.28 | 5 | 1.000x | 1.0000000 | 0.999131 | 2.66 |
| a8_S2_ctx2048_B1 | 1.78 (1.50–2.12) | 0.25 | 10 | 0.750x | 1.0000000 | 0.999131 | 2.66 |
| a8_S4_ctx2048_B1 | 2.78 (2.44–3.35) | 0.27 | 20 | 0.480x | 1.0000000 | 0.999131 | 2.66 |
| a8_S8_ctx2048_B1 | 4.55 (4.10–5.75) | 0.31 | 40 | 0.293x | 1.0000000 | 0.999132 | 2.66 |
| a8_S16_ctx2048_B1 | 7.91 (7.14–8.83) | 0.53 | 80 | 0.169x | 1.0000000 | 0.999132 | 2.66 |
| k8v4_S1_ctx2048_B8 | 10.80 (9.89–12.45) | 0.87 | 40 | 1.000x | 1.0000000 | 0.992675 | 19.06 |
| k8v4_S2_ctx2048_B8 | 13.23 (12.17–22.60) | 0.91 | 80 | 0.816x | 1.0000000 | 0.992675 | 19.06 |
| k8v4_S4_ctx2048_B8 | 21.69 (19.91–31.06) | 1.78 | 160 | 0.498x | 1.0000000 | 0.992675 | 19.06 |
| khv4h4_S1_ctx2048_B8 | 13.57 (12.21–16.75) | 1.16 | 40 | 1.000x | 1.0000000 | 0.995329 | 18.59 |
| khv4h4_S2_ctx2048_B8 | 15.14 (14.45–18.41) | 0.81 | 80 | 0.896x | 1.0000000 | 0.995329 | 18.59 |
| khv4h4_S4_ctx2048_B8 | 26.60 (23.39–31.23) | 1.73 | 160 | 0.510x | 1.0000000 | 0.995329 | 18.59 |
| a8_S1_ctx2048_B8 | 7.34 (6.45–8.65) | 0.67 | 40 | 1.000x | 1.0000000 | 0.997051 | 21.25 |
| a8_S2_ctx2048_B8 | 12.31 (10.99–15.51) | 1.33 | 80 | 0.596x | 1.0000000 | 0.997051 | 21.25 |
| a8_S4_ctx2048_B8 | 19.68 (18.67–24.00) | 1.48 | 160 | 0.373x | 1.0000000 | 0.997051 | 21.25 |
| k8v4_S1_ctx8192_B1 | 4.44 (3.93–6.82) | 0.66 | 5 | 1.000x | 1.0000000 | 0.993369 | 8.71 |
| k8v4_S2_ctx8192_B1 | 4.72 (4.10–6.47) | 1.09 | 10 | 0.940x | 1.0000000 | 0.993369 | 8.71 |
| k8v4_S4_ctx8192_B1 | 5.08 (4.21–6.99) | 1.45 | 20 | 0.873x | 1.0000000 | 0.993369 | 8.71 |
| k8v4_S8_ctx8192_B1 | 6.90 (5.99–9.38) | 0.54 | 40 | 0.643x | 1.0000000 | 0.993369 | 8.71 |
| k8v4_S16_ctx8192_B1 | 11.28 (9.49–21.06) | 1.97 | 80 | 0.393x | 1.0000000 | 0.993369 | 8.71 |
| khv4h4_S1_ctx8192_B1 | 5.20 (4.18–7.20) | 1.19 | 5 | 1.000x | 1.0000000 | 0.993836 | 8.42 |
| khv4h4_S2_ctx8192_B1 | 5.31 (4.74–7.17) | 0.87 | 10 | 0.979x | 1.0000000 | 0.993836 | 8.42 |
| khv4h4_S4_ctx8192_B1 | 5.79 (5.00–7.64) | 0.98 | 20 | 0.897x | 1.0000000 | 0.993836 | 8.42 |
| khv4h4_S8_ctx8192_B1 | 7.96 (7.10–10.28) | 1.36 | 40 | 0.653x | 1.0000000 | 0.993836 | 8.42 |
| khv4h4_S16_ctx8192_B1 | 13.30 (11.93–18.20) | 1.13 | 80 | 0.391x | 1.0000000 | 0.993836 | 8.42 |
| a8_S1_ctx8192_B1 | 3.82 (3.18–6.30) | 0.62 | 5 | 1.000x | 1.0000000 | 0.999217 | 10.62 |
| a8_S2_ctx8192_B1 | 4.00 (3.44–5.80) | 0.85 | 10 | 0.955x | 1.0000000 | 0.999217 | 10.62 |
| a8_S4_ctx8192_B1 | 3.84 (3.41–5.97) | 0.61 | 20 | 0.994x | 1.0000000 | 0.999217 | 10.62 |
| a8_S8_ctx8192_B1 | 6.22 (5.35–7.39) | 1.00 | 40 | 0.614x | 1.0000000 | 0.999217 | 10.62 |
| a8_S16_ctx8192_B1 | 10.20 (9.03–11.86) | 1.17 | 80 | 0.374x | 1.0000000 | 0.999217 | 10.62 |
| k8v4_S1_ctx8192_B8 | 28.38 (25.56–34.95) | 2.61 | 40 | 1.000x | 1.0000000 | 0.990494 | 69.69 |
| k8v4_S2_ctx8192_B8 | 31.06 (28.50–40.79) | 2.67 | 80 | 0.914x | 1.0000000 | 0.990494 | 69.69 |
| k8v4_S4_ctx8192_B8 | 33.09 (30.46–36.81) | 3.09 | 160 | 0.858x | 1.0000000 | 0.990494 | 69.69 |
| khv4h4_S1_ctx8192_B8 | 34.60 (31.16–39.08) | 2.78 | 40 | 1.000x | 1.0000000 | 0.993840 | 67.34 |
| khv4h4_S2_ctx8192_B8 | 37.08 (34.36–41.13) | 2.46 | 80 | 0.933x | 1.0000000 | 0.993840 | 67.34 |
| khv4h4_S4_ctx8192_B8 | 39.14 (36.20–44.94) | 3.09 | 160 | 0.884x | 1.0000000 | 0.993840 | 67.34 |
| a8_S1_ctx8192_B8 | 21.74 (19.80–25.11) | 1.91 | 40 | 1.000x | 1.0000000 | 0.996322 | 85.00 |
| a8_S2_ctx8192_B8 | 24.65 (23.17–28.54) | 1.40 | 80 | 0.882x | 1.0000000 | 0.996322 | 85.00 |
| a8_S4_ctx8192_B8 | 27.33 (25.49–28.85) | 1.36 | 160 | 0.795x | 1.0000000 | 0.996322 | 85.00 |
| k8v4_S1_ctx16384_B1 | 8.66 (6.82–11.36) | 1.59 | 5 | 1.000x | 1.0000000 | 0.994672 | 17.15 |
| k8v4_S2_ctx16384_B1 | 8.29 (7.02–11.20) | 1.07 | 10 | 1.045x | 1.0000000 | 0.994672 | 17.15 |
| k8v4_S4_ctx16384_B1 | 8.84 (8.10–9.74) | 0.71 | 20 | 0.979x | 1.0000000 | 0.994672 | 17.15 |
| k8v4_S8_ctx16384_B1 | 8.83 (7.78–9.81) | 0.58 | 40 | 0.981x | 1.0000000 | 0.994672 | 17.15 |
| k8v4_S16_ctx16384_B1 | 13.09 (11.97–15.79) | 0.91 | 80 | 0.661x | 1.0000000 | 0.994672 | 17.15 |
| khv4h4_S1_ctx16384_B1 | 9.43 (8.36–12.25) | 1.92 | 5 | 1.000x | 1.0000000 | 0.995963 | 16.54 |
| khv4h4_S2_ctx16384_B1 | 9.49 (8.29–12.52) | 1.47 | 10 | 0.994x | 1.0000000 | 0.995963 | 16.54 |
| khv4h4_S4_ctx16384_B1 | 10.12 (9.26–11.67) | 0.93 | 20 | 0.932x | 1.0000000 | 0.995963 | 16.54 |
| khv4h4_S8_ctx16384_B1 | 10.47 (9.09–13.32) | 1.46 | 40 | 0.901x | 1.0000000 | 0.995963 | 16.54 |
| khv4h4_S16_ctx16384_B1 | 15.90 (14.49–19.21) | 0.97 | 80 | 0.593x | 1.0000000 | 0.995963 | 16.54 |
| a8_S1_ctx16384_B1 | 7.31 (5.97–10.41) | 1.61 | 5 | 1.000x | 1.0000000 | 0.998619 | 21.25 |
| a8_S2_ctx16384_B1 | 7.18 (6.56–9.37) | 0.72 | 10 | 1.019x | 1.0000000 | 0.998619 | 21.25 |
| a8_S4_ctx16384_B1 | 7.21 (6.10–9.76) | 1.18 | 20 | 1.014x | 1.0000000 | 0.998619 | 21.25 |
| a8_S8_ctx16384_B1 | 7.36 (6.50–9.06) | 0.75 | 40 | 0.994x | 1.0000000 | 0.998619 | 21.25 |
| a8_S16_ctx16384_B1 | 11.66 (10.58–13.82) | 1.16 | 80 | 0.627x | 1.0000000 | 0.998619 | 21.25 |

### Run 2 (SEED 11210555)

| cfg | attn ms med (min–max) | IQR ms | tasks | speedup vs S=1 | split vs unsplit cos | vs f32 cos | KV MiB |
|---|---|---|---|---|---|---|---|
| k8v4_S1_ctx2048_B1 | 1.56 (1.27–2.51) | 0.28 | 5 | 1.000x | 1.0000000 | 0.994699 | 2.38 |
| k8v4_S2_ctx2048_B1 | 1.85 (1.61–2.32) | 0.25 | 10 | 0.846x | 1.0000000 | 0.994699 | 2.38 |
| k8v4_S4_ctx2048_B1 | 2.94 (2.61–6.15) | 0.26 | 20 | 0.530x | 1.0000000 | 0.994699 | 2.38 |
| k8v4_S8_ctx2048_B1 | 5.01 (4.42–5.92) | 0.40 | 40 | 0.312x | 1.0000000 | 0.994698 | 2.38 |
| k8v4_S16_ctx2048_B1 | 8.82 (7.57–14.89) | 1.10 | 80 | 0.177x | 1.0000000 | 0.994698 | 2.38 |
| khv4h4_S1_ctx2048_B1 | 1.88 (1.68–2.43) | 0.18 | 5 | 1.000x | 1.0000000 | 0.995228 | 2.32 |
| khv4h4_S2_ctx2048_B1 | 2.20 (1.86–2.70) | 0.26 | 10 | 0.854x | 1.0000000 | 0.995228 | 2.32 |
| khv4h4_S4_ctx2048_B1 | 3.53 (3.32–4.47) | 0.25 | 20 | 0.533x | 1.0000000 | 0.995228 | 2.32 |
| khv4h4_S8_ctx2048_B1 | 6.11 (5.54–14.05) | 0.41 | 40 | 0.308x | 1.0000000 | 0.995228 | 2.32 |
| khv4h4_S16_ctx2048_B1 | 10.85 (9.26–12.64) | 0.72 | 80 | 0.173x | 1.0000000 | 0.995228 | 2.32 |
| a8_S1_ctx2048_B1 | 1.39 (1.10–1.93) | 0.24 | 5 | 1.000x | 1.0000000 | 0.999057 | 2.66 |
| a8_S2_ctx2048_B1 | 1.73 (1.50–2.24) | 0.18 | 10 | 0.807x | 1.0000000 | 0.999057 | 2.66 |
| a8_S4_ctx2048_B1 | 2.81 (2.39–3.32) | 0.27 | 20 | 0.496x | 1.0000000 | 0.999057 | 2.66 |
| a8_S8_ctx2048_B1 | 4.71 (4.07–5.46) | 0.50 | 40 | 0.295x | 1.0000000 | 0.999057 | 2.66 |
| a8_S16_ctx2048_B1 | 8.14 (7.15–9.08) | 0.58 | 80 | 0.171x | 1.0000000 | 0.999057 | 2.66 |
| k8v4_S1_ctx2048_B8 | 10.76 (9.81–14.30) | 0.87 | 40 | 1.000x | 1.0000000 | 0.992416 | 19.06 |
| k8v4_S2_ctx2048_B8 | 12.74 (11.57–14.75) | 0.83 | 80 | 0.844x | 1.0000000 | 0.992416 | 19.06 |
| k8v4_S4_ctx2048_B8 | 21.92 (20.46–26.23) | 1.42 | 160 | 0.491x | 1.0000000 | 0.992416 | 19.06 |
| khv4h4_S1_ctx2048_B8 | 13.39 (12.16–16.13) | 1.49 | 40 | 1.000x | 1.0000000 | 0.995929 | 18.59 |
| khv4h4_S2_ctx2048_B8 | 15.61 (13.72–18.69) | 1.27 | 80 | 0.858x | 1.0000000 | 0.995929 | 18.59 |
| khv4h4_S4_ctx2048_B8 | 27.51 (24.15–35.48) | 2.98 | 160 | 0.487x | 1.0000000 | 0.995929 | 18.59 |
| a8_S1_ctx2048_B8 | 7.25 (6.17–8.58) | 1.28 | 40 | 1.000x | 1.0000000 | 0.996301 | 21.25 |
| a8_S2_ctx2048_B8 | 12.66 (11.25–16.28) | 1.21 | 80 | 0.573x | 1.0000000 | 0.996301 | 21.25 |
| a8_S4_ctx2048_B8 | 20.66 (18.93–25.59) | 2.80 | 160 | 0.351x | 1.0000000 | 0.996301 | 21.25 |
| k8v4_S1_ctx8192_B1 | 4.88 (3.84–16.25) | 0.75 | 5 | 1.000x | 1.0000000 | 0.994402 | 8.71 |
| k8v4_S2_ctx8192_B1 | 4.95 (4.40–6.25) | 0.52 | 10 | 0.985x | 1.0000000 | 0.994402 | 8.71 |
| k8v4_S4_ctx8192_B1 | 4.97 (4.31–6.21) | 0.45 | 20 | 0.981x | 1.0000000 | 0.994402 | 8.71 |
| k8v4_S8_ctx8192_B1 | 6.71 (5.91–9.83) | 0.91 | 40 | 0.727x | 1.0000000 | 0.994402 | 8.71 |
| k8v4_S16_ctx8192_B1 | 11.00 (10.11–13.14) | 0.94 | 80 | 0.443x | 1.0000000 | 0.994402 | 8.71 |
| khv4h4_S1_ctx8192_B1 | 5.28 (4.69–6.62) | 0.69 | 5 | 1.000x | 1.0000000 | 0.995648 | 8.42 |
| khv4h4_S2_ctx8192_B1 | 5.41 (4.73–6.37) | 0.62 | 10 | 0.976x | 1.0000000 | 0.995648 | 8.42 |
| khv4h4_S4_ctx8192_B1 | 5.80 (5.13–6.33) | 0.59 | 20 | 0.910x | 1.0000000 | 0.995648 | 8.42 |
| khv4h4_S8_ctx8192_B1 | 7.98 (7.18–10.83) | 0.55 | 40 | 0.661x | 1.0000000 | 0.995648 | 8.42 |
| khv4h4_S16_ctx8192_B1 | 13.30 (11.97–16.01) | 1.06 | 80 | 0.397x | 1.0000000 | 0.995648 | 8.42 |
| a8_S1_ctx8192_B1 | 3.71 (3.13–6.32) | 0.63 | 5 | 1.000x | 1.0000000 | 0.998464 | 10.62 |
| a8_S2_ctx8192_B1 | 4.09 (3.51–5.06) | 0.35 | 10 | 0.906x | 1.0000000 | 0.998464 | 10.62 |
| a8_S4_ctx8192_B1 | 3.90 (3.48–4.80) | 0.41 | 20 | 0.951x | 1.0000000 | 0.998464 | 10.62 |
| a8_S8_ctx8192_B1 | 6.04 (5.31–7.11) | 0.67 | 40 | 0.614x | 1.0000000 | 0.998464 | 10.62 |
| a8_S16_ctx8192_B1 | 10.08 (9.13–12.37) | 1.15 | 80 | 0.368x | 1.0000000 | 0.998464 | 10.62 |
| k8v4_S1_ctx8192_B8 | 34.52 (27.49–39.19) | 1.77 | 40 | 1.000x | 1.0000000 | 0.989290 | 69.69 |
| k8v4_S2_ctx8192_B8 | 37.21 (28.89–44.41) | 6.52 | 80 | 0.928x | 1.0000000 | 0.989290 | 69.69 |
| k8v4_S4_ctx8192_B8 | 39.26 (32.84–46.03) | 6.35 | 160 | 0.879x | 1.0000000 | 0.989290 | 69.69 |
| khv4h4_S1_ctx8192_B8 | 40.12 (31.33–49.77) | 7.87 | 40 | 1.000x | 1.0000000 | 0.993365 | 67.34 |
| khv4h4_S2_ctx8192_B8 | 44.53 (35.89–51.42) | 5.04 | 80 | 0.901x | 1.0000000 | 0.993365 | 67.34 |
| khv4h4_S4_ctx8192_B8 | 47.70 (39.05–56.55) | 3.19 | 160 | 0.841x | 1.0000000 | 0.993365 | 67.34 |
| a8_S1_ctx8192_B8 | 26.43 (21.14–30.92) | 2.52 | 40 | 1.000x | 1.0000000 | 0.995226 | 85.00 |
| a8_S2_ctx8192_B8 | 29.11 (23.80–31.70) | 1.53 | 80 | 0.908x | 1.0000000 | 0.995226 | 85.00 |
| a8_S4_ctx8192_B8 | 32.62 (26.14–38.03) | 2.99 | 160 | 0.810x | 1.0000000 | 0.995226 | 85.00 |
| k8v4_S1_ctx16384_B1 | 7.27 (6.93–9.17) | 0.48 | 5 | 1.000x | 1.0000000 | 0.993642 | 17.15 |
| k8v4_S2_ctx16384_B1 | 7.78 (7.17–13.69) | 0.91 | 10 | 0.934x | 1.0000000 | 0.993642 | 17.15 |
| k8v4_S4_ctx16384_B1 | 9.01 (8.34–14.18) | 1.23 | 20 | 0.806x | 1.0000000 | 0.993642 | 17.15 |
| k8v4_S8_ctx16384_B1 | 9.41 (8.58–11.91) | 0.68 | 40 | 0.772x | 1.0000000 | 0.993642 | 17.15 |
| k8v4_S16_ctx16384_B1 | 14.58 (12.90–16.84) | 0.95 | 80 | 0.498x | 1.0000000 | 0.993642 | 17.15 |
| khv4h4_S1_ctx16384_B1 | 8.82 (8.03–12.30) | 0.84 | 5 | 1.000x | 1.0000000 | 0.994580 | 16.54 |
| khv4h4_S2_ctx16384_B1 | 9.38 (8.23–13.54) | 1.33 | 10 | 0.940x | 1.0000000 | 0.994580 | 16.54 |
| khv4h4_S4_ctx16384_B1 | 10.76 (9.93–12.93) | 0.62 | 20 | 0.820x | 1.0000000 | 0.994580 | 16.54 |
| khv4h4_S8_ctx16384_B1 | 11.43 (10.15–12.99) | 0.83 | 40 | 0.772x | 1.0000000 | 0.994580 | 16.54 |
| khv4h4_S16_ctx16384_B1 | 17.48 (15.94–24.11) | 0.90 | 80 | 0.505x | 1.0000000 | 0.994580 | 16.54 |
| a8_S1_ctx16384_B1 | 6.80 (5.77–8.16) | 1.21 | 5 | 1.000x | 1.0000000 | 0.998763 | 21.25 |
| a8_S2_ctx16384_B1 | 7.64 (7.03–8.86) | 0.43 | 10 | 0.890x | 1.0000000 | 0.998763 | 21.25 |
| a8_S4_ctx16384_B1 | 7.26 (6.29–8.77) | 0.63 | 20 | 0.937x | 1.0000000 | 0.998763 | 21.25 |
| a8_S8_ctx16384_B1 | 7.82 (7.39–8.69) | 0.35 | 40 | 0.869x | 1.0000000 | 0.998763 | 21.25 |
| a8_S16_ctx16384_B1 | 12.73 (11.65–13.63) | 0.70 | 80 | 0.534x | 1.0000000 | 0.998763 | 21.25 |

## Verdict

**not-viable.** Best 1.045x, below the 1.5x bar in every cell, with exact results. Splitting the history adds tasks, and in numpy each task costs ~88–91 µs of mostly serialized overhead, which outweighs any parallelism. Keep K8V4 with one task per (stream, KV head) for i2sd.

## This wave's pair files

No quant pair reports reached chat this wave (the 5:30 quant derivation lane is paused), so there are no pair files to link. Previous note: [[2026-11-21_0555_full-bitnetd-decode-step-attention-share-u32-lut-2-bit-weights-with-khv4h4-k8v4-a8-kv-vs-int8-a8-b1-b8-ctx-1024-16384]].

## Script

split.py:

    #!/usr/bin/env python3
    """Flash-decoding split-K attention for i2sd decode, 10:55 11/21. Derived from iq.py (02:55 11/21, loaded without main()).
    Baseline (05:55 11/21 path): one task per (stream, KV head), online softmax over TK tiles, 8-thread pool.
    Split-K: each (stream, KV head) history is cut into S contiguous chunks (cut points snapped so no chunk straddles the
    old/A8-window boundary); every chunk is its own pool task returning partial (m, l, acc); partials are merged with the
    log-sum-exp rescale. Question: does splitting each stream's history across the 8 workers speed attention >= 1.5x?
    Formats: k8v4 (i2sd default), khv4h4, a8. Grid: ctx {2048, 8192, 16384} x B {1, 8} (16384 only B=1).
    Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 split.py out.json"""
    import json, sys, time, os, types
    import numpy as np
    src = open("/workspace/bench0255_1121/iq.py").read().rstrip()
    assert src.endswith("main()"); iq = types.ModuleType("iq"); exec(compile(src[:-6], "iq.py", "exec"), iq.__dict__)
    KVH, G, HD, SC, WIN, POOL = iq.KVH, iq.G, iq.HD, iq.SC, iq.WIN, iq.POOL
    FMTS = ("k8v4", "khv4h4", "a8")
    GRID = [(2048, 1), (2048, 8), (8192, 1), (8192, 8), (16384, 1)]
    SPLITS = {1: (1, 2, 4, 8, 16), 8: (1, 2, 4)}
    TK = 2048; ROUNDS, REPS = 7, 5
    def segs(kv, fmt, S):
        """S chunks of [0, n): list of (a, b, old) tiles per chunk; chunks never straddle the window boundary."""
        n = kv.n; wb = 0 if fmt in iq.A8FAM else max(0, n - WIN)
        cuts = sorted({int(round(i * n / S)) for i in range(S + 1)} | ({wb} if 0 < wb < n else set()))
        if len(cuts) - 1 > S:  # merge the smallest neighbour to keep exactly S chunks
            while len(cuts) - 1 > S:
                j = min(range(1, len(cuts) - 1), key=lambda j: (cuts[j + 1] - cuts[j - 1]) if cuts[j] != wb else 1 << 30); cuts.pop(j)
        out = []
        for a0, b0 in zip(cuts[:-1], cuts[1:]):
            old = b0 <= wb if wb else (fmt in iq.A8FAM)
            step = TK if old else WIN
            out.append([(a, min(a + step, b0), old) for a in range(a0, b0, step)])
        return out
    def chunk(kv, Q, h, fmt, tiles):
        st = [np.full((G, 1), -np.inf, np.float32), np.zeros((G, 1), np.float32), np.zeros((G, HD), np.float32)]
        for a, b, old in tiles: st[0] = iq.tile(kv, fmt, h, a, b, Q, st, old)
        return st
    def merge(parts):
        m = np.max(np.concatenate([p[0] for p in parts], 1), 1, keepdims=True)
        l = np.zeros((G, 1), np.float32); acc = np.zeros((G, HD), np.float32)
        for pm, pl, pa in parts: w = np.exp(pm - m); l += w * pl; acc += w * pa
        return acc / l
    def attn_split(kvs, q, fmt, plan):
        futs = [[POOL.submit(chunk, kvs[s], q[s, h], h, fmt, t) for t in plan[s][h]] for s in range(len(kvs)) for h in range(KVH)]
        return np.stack([merge([f.result() for f in fl]) for fl in futs]).reshape(len(kvs), KVH, G, HD)
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    def main():
        rng = iq.rng; out = {"seed": iq.SEED, "cfg": {}}
        print(f"seed {iq.SEED} | TK {TK} | rounds {ROUNDS} x reps {REPS} | pool {iq.W} threads")
        print(f"{'cfg':28s} {'attn ms med (min-max)':>24s} {'iqr':>6s} tasks  speedup  vs-unsplit-cos  vs-f32-cos  KV MiB")
        for n in sorted({g[0] for g in GRID}):
            Bmax = max(B for m, B in GRID if m == n)
            kvs8 = [iq.KV(*iq.kvhist(n)) for _ in range(Bmax)]
            for B in [B for m, B in GRID if m == n]:
                kvs = kvs8[:B]; q = (rng.standard_normal((B, KVH, G, HD)).astype(np.float32) * 2.0) * SC
                ref = np.stack([np.stack([iq.ref(kvs[s], q[s, h], h) for h in range(KVH)]) for s in range(B)])
                cfgs = {}
                for fmt in FMTS:
                    base = iq.attn(kvs, q, fmt, TK).reshape(B, KVH, G, HD)
                    for S in SPLITS[B]:
                        plan = [[segs(kvs[s], fmt, S) for h in range(KVH)] for s in range(B)]
                        o = base if S == 1 else attn_split(kvs, q, fmt, plan)  # S=1 is timed and scored as the unsplit iq.attn path
                        f = (lambda fmt=fmt: iq.attn(kvs, q, fmt, TK)) if S == 1 else (lambda fmt=fmt, plan=plan: attn_split(kvs, q, fmt, plan))
                        cfgs[(fmt, S)] = dict(f=f, t=[], tasks=B * KVH * len(plan[0][0]), cu=min(cos(o[s], base[s]) for s in range(B)),
                                              cr=min(cos(o[s], ref[s]) for s in range(B)), kv=B * kvs[0].bytes(fmt) / 2**20)
                order = list(cfgs)
                for _ in range(2):
                    for k in order: cfgs[k]["f"]()
                for rd in range(ROUNDS):
                    for k in order[rd % len(order):] + order[:rd % len(order)]:
                        for _ in range(REPS):
                            t0 = time.perf_counter(); cfgs[k]["f"](); cfgs[k]["t"].append((time.perf_counter() - t0) * 1e3)
                for (fmt, S), c in cfgs.items():
                    T = np.array(c["t"]); med = float(np.median(T)); b1 = float(np.median(cfgs[(fmt, 1)]["t"]))
                    key = f"{fmt}_S{S}_ctx{n}_B{B}"
                    d = dict(fmt=fmt, S=S, ctx=n, B=B, attn=med, lo=float(T.min()), hi=float(T.max()), iqr=float(np.percentile(T, 75) - np.percentile(T, 25)),
                             tasks=c["tasks"], speedup=b1 / med, cos_unsplit=c["cu"], cos_f32=c["cr"], kv_mib=c["kv"], n=len(T))
                    out["cfg"][key] = d
                    print(f"{key:28s} {med:8.2f} ({d['lo']:.2f}-{d['hi']:.2f}) {d['iqr']:6.2f} {d['tasks']:5d}  {d['speedup']:6.3f}x  {d['cos_unsplit']:.7f}  {d['cos_f32']:.6f}  {d['kv_mib']:6.2f}", flush=True)
            del kvs8
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Next question

Task count, not history length, drives numpy attention cost here. So the opposite move: at B = 8 / ctx 8192, does **head-batched attention** (one batched np.matmul per TK tile over all B × 5 KV heads, i.e. a (40, 4, 128) @ (40, 128, TK) score GEMM and one batched P @ V, with K8V4 dequant done per tile for all heads at once) beat the 40-task pool by ≥ 1.5x? And how does attention scale from a 1-thread to an 8-thread pool (is the pool giving any real parallelism)?

close: 494d66744467fcef
