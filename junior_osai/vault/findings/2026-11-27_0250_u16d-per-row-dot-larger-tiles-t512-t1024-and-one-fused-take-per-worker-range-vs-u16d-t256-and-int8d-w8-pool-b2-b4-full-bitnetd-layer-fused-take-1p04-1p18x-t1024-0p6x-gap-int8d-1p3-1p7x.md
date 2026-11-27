---
local: true
date: 2026-11-27
wave: "2026-11-27 02:55 slot (fired 02:47 MT, ran ~02:47–02:52 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (u16d row-major per-row np.dot on the W 8 decode layer, full bitnetd layer, ctx-free, at B 2 / B 4: larger tiles do not help: T 512 is 0.93–1.03x of u16d_T256 and T 1024 is 0.59–0.69x (slower; f32 tile falls out of cache and the last tiles load-imbalance). One fused take per static worker range followed by 256-row cast + dot tiles (u16ft) is the best new config at 1.06–1.18x (B 2) and 1.04–1.14x (B 4) over u16d_T256 (bar >= 1.3x: missed); a fully fused take + cast + dot over the whole range (u16f) is 0.83–1.12x. Best ternary is still 1.52–1.70x (B 2) and 1.28–1.42x (B 4) slower than int8d (cast + per-row np.dot). All u16 variants are bit-identical to u16d_T256 and within <= 1.04e-6 of the float64 ternary reference (cos 1.0). Memory unchanged: 17.46 MB vs 69.56 MB int8 (3.98x, the known weight-memory win), LUT 52,488 B; the fused variants cost more scratch (u16ft 19.1 MB / worker, u16f 60.0 MB / worker vs 8.8 MB at T 256). No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, u64, uint16, eight-trit-pack, base-3, 2-bit, np-take, per-row-dot, row-major, tile-size, fused-take, static-split, int8, numpy, not-viable]
---

# u16d larger tiles / fused take per worker range: T 1024 is slower, fused take only 1.04–1.18x over T 256, still 1.3–1.7x behind int8d

The 18:43 note ([[2026-11-26_1843_u16-lut-take-contiguous-cast-per-row-np-dot-row-major-vs-t5-tml-t384-and-int8d-w8-pool-b1-b2-full-bitnetd-layer-ties-tml-b1-1p2-1p3x-b2-gap-int8d-1p6-1p8x]]) made u16d (u16 LUT take → contiguous cast → per-row np.dot) the best ternary path at B 2, with take + cast as most of its worker time and T 128 slower than T 256. It asked: at B 2 / B 4, does a larger u16d tile (T 512 / 1024, fewer take/cast calls per worker) or one fused take over a worker's whole row range beat u16d_T256 by >= 1.3x, and does that bring u16d within 1.3x of int8d?

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 2 and B 4, does u16d with T 512 / T 1024, or with one np.take over each worker's whole static row range (u16f: one take, one cast, one dot per b; u16ft: one take, then cast + dot per 256-row tile), beat u16d_T256 by >= 1.3x, and how far is the best of them from int8d?

## Method

- Box: 8 CPUs, numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents, so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1855_1126/v16d.py exec'd without main() (which execs bench1455_1126/v16.py: same weights, P16 8-trit uint16 pack with exact trit round-trip asserts, float64 references, dy / gd2 schedules, r_u16d kernel, profiling hooks).
- Kernels (row-major x (B, in) → y (B, out)):
  - **u16d T 256 / 512 / 1024** (dy and gd2): r_u16d unchanged: np.take(LUT16U 6561 × u64, P16 tile, out=u64 scratch) → int8 view → np.copyto f32 tile → for b: np.dot(tile, x[b]).
  - **u16f_st**: static split (one contiguous row range per worker, ≤ RMAX rows): one np.take over the whole range → one cast to an f32 range buffer → one np.dot per b over the range.
  - **u16ft_st**: static split: one np.take over the whole range into a u64 buffer, then np.copyto + per-b np.dot per 256-row tile.
  - Baselines: u16d_T256 dy / gd2 (18:43 path), int8d_T128 dy / gd2 (cast + per-row np.dot).
- Error: max abs error and cos vs the float64 ternary (u16) / int8 reference, instrumented vs plain output identical, and every u16 variant checked with np.array_equal against u16d_T256 (bit-identical).
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round; median of round medians, round min–max and per-round ratios. Then 15 instrumented layers per u16 config for stage worker-ms (wall/cpu summed over workers) and per-site span. Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench0255_1127 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16t.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Layer ms, median of round medians [round min–max], runs 1 / 2 / 3; then ratios; then stage worker ms (wall/cpu, runs 1, 2, 3) and error. Bytes: weights u16 17,458,176 B vs int8 69,559,296 B (3.98x); LUT 52,488 B; per-worker scratch T 256 8,847,360 B, T 1024 35,389,440 B, u16f 59,996,160 B, u16ft 19,077,120 B.

bytes {'u16': 17458176, 'int8': 69559296, 'int8_over_u16': 3.9843392574344536, 'lut_u16': 52488, 'scratch_T256_per_worker': 8847360, 'scratch_T1024_per_worker': 35389440, 'scratch_u16f_per_worker': 59996160, 'scratch_u16ft_per_worker': 19077120}
### B2
| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | u16d_dy | 12.69 [11.65–13.34] | 12.96 [12.66–13.73] | 12.46 [12.24–12.84] |
| 2 | u16d_gd2 | 11.26 [10.78–11.79] | 12.56 [12.31–13.53] | 11.70 [11.51–11.97] |
| 2 | u16d512_dy | 13.12 [11.74–13.44] | 15.35 [14.67–16.75] | 12.54 [12.02–13.52] |
| 2 | u16d512_gd2 | 11.77 [10.82–12.49] | 12.23 [11.24–15.01] | 11.36 [10.82–12.13] |
| 2 | u16d1024_dy | 20.90 [19.34–31.83] | 23.02 [19.01–40.50] | 20.01 [18.54–23.49] |
| 2 | u16d1024_gd2 | 19.08 [16.99–22.90] | 19.61 [18.50–25.32] | 17.50 [16.03–29.55] |
| 2 | u16f_st | 11.02 [10.03–15.12] | 14.65 [12.53–23.32] | 10.41 [9.73–15.44] |
| 2 | u16ft_st | 10.49 [8.80–11.43] | 11.90 [11.23–12.71] | 9.94 [9.47–11.61] |
| 2 | i8d_dy | 7.42 [7.13–7.61] | 8.19 [7.73–8.53] | 7.24 [6.95–7.46] |
| 2 | i8d_gd2 | 6.60 [6.27–7.11] | 7.00 [6.79–7.67] | 6.56 [6.36–6.98] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | T256/best_new | 1.074x [1.015–1.251] (u16d_gd2/u16ft_st) | 1.056x [1.028–1.121] (u16d_gd2/u16ft_st) | 1.177x [1.008–1.264] (u16d_gd2/u16ft_st) |
| 2 | T256/best_T512 | 0.957x [0.902–0.996] (u16d_gd2/u16d512_gd2) | 1.028x [0.878–1.103] (u16d_gd2/u16d512_gd2) | 1.030x [0.950–1.065] (u16d_gd2/u16d512_gd2) |
| 2 | T256/best_T1024 | 0.590x [0.506–0.659] (u16d_gd2/u16d1024_gd2) | 0.641x [0.534–0.675] (u16d_gd2/u16d1024_gd2) | 0.669x [0.390–0.723] (u16d_gd2/u16d1024_gd2) |
| 2 | T256/u16f | 1.022x [0.767–1.124] (u16d_gd2/u16f_st) | 0.858x [0.580–1.003] (u16d_gd2/u16f_st) | 1.124x [0.750–1.209] (u16d_gd2/u16f_st) |
| 2 | T256/u16ft | 1.074x [1.015–1.251] (u16d_gd2/u16ft_st) | 1.056x [1.028–1.121] (u16d_gd2/u16ft_st) | 1.177x [1.008–1.264] (u16d_gd2/u16ft_st) |
| 2 | best_new/best_i8d | 1.590x [1.403–1.676] (u16ft_st/i8d_gd2) | 1.699x [1.464–1.769] (u16ft_st/i8d_gd2) | 1.516x [1.367–1.770] (u16ft_st/i8d_gd2) |
| 2 | T256/best_i8d | 1.708x [1.517–1.786] (u16d_gd2/i8d_gd2) | 1.794x [1.642–1.941] (u16d_gd2/i8d_gd2) | 1.784x [1.675–1.859] (u16d_gd2/i8d_gd2) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 2 | u16d_dy | 35.6/21.0, 39.5/21.0, 38.8/20.1 | 15.5/12.7, 17.9/14.4, 15.5/12.5 | 26.5/18.5, 27.0/19.2, 27.3/18.7 | 78.4/54.1, 86.5/56.3, 83.6/52.7 | 12.19, 13.12, 12.85 |
| 2 | u16d_gd2 | 33.2/19.7, 39.9/21.8, 40.4/21.0 | 15.7/13.3, 18.9/15.3, 17.1/13.9 | 24.0/17.6, 26.5/18.7, 27.1/18.5 | 74.6/51.9, 86.1/57.3, 86.4/55.1 | 10.86, 12.54, 12.49 |
| 2 | u16d512_dy | 27.2/20.1, 30.9/23.1, 27.5/20.2 | 14.7/14.2, 17.2/16.7, 14.8/14.3 | 21.4/19.6, 22.6/21.1, 21.1/19.5 | 64.2/54.9, 72.0/61.1, 64.3/54.8 | 11.92, 15.00, 12.05 |
| 2 | u16d512_gd2 | 28.4/20.5, 30.6/21.3, 33.0/21.9 | 17.0/15.2, 18.2/16.5, 18.7/17.2 | 23.5/19.7, 23.2/20.1, 25.2/21.2 | 70.2/56.6, 72.9/59.2, 79.5/61.5 | 11.42, 11.72, 12.65 |
| 2 | u16d1024_dy | 26.5/23.2, 29.9/26.1, 27.9/24.6 | 17.8/17.4, 20.5/20.3, 18.2/18.2 | 21.7/21.3, 23.6/23.2, 22.0/21.6 | 66.9/63.8, 75.4/71.0, 69.1/65.7 | 19.63, 24.00, 19.82 |
| 2 | u16d1024_gd2 | 32.7/28.4, 29.4/24.7, 28.9/25.1 | 24.5/24.3, 22.2/21.9, 22.5/22.2 | 26.2/25.5, 23.6/23.0, 24.5/24.1 | 87.2/79.8, 77.0/71.1, 77.6/72.6 | 19.98, 17.77, 16.76 |
| 2 | u16f_st | 25.3/21.0, 28.4/23.3, 27.5/22.8 | 14.7/14.3, 18.1/17.6, 17.2/16.7 | 20.0/19.3, 21.5/21.0, 21.8/21.0 | 61.3/55.4, 68.0/62.7, 67.6/61.4 | 9.95, 13.92, 11.52 |
| 2 | u16ft_st | 24.6/19.4, 26.1/21.4, 25.7/20.6 | 15.9/12.9, 16.2/14.1, 16.4/13.5 | 23.8/17.6, 22.2/17.3, 24.2/18.3 | 65.4/50.8, 65.7/54.0, 68.2/53.4 | 10.09, 11.33, 10.45 |

| B | config | max abs err vs ref | cos | instr identical | bit-identical to u16d_T256 |
|---|---|---|---|---|---|
| 2 | u16d_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 2 | u16d_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 2 | u16d512_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 2 | u16d512_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 2 | u16d1024_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 2 | u16d1024_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 2 | u16f_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 2 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 2 | i8d_dy | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | – |
| 2 | i8d_gd2 | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | – |

### B4
| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 4 | u16d_dy | 14.48 [14.10–15.53] | 15.43 [14.84–16.16] | 16.43 [15.20–16.97] |
| 4 | u16d_gd2 | 13.49 [13.34–13.76] | 14.05 [13.82–16.70] | 15.53 [14.38–15.85] |
| 4 | u16d512_dy | 15.53 [15.27–15.72] | 17.17 [15.22–18.44] | 17.96 [14.95–19.39] |
| 4 | u16d512_gd2 | 14.53 [13.91–15.02] | 14.41 [13.59–16.43] | 15.57 [14.93–16.26] |
| 4 | u16d1024_dy | 25.07 [24.01–25.24] | 27.98 [25.12–30.08] | 26.97 [24.93–29.37] |
| 4 | u16d1024_gd2 | 21.11 [20.22–21.36] | 20.24 [19.60–24.64] | 24.20 [21.59–26.52] |
| 4 | u16f_st | 13.01 [12.00–15.62] | 16.98 [13.51–18.33] | 17.92 [17.22–19.06] |
| 4 | u16ft_st | 12.61 [11.51–14.47] | 12.32 [11.30–15.29] | 14.99 [12.48–15.76] |
| 4 | i8d_dy | 9.59 [9.43–9.95] | 9.96 [9.47–11.72] | 11.44 [10.70–12.92] |
| 4 | i8d_gd2 | 8.95 [8.79–9.59] | 9.61 [8.44–10.24] | 10.53 [9.27–10.99] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 4 | T256/best_new | 1.070x [0.940–1.160] (u16d_gd2/u16ft_st) | 1.140x [0.967–1.244] (u16d_gd2/u16ft_st) | 1.036x [1.006–1.152] (u16d_gd2/u16ft_st) |
| 4 | T256/best_T512 | 0.928x [0.888–0.989] (u16d_gd2/u16d512_gd2) | 0.975x [0.900–1.142] (u16d_gd2/u16d512_gd2) | 0.998x [0.950–1.035] (u16d_gd2/u16d512_gd2) |
| 4 | T256/best_T1024 | 0.639x [0.624–0.666] (u16d_gd2/u16d1024_gd2) | 0.694x [0.600–0.705] (u16d_gd2/u16d1024_gd2) | 0.642x [0.593–0.695] (u16d_gd2/u16d1024_gd2) |
| 4 | T256/u16f | 1.037x [0.881–1.130] (u16d_gd2/u16f_st) | 0.828x [0.807–1.035] (u16d_gd2/u16f_st) | 0.867x [0.797–0.913] (u16d_gd2/u16f_st) |
| 4 | T256/u16ft | 1.070x [0.940–1.160] (u16d_gd2/u16ft_st) | 1.140x [0.967–1.244] (u16d_gd2/u16ft_st) | 1.036x [1.006–1.152] (u16d_gd2/u16ft_st) |
| 4 | best_new/best_i8d | 1.409x [1.305–1.646] (u16ft_st/i8d_gd2) | 1.282x [1.175–1.530] (u16ft_st/i8d_gd2) | 1.424x [1.347–1.497] (u16ft_st/i8d_gd2) |
| 4 | T256/best_i8d | 1.508x [1.391–1.546] (u16d_gd2/i8d_gd2) | 1.461x [1.461–1.679] (u16d_gd2/i8d_gd2) | 1.475x [1.414–1.551] (u16d_gd2/i8d_gd2) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 4 | u16d_dy | 31.9/19.6, 38.7/19.7, 37.2/19.2 | 14.7/12.0, 14.7/12.0, 14.4/12.0 | 44.4/34.0, 46.5/34.7, 46.2/34.6 | 93.6/67.6, 102.4/68.5, 99.0/67.4 | 14.63, 16.03, 15.40 |
| 4 | u16d_gd2 | 33.0/19.7, 36.9/19.8, 37.7/19.7 | 14.8/12.7, 15.7/12.9, 15.1/12.4 | 43.1/33.2, 44.8/33.5, 46.3/33.4 | 93.9/67.5, 99.2/68.3, 101.0/67.7 | 13.38, 14.11, 14.37 |
| 4 | u16d512_dy | 27.8/21.0, 26.6/19.4, 25.8/18.9 | 14.6/13.9, 13.8/13.1, 13.1/12.6 | 38.7/36.8, 38.8/36.9, 38.0/35.9 | 82.1/73.4, 80.5/70.7, 78.3/68.4 | 15.35, 14.86, 13.91 |
| 4 | u16d512_gd2 | 27.9/21.0, 29.5/19.9, 30.4/19.5 | 15.9/14.8, 16.2/14.6, 14.6/13.5 | 40.9/36.1, 41.1/36.3, 40.2/35.4 | 86.2/72.9, 88.9/71.9, 87.1/69.2 | 14.00, 14.14, 14.04 |
| 4 | u16d1024_dy | 26.1/23.0, 27.0/23.4, 27.5/23.2 | 16.4/16.4, 16.5/16.5, 16.3/16.2 | 38.8/38.0, 39.4/38.9, 38.9/38.2 | 82.1/78.6, 84.9/79.8, 83.9/79.3 | 23.25, 24.17, 23.90 |
| 4 | u16d1024_gd2 | 29.7/24.9, 30.6/26.0, 31.2/25.8 | 20.1/19.8, 22.0/21.8, 21.3/21.2 | 40.5/39.6, 42.5/41.4, 41.9/41.4 | 91.1/84.7, 96.3/90.0, 98.3/89.6 | 19.41, 20.53, 20.93 |
| 4 | u16f_st | 26.1/21.4, 30.5/25.1, 27.6/22.6 | 14.1/13.6, 16.8/16.2, 16.8/16.5 | 38.3/36.8, 40.3/39.0, 40.3/38.5 | 80.3/72.4, 89.6/80.9, 85.6/78.8 | 12.70, 13.62, 14.56 |
| 4 | u16ft_st | 23.5/18.4, 23.9/18.8, 28.7/24.0 | 14.3/12.1, 15.1/12.6, 16.9/14.5 | 40.0/31.5, 41.6/32.9, 43.2/34.0 | 78.5/62.8, 83.2/65.3, 90.8/74.0 | 11.98, 12.48, 14.84 |

| B | config | max abs err vs ref | cos | instr identical | bit-identical to u16d_T256 |
|---|---|---|---|---|---|
| 4 | u16d_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 4 | u16d_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 4 | u16d512_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 4 | u16d512_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 4 | u16d1024_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 4 | u16d1024_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 4 | u16f_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 4 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True |
| 4 | i8d_dy | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | – |
| 4 | i8d_gd2 | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | – |

## Verdict

**not-viable.** Bigger tiles are not the lever: T 512 is a wash (0.93–1.03x of u16d_T256) and T 1024 is clearly slower (0.59–0.69x): its take gets cheaper (worker-ms 33–40 → 26–33 at B 2) but cast and dot lose cache residency (f32 tile 28 MB at in 6912) and the coarse tiles load-imbalance (span 17–24 ms vs 11–13 ms). One fused take per worker range is the only gain, and it is small: u16ft (fused take, then 256-row cast + dot) is 1.06–1.18x at B 2 and 1.04–1.14x at B 4 over u16d_T256, short of the 1.3x bar; fusing the cast and dot over the whole range too (u16f) loses the cache benefit again (0.83–1.12x). Best ternary stays 1.52–1.70x (B 2) and 1.28–1.42x (B 4) slower than int8d; B 4 dipped just under 1.3x in one run of three, but not consistently. Outputs are bit-identical to u16d_T256 for all u16 variants. Ternary keeps only the known 3.98x weight-memory win (17.46 vs 69.56 MB); the fused take adds 10–51 MB of scratch per worker.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-27 02:55 wave reached this lane. JuniorLLM HEAD at clone was cab33d0 (cloudcover95, 18:47 MT 11/26, this lane's 18:43 note); nothing else landed after it, so no Trit Pack, Handshake or LLC commit falls in the last 6 hours. This note only adds one new file under junior_osai/vault/findings/.

## Next question

The cast stage (~15 worker-ms) is now the cheapest thing left to remove in the u16ft pipeline. At B 2 / B 4 row-major, does taking 8 trits per uint16 straight into the f32 tile from a 6561 × 8 f32 LUT (210 KB, no int8 cast), with the fused per-worker-range index pass and 256-row dot tiles, beat u16ft_st by >= 1.3x and bring u16 within 1.3x of int8d? (The 11-26 05:41 v20 note tested direct f32 take only for the 243-row t5 weight-major path.)

## Script (v16t.py)

```python
#!/usr/bin/env python3
"""V16T: u16d (8-trit uint16 pack + 6561-row u64 LUT take -> contiguous int8 tile -> f32 -> per-row np.dot, row-major) with larger
tiles T512 / T1024 and with one fused take per worker range, vs u16d_T256 and int8d, at B2 / B4, W=8 decode layer, full bitnetd layer, ctx-free.
Base: /workspace/bench1855_1126/v16d.py (exec'd without main(); it execs bench1455_1126/v16.py: same weights, P5 + P16 packs with exact
trit round-trip asserts, float64 refs, dy / gd2 schedules, r_u16d kernel, profiling hooks).
  u16d{T}  : r_u16d unchanged, T 256 (baseline) / 512 / 1024, dy and gd2 schedules
  u16f     : static split (one contiguous row range per worker): ONE np.take over the whole range -> ONE cast -> per-b np.dot over the range
  u16ft    : static split: ONE np.take over the whole range, then cast + per-b np.dot per 256-row tile (f32 tile stays small)
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16t.py out.json"""
import json, os, sys
import numpy as np
_S = open("/workspace/bench1855_1126/v16d.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "v16d.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
RMAX = max(-(-par.F8[k]["Q"].shape[0] // W) + 8 for k in SITES)          # max rows in one static worker range
def r_u16f(PR, x, y, r0, r1, T, w, i):
    q = PR.shape[1]; n = r1 - r0; P = PRF[0]
    su = scr(("u16f", w), (RMAX * Q8,), np.uint64)[:n * q].reshape(n, q); s = scr(("f16f", w), (RMAX * F,), np.float32)[:n * q * 8].reshape(n, q * 8)
    if P: t0 = pc(); c0 = tt()
    np.take(LUT16U, PR[r0:r1], out=su)
    if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
    np.copyto(s, su.view(np.int8).reshape(n, q * 8), casting="unsafe")
    if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
    for b in range(x.shape[0]): np.dot(s, x[b], out=y[b, r0:r1])
    if P: _add(w, "dot", pc() - t0, tt() - c0)
def r_u16ft(PR, x, y, r0, r1, T, w, i):
    q = PR.shape[1]; n = r1 - r0; P = PRF[0]; TT = 256
    su = scr(("u16f", w), (RMAX * Q8,), np.uint64)[:n * q].reshape(n, q); s8 = su.view(np.int8).reshape(n, q * 8)
    s = scr(("f16", w, TT), (TT * F,), np.float32)[:TT * q * 8].reshape(TT, q * 8)
    if P: t0 = pc(); c0 = tt()
    np.take(LUT16U, PR[r0:r1], out=su)
    if P: _add(w, "take", pc() - t0, tt() - c0)
    for a in range(0, n, TT):
        m = min(TT, n - a)
        if P: t0 = pc(); c0 = tt()
        np.copyto(s[:m], s8[a:a+m], casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        for b in range(x.shape[0]): np.dot(s[:m], x[b], out=y[b, r0+a:r0+a+m])
        if P: _add(w, "dot", pc() - t0, tt() - c0)
KD["u16f"] = r_u16f; KD["u16ft"] = r_u16ft
C = {"u16d_dy": ("u16d", 256, "dy"), "u16d_gd2": ("u16d", 256, "gd2"), "u16d512_dy": ("u16d", 512, "dy"), "u16d512_gd2": ("u16d", 512, "gd2"),
     "u16d1024_dy": ("u16d", 1024, "dy"), "u16d1024_gd2": ("u16d", 1024, "gd2"), "u16f_st": ("u16f", 8, "st"), "u16ft_st": ("u16ft", 8, "st"),
     "i8d_dy": ("8d", 128, "dy"), "i8d_gd2": ("8d", 128, "gd2")}
U = [c for c in C if C[c][0] in KD]; BASE = ["u16d_dy", "u16d_gd2"]; NEW = [c for c in U if c not in BASE]; I8 = ["i8d_dy", "i8d_gd2"]
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, RMAX=RMAX, time={}, prof={}, err={})
    W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES); W16 = sum(P16[k]["PR"].nbytes + P16[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(u16=W16, int8=W8, int8_over_u16=W8 / W16, lut_u16=LUT16U.nbytes,
                        scratch_T256_per_worker=256 * (Q8 * 8 + F * 4), scratch_T1024_per_worker=1024 * (Q8 * 8 + F * 4),
                        scratch_u16f_per_worker=RMAX * (Q8 * 8 + F * 4), scratch_u16ft_per_worker=RMAX * Q8 * 8 + 256 * F * 4)
    print("bytes", res["bytes"], flush=True)
    names = list(C)
    for B in (2, 4):
        xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0] == "8d" else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in U: e[c]["vs_u16dT256_mab"] = float(np.abs(outs[c] - outs["u16d_dy"]).max()); e[c]["bitident_u16dT256"] = bool(np.array_equal(outs[c], outs["u16d_dy"]))
        res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.12f}", v["instr_identical"], v.get("bitident_u16dT256")) for c, v in e.items()}, flush=True)
        xb = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32); rm = {c: [] for c in names}
        for c in names: layer(xb, *C[c], [])
        for rd in range(ROUNDS):
            for c in names[rd % len(names):] + names[:rd % len(names)]:
                s = []
                for _ in range(REPS):
                    t0 = pc(); layer(xb, *C[c], []); s.append((pc() - t0) * 1e3)
                rm[c].append(float(np.median(s)))
        o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), rm=v) for c, v in rm.items()}; rat = {}
        def pr_(a, b, key=None):
            v = [p / q for p, q in zip(rm[a], rm[b])]; rat[key or f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v), a=a, b=b)
        b0 = min(BASE, key=lambda c: o[c]["med"]); bn = min(NEW, key=lambda c: o[c]["med"]); bi = min(I8, key=lambda c: o[c]["med"])
        pr_(b0, bn, "T256/best_new"); pr_(b0, "u16d512_gd2" if o["u16d512_gd2"]["med"] < o["u16d512_dy"]["med"] else "u16d512_dy", "T256/best_T512")
        pr_(b0, "u16d1024_gd2" if o["u16d1024_gd2"]["med"] < o["u16d1024_dy"]["med"] else "u16d1024_dy", "T256/best_T1024")
        pr_(b0, "u16f_st", "T256/u16f"); pr_(b0, "u16ft_st", "T256/u16ft")
        pr_(bn, bi, "best_new/best_i8d"); pr_(b0, bi, "T256/best_i8d")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:13s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:20s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
        pr = {}; PRF[0] = True
        for c in U:
            per = {}
            for _ in range(PROF):
                for d in ACC: d.clear()
                rec = []; layer(xb, *C[c], rec); it = {}
                for (site, st), (dw, dc, n) in ((k, v) for d in ACC for k, v in d.items()):
                    a = it.setdefault(st, [0.0, 0.0]); a[0] += dw; a[1] += dc
                a = it.setdefault("task", [0.0, 0.0]); sp = {}
                for site, w, t0, t1, cpu, n in rec: a[0] += t1 - t0; a[1] += cpu; sp[site] = max(sp.get(site, 0), t1)
                it["span"] = [sum(sp.values()), 0.0]
                for st, v in it.items(): per.setdefault(st, []).append(v)
            pr[c] = {st: dict(wall=float(np.median([v[0] for v in L])) * 1e3, cpu=float(np.median([v[1] for v in L])) * 1e3) for st, L in per.items()}
            print(f"B{B} prof {c:13s} " + "  ".join(f"{st} {v['wall']:.1f}/{v['cpu']:.1f}ms" for st, v in pr[c].items()), flush=True)
        PRF[0] = False; res["prof"][f"B{B}"] = pr
    json.dump(res, open(sys.argv[1], "w"), indent=1)
main()
```

close: bbf0952b4256f0ea
