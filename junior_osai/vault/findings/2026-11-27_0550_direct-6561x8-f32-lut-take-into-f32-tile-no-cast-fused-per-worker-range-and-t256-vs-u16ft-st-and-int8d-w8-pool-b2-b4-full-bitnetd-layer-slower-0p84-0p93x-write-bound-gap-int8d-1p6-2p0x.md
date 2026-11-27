---
local: true
date: 2026-11-27
wave: "2026-11-27 05:55 slot (fired 05:49 MT, ran ~05:50–05:53 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (direct 6561 x 8 f32 LUT take into the f32 tile, skipping the int8 -> f32 cast, on the W 8 decode layer, full bitnetd layer, ctx-free, at B 2 / B 4: every variant is SLOWER than u16ft_st: best new is 0.84–0.89x (B 2) and 0.86–0.93x (B 4) of u16ft_st (bar >= 1.3x: missed), and 0.81–0.88x of u16d_T256. One fused f32 take over the whole worker range is 0.24–0.34x (48 MB f32 scratch per worker). Single-thread 256-row tile: f32 take 1291–1385 us vs u64 take + cast 722–768 us (1.7–1.8x slower); it is write-bandwidth bound (7.08 MB f32 tile vs 1.77 MB u64 tile), not LUT-cache bound: codes restricted to 64 hot LUT rows take the same time as random codes for the 210 KB f32 LUT (and for the 52 KB u64 LUT). Best ternary stays behind int8d: best new 1.76–2.03x (B 2) / 1.62–1.71x (B 4), u16ft_st 1.57–1.70x / 1.45–1.58x this wave. All variants bit-identical to u16ft_st and u16d_T256, <= 1.04e-6 vs the float64 ternary reference (cos 1.0). Memory unchanged: 17.46 MB vs 69.56 MB int8 (3.98x, the known weight-memory win); LUT 209,952 B f32 vs 52,488 B u64. No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, f32-lut, uint16, eight-trit-pack, base-3, 2-bit, np-take, void-dtype, per-row-dot, row-major, fused-take, static-split, cache, write-bandwidth, int8, numpy, not-viable]
---

# Direct f32 LUT take (no cast) is slower than u64 take + cast: 0.84–0.93x of u16ft, write-bound, gap to int8d widens to 1.6–2.0x

The 02:47 note ([[2026-11-27_0250_u16d-per-row-dot-larger-tiles-t512-t1024-and-one-fused-take-per-worker-range-vs-u16d-t256-and-int8d-w8-pool-b2-b4-full-bitnetd-layer-fused-take-1p04-1p18x-t1024-0p6x-gap-int8d-1p3-1p7x]]) made u16ft_st (one u64 LUT take per static worker range, then 256-row cast + per-row np.dot) the best ternary row-major path at B 2 / B 4, with the cast (~15–20 worker-ms) as the cheapest stage left to remove. It asked: does taking 8 trits per uint16 straight into the f32 tile from a 6561 × 8 f32 LUT (210 KB, no int8 cast), with the fused per-worker-range index pass and 256-row dot tiles, beat u16ft_st by >= 1.3x and bring u16 within 1.3x of int8d? (The 11-26 05:41 v20 note tested direct f32 take only for the 243-row t5 weight-major path.)

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 2 and B 4, does a direct take from a 6561 × 8 f32 LUT into the f32 tile (no int8 → f32 cast) beat u16ft_st by >= 1.3x, how far is it from int8d, and what does the 4x larger LUT (210 KB vs 52 KB) do to cache behaviour?

## Method

- Box: 8 CPUs (Xeon, L1d 48 KiB / L2 2 MiB per core, L3 320 MiB), numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents (load avg ~2 during the runs, higher than at 02:47, so absolute ms are ~10–40% up); ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench0255_1127/v16t.py exec'd without main() (it execs v16d.py → v16.py: same weights, P16 8-trit uint16 pack with exact trit round-trip asserts, float64 references, dy / gd2 schedules, r_u16d / r_u16ft kernels, profiling hooks).
- New LUT: LUT16F = L16 as f32 (6561, 8) = 209,952 B; LUT16V = the same bytes viewed as one 32-byte void element per code, so np.take is a 1-D take that memcpys a 32-byte row per code (asserted equal to L16).
- Kernels (row-major x (B, in) → y (B, out)):
  - **u16fvt_st**: static split; per 256-row tile np.take(LUT16V, P16 tile, out=void view of the f32 tile) → per-b np.dot. No cast.
  - **u16fvf_st**: static split; ONE void take over the whole worker range into an f32 range buffer (≤ RMAX rows) → per-b np.dot per 256-row slice (the literal "one fused take per worker range" in f32).
  - **u16fv_dy / u16fv_gd2**: the u16d T 256 dy / gd2 queue schedules with the void f32 take instead of u64 take + cast.
  - **u16fa_st**: as u16fvt_st but np.take(LUT16F, idx, axis=0) on the 2-D f32 LUT (no void trick).
  - Baselines: u16ft_st (02:47 best), u16d_T256 gd2, int8d_T128 dy / gd2 (cast + per-row np.dot).
- Cache microbench (single thread, largest site "down", 256 rows × q8 864): u64 take, u64 take + cast, f32 void take, f32 axis-0 take, and the f32 / u64 takes with random codes over all 6561 LUT rows vs codes restricted to 64 hot rows (LUT resident in L1); 41 reps, median [min–max].
- Error: max abs error and cos vs the float64 ternary (u16) / int8 reference, instrumented vs plain output identical, and every u16 variant checked with np.array_equal against u16ft_st and u16d_T256.
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round; median of round medians, round min–max and per-round ratios. Then 15 instrumented layers per u16 config for stage worker-ms (wall/cpu summed over workers) and per-site span. Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench0555_1127 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16fl.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Bytes: weights u16 17,458,176 B vs int8 69,559,296 B (3.98x); LUT u64 52,488 B vs f32 209,952 B; per-worker scratch u16ft 19,077,120 B, u16fvt 7,077,888 B, u16fvf 47,996,928 B. Then the microbench, layer ms (median of round medians [round min–max], runs 1 / 2 / 3), ratios, stage worker ms (wall/cpu, runs 1, 2, 3) and error.

bytes {'u16': 17458176, 'int8': 69559296, 'int8_over_u16': 3.9843392574344536, 'lut_u64': 52488, 'lut_f32': 209952, 'scratch_u16ft_per_worker': 19077120, 'scratch_u16fvt_per_worker': 7077888, 'scratch_u16fvf_per_worker': 47996928}

Single-thread 256-row tile microbench, site down (q8 864 ; f32 tile 7077888 B, u64 tile 1769472 B), µs median [min–max], runs 1 / 2 / 3:

| op | run 1 | run 2 | run 3 |
|---|---|---|---|
| u64_take | 450 [436–717] | 443 [421–639] | 431 [403–535] |
| u64_take_cast | 757 [732–1461] | 768 [725–1398] | 722 [675–1368] |
| f32_void_take | 1355 [1282–1571] | 1385 [1298–1512] | 1291 [1224–1463] |
| f32_axis0_take | 1336 [1300–1434] | 1330 [1276–1468] | 1308 [1267–1414] |
| f32_void_take_random | 1338 [1251–1485] | 1344 [1277–1468] | 1352 [1237–1497] |
| f32_void_take_hot64 | 1310 [1205–1413] | 1279 [1184–1387] | 1381 [1280–1600] |
| u64_take_random | 451 [416–498] | 443 [405–510] | 445 [414–677] |
| u64_take_hot64 | 441 [411–474] | 429 [398–491] | 428 [394–481] |

### B2
| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | u16d_gd2 | 14.22 [13.73–16.13] | 15.29 [13.65–16.03] | 13.46 [11.74–15.12] |
| 2 | u16ft_st | 15.62 [14.26–17.50] | 15.44 [13.48–16.66] | 13.94 [10.69–15.52] |
| 2 | u16fvt_st | 19.08 [17.21–23.73] | 19.33 [17.58–21.38] | 16.38 [15.09–18.96] |
| 2 | u16fvf_st | 55.74 [47.89–65.79] | 59.54 [56.13–64.62] | 58.04 [51.40–68.18] |
| 2 | u16fv_dy | 17.86 [16.89–19.69] | 18.68 [17.40–20.69] | 16.72 [16.23–17.99] |
| 2 | u16fv_gd2 | 17.56 [17.01–18.00] | 18.45 [16.75–21.26] | 16.59 [15.38–18.94] |
| 2 | u16fa_st | 18.38 [17.10–19.64] | 19.48 [17.16–21.16] | 17.32 [16.42–19.77] |
| 2 | i8d_dy | 10.79 [10.02–11.82] | 10.45 [9.20–11.88] | 9.48 [8.84–11.19] |
| 2 | i8d_gd2 | 9.96 [9.25–10.81] | 9.09 [7.92–10.64] | 8.38 [7.60–9.76] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | u16ft/best_new | 0.890x [0.793–1.010] (u16ft_st/u16fv_gd2) | 0.837x [0.752–0.923] (u16ft_st/u16fv_gd2) | 0.851x [0.708–0.934] (u16ft_st/u16fvt_st) |
| 2 | u16ft/u16fvt_st | 0.819x [0.653–0.947] (u16ft_st/u16fvt_st) | 0.799x [0.710–0.861] (u16ft_st/u16fvt_st) | 0.851x [0.708–0.934] (u16ft_st/u16fvt_st) |
| 2 | u16ft/u16fvf_st | 0.280x [0.220–0.340] (u16ft_st/u16fvf_st) | 0.259x [0.224–0.279] (u16ft_st/u16fvf_st) | 0.240x [0.199–0.261] (u16ft_st/u16fvf_st) |
| 2 | u16ft/u16fv_dy | 0.875x [0.781–0.989] (u16ft_st/u16fv_dy) | 0.827x [0.731–0.887] (u16ft_st/u16fv_dy) | 0.834x [0.637–0.931] (u16ft_st/u16fv_dy) |
| 2 | u16ft/u16fv_gd2 | 0.890x [0.793–1.010] (u16ft_st/u16fv_gd2) | 0.837x [0.752–0.923] (u16ft_st/u16fv_gd2) | 0.840x [0.645–0.939] (u16ft_st/u16fv_gd2) |
| 2 | u16ft/u16fa_st | 0.850x [0.726–0.956] (u16ft_st/u16fa_st) | 0.793x [0.764–0.876] (u16ft_st/u16fa_st) | 0.805x [0.541–0.893] (u16ft_st/u16fa_st) |
| 2 | u16dT256/best_new | 0.810x [0.782–0.931] (u16d_gd2/u16fv_gd2) | 0.829x [0.701–0.957] (u16d_gd2/u16fv_gd2) | 0.822x [0.747–0.910] (u16d_gd2/u16fvt_st) |
| 2 | best_new/best_i8d | 1.763x [1.639–1.898] (u16fv_gd2/i8d_gd2) | 2.030x [1.858–2.262] (u16fv_gd2/i8d_gd2) | 1.956x [1.793–2.150] (u16fvt_st/i8d_gd2) |
| 2 | u16ft/best_i8d | 1.569x [1.428–1.757] (u16ft_st/i8d_gd2) | 1.699x [1.566–1.757] (u16ft_st/i8d_gd2) | 1.664x [1.407–1.785] (u16ft_st/i8d_gd2) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 2 | u16d_gd2 | 39.3/24.9, 44.3/29.1, 37.6/22.8 | 24.7/21.8, 26.5/23.0, 19.4/16.0 | 27.5/21.6, 29.6/22.8, 26.9/20.2 | 93.7/70.9, 102.2/76.7, 86.3/61.1 | 14.02, 15.27, 12.63 |
| 2 | u16ft_st | 33.2/29.1, 40.1/34.9, 28.3/23.0 | 22.9/19.7, 23.3/20.6, 18.4/15.3 | 27.2/20.8, 28.1/21.9, 26.4/19.7 | 85.6/72.3, 92.3/78.7, 75.8/60.1 | 13.92, 15.05, 12.08 |
| 2 | u16fvt_st | 82.0/69.6, 91.5/77.1, 69.7/58.5 | – | 26.4/20.8, 26.9/22.0, 26.2/19.6 | 109.8/91.6, 119.4/100.4, 97.3/79.7 | 16.89, 18.83, 14.29 |
| 2 | u16fvf_st | 212.4/202.4, 233.7/224.3, 208.8/196.5 | – | 29.1/25.2, 34.8/26.5, 27.4/23.1 | 243.0/229.3, 272.6/253.2, 242.1/220.9 | 52.48, 56.96, 52.93 |
| 2 | u16fv_dy | 82.2/70.4, 93.1/77.8, 74.8/58.7 | – | 26.8/21.9, 29.1/23.0, 27.5/20.3 | 113.4/95.2, 124.6/102.6, 104.6/80.7 | 18.26, 19.66, 16.48 |
| 2 | u16fv_gd2 | 76.3/62.2, 95.3/79.9, 83.6/63.8 | – | 26.8/20.7, 27.4/22.0, 29.1/21.2 | 105.1/84.6, 124.2/103.1, 113.4/86.7 | 15.24, 18.92, 16.49 |
| 2 | u16fa_st | 71.3/59.5, 88.6/74.5, 82.1/61.6 | – | 25.0/19.1, 28.5/22.4, 28.4/20.4 | 98.0/79.8, 118.7/97.8, 110.7/83.4 | 14.61, 19.05, 17.76 |

| B | config | max abs err vs ref | cos | instr identical | bit-identical to u16ft_st | bit-identical to u16d_T256 |
|---|---|---|---|---|---|---|
| 2 | u16d_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 2 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 2 | u16fvt_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 2 | u16fvf_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 2 | u16fv_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 2 | u16fv_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 2 | u16fa_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 2 | i8d_dy | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | – | – |
| 2 | i8d_gd2 | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | – | – |

### B4
| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 4 | u16d_gd2 | 17.36 [14.58–17.86] | 17.54 [16.24–18.55] | 18.52 [15.19–32.11] |
| 4 | u16ft_st | 18.33 [12.98–21.42] | 17.64 [16.38–18.47] | 19.88 [15.42–40.81] |
| 4 | u16fvt_st | 20.82 [17.00–25.17] | 21.28 [19.54–22.59] | 22.63 [19.86–27.71] |
| 4 | u16fvf_st | 73.87 [46.36–406.71] | 59.56 [55.35–76.83] | 57.80 [52.60–70.80] |
| 4 | u16fv_dy | 21.55 [18.21–22.47] | 21.09 [19.35–25.61] | 22.09 [19.79–23.31] |
| 4 | u16fv_gd2 | 19.64 [17.81–21.55] | 21.87 [19.49–23.85] | 21.51 [20.09–22.41] |
| 4 | u16fa_st | 20.17 [19.55–21.68] | 20.60 [20.20–22.68] | 22.39 [18.93–25.14] |
| 4 | i8d_dy | 14.62 [12.11–16.70] | 14.03 [12.84–15.72] | 14.78 [12.07–16.30] |
| 4 | i8d_gd2 | 12.14 [9.68–12.84] | 12.20 [11.41–13.84] | 12.59 [10.81–13.59] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 4 | u16ft/best_new | 0.933x [0.678–0.994] (u16ft_st/u16fv_gd2) | 0.856x [0.766–0.909] (u16ft_st/u16fa_st) | 0.924x [0.757–1.933] (u16ft_st/u16fv_gd2) |
| 4 | u16ft/u16fvt_st | 0.880x [0.632–0.892] (u16ft_st/u16fvt_st) | 0.829x [0.751–0.910] (u16ft_st/u16fvt_st) | 0.878x [0.719–1.473] (u16ft_st/u16fvt_st) |
| 4 | u16ft/u16fvf_st | 0.248x [0.032–0.297] (u16ft_st/u16fvf_st) | 0.296x [0.221–0.317] (u16ft_st/u16fvf_st) | 0.344x [0.282–0.601] (u16ft_st/u16fvf_st) |
| 4 | u16ft/u16fv_dy | 0.850x [0.622–0.981] (u16ft_st/u16fv_dy) | 0.836x [0.717–0.955] (u16ft_st/u16fv_dy) | 0.900x [0.747–1.847] (u16ft_st/u16fv_dy) |
| 4 | u16ft/u16fv_gd2 | 0.933x [0.678–0.994] (u16ft_st/u16fv_gd2) | 0.806x [0.712–0.947] (u16ft_st/u16fv_gd2) | 0.924x [0.757–1.933] (u16ft_st/u16fv_gd2) |
| 4 | u16ft/u16fa_st | 0.909x [0.664–0.991] (u16ft_st/u16fa_st) | 0.856x [0.766–0.909] (u16ft_st/u16fa_st) | 0.888x [0.753–1.639] (u16ft_st/u16fa_st) |
| 4 | u16dT256/best_new | 0.884x [0.774–0.921] (u16d_gd2/u16fv_gd2) | 0.852x [0.772–0.918] (u16d_gd2/u16fa_st) | 0.861x [0.706–1.521] (u16d_gd2/u16fv_gd2) |
| 4 | best_new/best_i8d | 1.617x [1.495–1.840] (u16fv_gd2/i8d_gd2) | 1.689x [1.608–1.804] (u16fa_st/i8d_gd2) | 1.709x [1.572–1.963] (u16fv_gd2/i8d_gd2) |
| 4 | u16ft/best_i8d | 1.509x [1.087–1.743] (u16ft_st/i8d_gd2) | 1.446x [1.232–1.501] (u16ft_st/i8d_gd2) | 1.579x [1.426–3.242] (u16ft_st/i8d_gd2) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 4 | u16d_gd2 | 37.8/24.6, 45.0/29.6, 44.3/27.8 | 22.2/19.0, 26.8/22.4, 26.5/22.1 | 50.1/38.6, 53.4/40.7, 54.6/40.8 | 112.4/84.2, 127.0/94.5, 126.1/93.3 | 16.25, 18.47, 18.26 |
| 4 | u16ft_st | 33.5/30.1, 39.3/33.6, 41.8/35.4 | 19.9/17.6, 23.8/20.4, 23.7/20.3 | 46.3/37.9, 51.9/39.7, 56.4/44.3 | 102.8/86.7, 116.1/95.4, 122.7/101.9 | 17.03, 18.12, 20.28 |
| 4 | u16fvt_st | 82.7/70.1, 91.0/74.8, 86.6/68.9 | – | 46.7/38.3, 50.9/39.4, 48.6/37.4 | 134.2/110.3, 142.7/115.3, 136.4/107.6 | 21.08, 21.78, 21.75 |
| 4 | u16fvf_st | 235.3/207.2, 217.4/205.9, 225.3/206.2 | – | 48.8/42.9, 49.7/41.8, 49.1/41.5 | 285.7/252.0, 268.5/248.2, 275.4/249.0 | 66.54, 59.47, 59.39 |
| 4 | u16fv_dy | 79.1/67.2, 89.9/73.6, 89.0/72.1 | – | 48.7/41.4, 49.6/40.0, 50.3/40.4 | 131.0/110.3, 139.5/115.8, 141.3/114.5 | 20.22, 22.03, 22.26 |
| 4 | u16fv_gd2 | 72.6/62.6, 91.2/76.9, 95.0/74.7 | – | 48.2/37.4, 50.2/39.3, 54.8/40.6 | 122.4/101.0, 143.5/118.0, 153.5/116.9 | 17.31, 21.10, 22.02 |
| 4 | u16fa_st | 81.0/69.0, 84.7/70.2, 85.5/68.9 | – | 51.0/40.8, 49.2/38.0, 47.4/36.9 | 135.7/114.0, 135.5/109.9, 134.6/106.9 | 22.61, 20.50, 22.12 |

| B | config | max abs err vs ref | cos | instr identical | bit-identical to u16ft_st | bit-identical to u16d_T256 |
|---|---|---|---|---|---|---|
| 4 | u16d_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 4 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 4 | u16fvt_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 4 | u16fvf_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 4 | u16fv_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 4 | u16fv_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 4 | u16fa_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | True | True |
| 4 | i8d_dy | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | – | – |
| 4 | i8d_gd2 | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | – | – |

## Verdict

**not-viable.** Skipping the cast by taking f32 rows directly makes the expand slower, not faster: in the single-thread tile microbench, the f32 take costs 1291–1385 µs per 256-row tile vs 722–768 µs for u64 take + cast (1.7–1.8x slower) and 431–450 µs for the u64 take alone. The 210 KB LUT is not the problem. It fits in the 2 MiB L2, and restricting codes to 64 hot rows (LUT in L1) changes nothing for either LUT (f32 1279–1381 vs 1338–1352 µs random; u64 428–441 vs 443–451 µs). The cost is output bytes: the take now writes a 7.08 MB f32 tile (4 B/weight) through a 32-byte-per-element gather, vs 1.77 MB of u64 (1 B/weight) followed by a streaming SIMD cast. In the layer, take worker-ms goes from 28–42 (u64) + 18–24 (cast) to 70–95 (f32, no cast) at both B. So every f32 variant loses: best new is 0.84–0.89x (B 2) and 0.86–0.93x (B 4) of u16ft_st, the void and axis-0 takes are within noise of each other, and one f32 take over the whole worker range (48 MB scratch) collapses to 0.24–0.34x. The gap to int8d widens to 1.76–2.03x (B 2) and 1.62–1.71x (B 4); u16ft_st itself is 1.57–1.70x / 1.45–1.58x behind int8d this wave. Outputs are bit-identical to u16ft_st and u16d_T256. Ternary keeps only the known 3.98x weight-memory win (17.46 vs 69.56 MB). The f32 LUT adds 157 KB, which is harmless but buys nothing.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-27 05:55 wave reached this lane, and no transcript tool was available to read a Chief-named item, so this fire took the chained question from the 02:47 note. JuniorLLM HEAD at clone was 99e14d9 (cloudcover95, 02:52 MT, "rails/tickgap: offline tick-gap guard with selftest", added rails/tickgap/tick_gap_check.py only; unattributed), on top of 15c8c06 (this lane's 02:47 note). No Trit Pack, Handshake or LLC commit falls in the last 6 hours. This note only adds one new file under junior_osai/vault/findings/.

## Next question

The expand step is write-bound and the f32 tile (7.08 MB at 256 rows × in 6912) does not fit in the 2 MiB L2, so u16ft's take → cast → dot streams the tile through L3 three times. At B 2 / B 4 row-major, does a K-blocked u16ft help? It would run take + cast + per-b partial np.dot on 256-row × K-chunk sub-tiles (K = 864 / 1728 columns, f32 sub-tile 0.9–1.8 MB, L2-resident) and accumulate into y. The bar is >= 1.3x vs u16ft_st, and it must close the int8d gap to <= 1.3x. Run int8d with the same K-blocking as a control. (The 11-23 02:55 note K-blocked only the u32-LUT weight-major path at B 8 / B 32.)

## Script (v16fl.py)

```python
#!/usr/bin/env python3
"""V16FL: direct 6561 x 8 f32 LUT take into the f32 tile (no int8 -> f32 cast) for the 8-trit uint16 pack, row-major per-row np.dot,
vs u16ft_st (fused u64 take per worker range + 256-row cast + dot, 02:47 best) / u16d_T256 / int8d, at B2 / B4, W=8 decode layer, full bitnetd layer, ctx-free.
Base: /workspace/bench0255_1127/v16t.py (exec'd without main(); it execs v16d.py -> v16.py: same weights, P16 pack with exact trit
round-trip asserts, float64 refs, dy / gd2 schedules, r_u16d / r_u16ft kernels, profiling hooks).
  LUT16F = L16 as f32 (6561, 8) = 209,952 B; LUT16V = same bytes viewed as one 32-byte void element per code (1-D take, memcpy rows)
  u16fvt_st : static split: per 256-row tile np.take(LUT16V, P16 tile, out=void view of the f32 tile) -> per-b np.dot   (no cast)
  u16fvf_st : static split: ONE void take over the whole worker range into an f32 range buffer -> per-b np.dot per 256-row slice
  u16fv_gd2/dy : u16d schedule (T256 tiles from the gd2 / dy queue) with the void f32 take instead of u64 take + cast
  u16fa_st  : as u16fvt_st but np.take(LUT16F, idx, axis=0) on the 2-D f32 LUT (no void trick)
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16fl.py out.json"""
import json, os, sys
import numpy as np
_S = open("/workspace/bench0255_1127/v16t.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "v16t.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
LUT16F = np.ascontiguousarray(L16.astype(np.float32)); LUT16V = LUT16F.view("V32").ravel()
assert LUT16V.shape == (6561,) and LUT16F.nbytes == 209952 and np.array_equal(LUT16V.view(np.float32).reshape(6561, 8), L16.astype(np.float32))
def r_u16fvt(PR, x, y, r0, r1, T, w, i):
    q = PR.shape[1]; n = r1 - r0; P = PRF[0]; TT = 256
    s = scr(("f16", w, TT), (TT * F,), np.float32)[:TT * q * 8].reshape(TT, q * 8); sv = s.view("V32")
    for a in range(0, n, TT):
        m = min(TT, n - a)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16V, PR[r0+a:r0+a+m], out=sv[:m])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        for b in range(x.shape[0]): np.dot(s[:m], x[b], out=y[b, r0+a:r0+a+m])
        if P: _add(w, "dot", pc() - t0, tt() - c0)
def r_u16fvf(PR, x, y, r0, r1, T, w, i):
    q = PR.shape[1]; n = r1 - r0; P = PRF[0]; TT = 256
    s = scr(("f16f", w), (RMAX * F,), np.float32)[:n * q * 8].reshape(n, q * 8); sv = s.view("V32")
    if P: t0 = pc(); c0 = tt()
    np.take(LUT16V, PR[r0:r1], out=sv)
    if P: _add(w, "take", pc() - t0, tt() - c0)
    for a in range(0, n, TT):
        m = min(TT, n - a)
        if P: t0 = pc(); c0 = tt()
        for b in range(x.shape[0]): np.dot(s[a:a+m], x[b], out=y[b, r0+a:r0+a+m])
        if P: _add(w, "dot", pc() - t0, tt() - c0)
def r_u16fv(PR, x, y, r0, r1, T, w, i):
    q = PR.shape[1]; s = scr(("f16", w, T), (T * F,), np.float32)[:T * q * 8].reshape(T, q * 8); sv = s.view("V32"); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16V, PR[o0:o0+n], out=sv[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        for b in range(x.shape[0]): np.dot(s[:n], x[b], out=y[b, o0:o0+n])
        if P: _add(w, "dot", pc() - t0, tt() - c0)
def r_u16fa(PR, x, y, r0, r1, T, w, i):
    q = PR.shape[1]; n = r1 - r0; P = PRF[0]; TT = 256
    s = scr(("f16", w, TT), (TT * F,), np.float32)[:TT * q * 8].reshape(TT, q * 8); s3 = s.reshape(TT, q, 8)
    for a in range(0, n, TT):
        m = min(TT, n - a)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16F, PR[r0+a:r0+a+m], axis=0, out=s3[:m])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        for b in range(x.shape[0]): np.dot(s[:m], x[b], out=y[b, r0+a:r0+a+m])
        if P: _add(w, "dot", pc() - t0, tt() - c0)
KD["u16fvt"] = r_u16fvt; KD["u16fvf"] = r_u16fvf; KD["u16fv"] = r_u16fv; KD["u16fa"] = r_u16fa
C = {"u16d_gd2": ("u16d", 256, "gd2"), "u16ft_st": ("u16ft", 8, "st"), "u16fvt_st": ("u16fvt", 8, "st"), "u16fvf_st": ("u16fvf", 8, "st"),
     "u16fv_dy": ("u16fv", 256, "dy"), "u16fv_gd2": ("u16fv", 256, "gd2"), "u16fa_st": ("u16fa", 8, "st"),
     "i8d_dy": ("8d", 128, "dy"), "i8d_gd2": ("8d", 128, "gd2")}
U = [c for c in C if C[c][0] in KD]; BASE = "u16ft_st"; NEW = ["u16fvt_st", "u16fvf_st", "u16fv_dy", "u16fv_gd2", "u16fa_st"]; I8 = ["i8d_dy", "i8d_gd2"]
def micro(seed):
    """single-thread tile microbench on the largest site (in = F): u64 take + cast vs direct f32 void take vs 2-D f32 take, 256 rows."""
    k = max(SITES, key=lambda k: P16[k]["PR"].shape[1]); PR = P16[k]["PR"][:256]; q = PR.shape[1]
    su = np.empty((256, q), np.uint64); s = np.empty((256, q * 8), np.float32); sv = s.view("V32"); s3 = s.reshape(256, q, 8); out = {}
    def t_(f, n=41):
        f(); v = []
        for _ in range(n): t0 = pc(); f(); v.append((pc() - t0) * 1e6)
        return dict(med=float(np.median(v)), lo=float(min(v)), hi=float(max(v)))
    out["site"] = k; out["q8"] = q; out["tile_bytes_f32"] = s.nbytes; out["tile_bytes_u64"] = su.nbytes
    out["u64_take"] = t_(lambda: np.take(LUT16U, PR, out=su))
    out["u64_take_cast"] = t_(lambda: (np.take(LUT16U, PR, out=su), np.copyto(s, su.view(np.int8).reshape(256, q * 8), casting="unsafe")))
    out["f32_void_take"] = t_(lambda: np.take(LUT16V, PR, out=sv))
    out["f32_axis0_take"] = t_(lambda: np.take(LUT16F, PR, axis=0, out=s3))
    # cache sensitivity: random codes (whole LUT touched) vs codes restricted to 64 rows (LUT hot in L1)
    rng = np.random.default_rng(seed); Rr = rng.integers(0, 6561, PR.shape).astype(np.uint16); Rh = rng.integers(0, 64, PR.shape).astype(np.uint16)
    out["f32_void_take_random"] = t_(lambda: np.take(LUT16V, Rr, out=sv)); out["f32_void_take_hot64"] = t_(lambda: np.take(LUT16V, Rh, out=sv))
    out["u64_take_random"] = t_(lambda: np.take(LUT16U, Rr, out=su)); out["u64_take_hot64"] = t_(lambda: np.take(LUT16U, Rh, out=su))
    np.take(LUT16U, PR, out=su); s2 = su.view(np.int8).reshape(256, q * 8).astype(np.float32); np.take(LUT16V, PR, out=sv); assert np.array_equal(s, s2)
    return out
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, RMAX=RMAX, time={}, prof={}, err={})
    W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES); W16 = sum(P16[k]["PR"].nbytes + P16[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(u16=W16, int8=W8, int8_over_u16=W8 / W16, lut_u64=LUT16U.nbytes, lut_f32=LUT16F.nbytes,
                        scratch_u16ft_per_worker=RMAX * Q8 * 8 + 256 * F * 4, scratch_u16fvt_per_worker=256 * F * 4, scratch_u16fvf_per_worker=RMAX * F * 4)
    print("bytes", res["bytes"], flush=True)
    res["micro"] = micro(seed); print("micro", {k: (round(v["med"], 1) if isinstance(v, dict) else v) for k, v in res["micro"].items()}, flush=True)
    names = list(C)
    for B in (2, 4):
        xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0] == "8d" else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in U: e[c]["bitident_u16ft"] = bool(np.array_equal(outs[c], outs["u16ft_st"])); e[c]["bitident_u16d"] = bool(np.array_equal(outs[c], outs["u16d_gd2"]))
        res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.12f}", v["instr_identical"], v.get("bitident_u16ft")) for c, v in e.items()}, flush=True)
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
        bn = min(NEW, key=lambda c: o[c]["med"]); bi = min(I8, key=lambda c: o[c]["med"])
        pr_(BASE, bn, "u16ft/best_new")
        for c in NEW: pr_(BASE, c, f"u16ft/{c}")
        pr_("u16d_gd2", bn, "u16dT256/best_new"); pr_(bn, bi, "best_new/best_i8d"); pr_(BASE, bi, "u16ft/best_i8d")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:13s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
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

close: 3156710abaddf4b1
