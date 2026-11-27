---
local: true
date: 2026-11-27
wave: "2026-11-27 10:55 slot (fired 10:49 MT, ran ~10:51–10:55 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (column-chunk (K-blocked) u16ft on the W 8 decode layer, full bitnetd layer, ctx-free, at B 2 / B 4: one fused u64 take per static worker range, then cast + per-b partial np.dot on 256-row x Kc sub-tiles (Kc 864 / 1728 trit columns = 108 / 216 u16 codes, f32 sub-tile 0.88 / 1.77 MB, inside the 2 MiB L2) accumulated into y. Every variant is SLOWER than u16ft_st: K-blocking all sites 0.65–0.72x (Kc 864) / 0.81–0.89x (Kc 1728) at B 2 and 0.60–0.64x / 0.75–0.82x at B 4; per-sub-tile take 0.53–0.76x; best is K-blocking only the in-6912 down site at Kc 1728: 0.95–0.99x (B 2) / 0.86–0.97x (B 4) (bar >= 1.3x: missed). Single-thread the same down tile IS faster K-blocked: 256 x 6912 cast + 2 dots 747–777 us unsplit vs 385–406 us at Kc 864 (1.9x) and 472–496 us at Kc 1728 (1.6x), but in the 8-thread pool cast + dot wall-ms rise 1.3–1.9x while their CPU-ms stay flat, so the extra numpy calls go to off-CPU wait, not to cache misses. The int8d control loses the same way: K-blocked best 0.80–0.90x of unsplit int8d. Gap to unsplit int8d: best new 1.53–1.59x (B 2) / 1.48–1.54x (B 4); u16ft_st 1.45–1.57x / 1.32–1.47x this wave. Error: K-split reorders the f32 sums, so outputs are not bit-identical to u16ft_st (max diff <= 1.1e-6) but stay <= 1.12e-6 vs the float64 ternary reference (cos 1.0); trits exact (pack round-trip asserted). Memory unchanged: 17.46 MB vs 69.56 MB int8 (3.98x, the known weight-memory win); per-worker scratch drops 19.08 -> 12.88–13.77 MB (fused take) or 1.11–2.21 MB (per-sub-tile take). No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, uint16, eight-trit-pack, base-3, 2-bit, np-take, per-row-dot, row-major, fused-take, static-split, k-blocking, column-chunk, cache, l2, gil, int8, numpy, not-viable]
---

# Column-chunk (K-blocked) u16ft: 1.9x faster single-thread, 0.53–0.99x in the W 8 pool, gap to int8d stays ~1.5x

The 05:50 note ([[2026-11-27_0550_direct-6561x8-f32-lut-take-into-f32-tile-no-cast-fused-per-worker-range-and-t256-vs-u16ft-st-and-int8d-w8-pool-b2-b4-full-bitnetd-layer-slower-0p84-0p93x-write-bound-gap-int8d-1p6-2p0x]]) found the u16 expand write-bound: u16ft's 256-row f32 tile (7.08 MB at in 6912, 2.6 MB at in 2560) does not fit in the 2 MiB L2, so take → cast → dot streams it through L3. It asked whether a K-blocked u16ft, with cast + per-b partial np.dot on 256-row × K-chunk sub-tiles (K 864 / 1728, f32 sub-tile 0.9–1.8 MB, L2-resident) accumulated into y, beats u16ft_st by >= 1.3x and closes the int8d gap to <= 1.3x, with int8d K-blocked the same way as a control.

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 2 and B 4, does column-chunking u16ft (fused take per worker range, T 256 rows × Kc 864 / 1728 columns per sub-tile) beat u16ft_st by >= 1.3x? Where does it land against int8d, both unsplit and split the same way?

## Method

- Box: 8 CPUs (Xeon, L2 2 MiB per core, L3 320 MiB), numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents (load avg 1–3.4 during the runs), so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench0255_1127/v16t.py exec'd without main() (it execs v16d.py → v16.py: same weights, P16 8-trit uint16 pack with exact trit round-trip asserts, float64 references, dy / gd2 schedules, r_u16ft kernel, profiling hooks).
- Chunk sizes: Kc = 864 and 1728 trit columns are both multiples of 8, so they align with the 8-trit u16 groups (108 / 216 codes) and needed no adjustment. The in-2560 sites (qkv / o / gu) split 864 + 864 + 832 and 1728 + 832 (the last chunk is ragged). The in-6912 down site splits into 8 × 864 or 4 × 1728. f32 sub-tiles are 884,736 B and 1,769,472 B.
- Kernels (row-major x (B, in) → y (B, out); chunk 0 does np.dot into y, later chunks np.dot into a 256-float tmp, then np.add into y):
  - **u16ftk{864,1728}_st**: static split; ONE u64 LUT take over the whole worker range (as u16ft), then per 256-row tile, per K chunk: cast the strided int8 view into a contiguous (256, Kc) f32 sub-tile → per-b partial np.dot.
  - **u16dk{864,1728}_st**: static split; per 256-row × Kc sub-tile: take (strided P16 slice → contiguous u64 sub-tile) → cast → per-b partial dot. All scratch stays in L2.
  - **u16ftk{864,1728}dn_st**: as u16ftk but only the down site (in 6912, 7.08 MB tile) is K-blocked; the in-2560 sites run unsplit.
  - int8d control: **i8dk{864,1728}_gd2** (T 128 gd2 schedule) and **i8dk{864,1728}_st** (256-row static) K-blocked over all sites, plus **i8dk{864,1728}dn_gd2** (down site only). Unsplit baselines: **u16ft_st** (02:47 best), **i8d_dy / i8d_gd2** (T 128).
- Microbench (single thread, down site, 256 rows, B 2): unsplit cast of the 256 × 6912 tile + 2 np.dot, vs the K-blocked cast + partial dots at Kc 864 / 1728; 41 reps, median [min–max].
- Error: max abs error and cos vs the float64 ternary (u16) or int8 reference. Each run also checks that instrumented and plain output are identical, and records max abs diff and np.array_equal vs the unsplit kernel (u16ft_st / i8d_gd2).
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round. Reported as median of round medians, round min–max and per-round ratios. Then 15 instrumented layers per config give stage worker-ms (wall/cpu summed over workers) and per-site span. Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench1055_1127 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16k.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Bytes: weights u16 17,458,176 B vs int8 69,559,296 B (3.98x); LUT u64 52,488 B; per-worker scratch u16ft 19,077,120 B, u16ftk 12,883,968 / 13,768,704 B, u16dk 1,105,920 / 2,211,840 B. Below: the microbench, layer ms (median of round medians [round min–max], runs 1 / 2 / 3), ratios, stage worker-ms (wall/cpu, runs 1, 2, 3) and error.

bytes {'u16': 17458176, 'int8': 69559296, 'int8_over_u16': 3.9843392574344536, 'lut_u64': 52488, 'scratch_u16ft_per_worker': 19077120, 'scratch_u16ftk864_per_worker': 12883968, 'scratch_u16ftk1728_per_worker': 13768704, 'scratch_u16dk864_per_worker': 1105920, 'scratch_u16dk1728_per_worker': 2211840, 'subtile_f32_864': 884736, 'subtile_f32_1728': 1769472}

Single-thread 256-row tile microbench, site down (in 6912 ), B 2, cast + 2 per-row dots, µs median [min–max], runs 1 / 2 / 3:

| op | run 1 | run 2 | run 3 |
|---|---|---|---|
| full_cast_dot | 768 [731–954] | 747 [737–868] | 777 [741–931] |
| k864_cast_dot | 406 [381–508] | 385 [375–502] | 398 [384–508] |
| k864_maxdiff_vs_full | 8.1e-05 | 8.0e-05 | 7.2e-05 |
| k1728_cast_dot | 472 [450–550] | 474 [459–597] | 496 [450–592] |
| k1728_maxdiff_vs_full | 7.2e-05 | 8.8e-05 | 8.8e-05 |

### B2
| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | u16ft_st | 10.52 [9.34–14.10] | 13.77 [10.57–14.73] | 10.46 [9.12–14.43] |
| 2 | u16ftk864_st | 16.24 [13.64–17.31] | 19.06 [16.60–20.70] | 15.31 [14.75–17.09] |
| 2 | u16ftk1728_st | 12.93 [12.41–14.98] | 15.46 [13.80–17.31] | 12.72 [12.18–14.14] |
| 2 | u16dk864_st | 18.41 [17.45–19.86] | 24.48 [20.70–26.47] | 18.54 [18.26–20.66] |
| 2 | u16dk1728_st | 14.89 [13.85–18.08] | 19.08 [15.86–20.67] | 15.10 [14.56–16.57] |
| 2 | u16ftk864dn_st | 11.92 [11.44–13.01] | 15.59 [12.54–16.86] | 12.62 [11.93–13.56] |
| 2 | u16ftk1728dn_st | 11.07 [9.73–12.36] | 13.96 [11.74–14.90] | 11.02 [10.18–14.54] |
| 2 | i8d_dy | 7.84 [7.28–9.02] | 9.62 [8.58–10.76] | 7.76 [7.43–8.26] |
| 2 | i8d_gd2 | 7.25 [6.76–7.95] | 8.78 [7.27–9.35] | 7.07 [6.76–7.72] |
| 2 | i8dk864_gd2 | 14.09 [13.33–15.77] | 18.16 [15.03–19.27] | 14.52 [13.62–14.99] |
| 2 | i8dk1728_gd2 | 11.01 [10.60–11.88] | 13.68 [11.46–14.06] | 10.85 [10.52–11.77] |
| 2 | i8dk864_st | 10.35 [9.70–11.95] | 12.49 [10.71–15.04] | 10.15 [9.52–11.92] |
| 2 | i8dk1728_st | 8.46 [8.00–10.83] | 10.31 [8.99–11.11] | 8.72 [8.21–9.88] |
| 2 | i8dk864dn_gd2 | 9.47 [9.09–11.04] | 11.52 [10.57–12.96] | 9.75 [9.45–10.52] |
| 2 | i8dk1728dn_gd2 | 8.09 [7.88–9.46] | 10.04 [9.08–10.61] | 8.22 [7.81–8.86] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | u16ft/best_new | 0.951x [0.808–1.254] (u16ft_st/u16ftk1728dn_st) | 0.986x [0.767–1.021] (u16ft_st/u16ftk1728dn_st) | 0.950x [0.828–0.992] (u16ft_st/u16ftk1728dn_st) |
| 2 | u16ft/u16ftk864_st | 0.648x [0.598–0.866] (u16ft_st/u16ftk864_st) | 0.722x [0.637–0.742] (u16ft_st/u16ftk864_st) | 0.683x [0.571–0.844] (u16ft_st/u16ftk864_st) |
| 2 | u16ft/u16ftk1728_st | 0.814x [0.667–1.137] (u16ft_st/u16ftk1728_st) | 0.890x [0.766–0.953] (u16ft_st/u16ftk1728_st) | 0.822x [0.712–1.021] (u16ft_st/u16ftk1728_st) |
| 2 | u16ft/u16dk864_st | 0.572x [0.488–0.781] (u16ft_st/u16dk864_st) | 0.562x [0.426–0.603] (u16ft_st/u16dk864_st) | 0.564x [0.485–0.698] (u16ft_st/u16dk864_st) |
| 2 | u16ft/u16dk1728_st | 0.707x [0.516–0.965] (u16ft_st/u16dk1728_st) | 0.722x [0.657–0.754] (u16ft_st/u16dk1728_st) | 0.693x [0.568–0.956] (u16ft_st/u16dk1728_st) |
| 2 | u16ft/u16ftk864dn_st | 0.883x [0.769–1.182] (u16ft_st/u16ftk864dn_st) | 0.883x [0.678–0.957] (u16ft_st/u16ftk864dn_st) | 0.829x [0.731–1.136] (u16ft_st/u16ftk864dn_st) |
| 2 | u16ft/u16ftk1728dn_st | 0.951x [0.808–1.254] (u16ft_st/u16ftk1728dn_st) | 0.986x [0.767–1.021] (u16ft_st/u16ftk1728dn_st) | 0.950x [0.828–0.992] (u16ft_st/u16ftk1728dn_st) |
| 2 | best_new/best_i8d | 1.527x [1.224–1.663] (u16ftk1728dn_st/i8d_gd2) | 1.590x [1.542–1.898] (u16ftk1728dn_st/i8d_gd2) | 1.558x [1.434–1.884] (u16ftk1728dn_st/i8d_gd2) |
| 2 | u16ft/best_i8d | 1.451x [1.309–2.085] (u16ft_st/i8d_gd2) | 1.568x [1.455–1.611] (u16ft_st/i8d_gd2) | 1.480x [1.186–1.869] (u16ft_st/i8d_gd2) |
| 2 | best_i8d/best_i8dk | 0.897x [0.714–1.002] (i8d_gd2/i8dk1728dn_gd2) | 0.875x [0.800–0.921] (i8d_gd2/i8dk1728dn_gd2) | 0.860x [0.815–0.946] (i8d_gd2/i8dk1728dn_gd2) |
| 2 | best_new/best_i8dk | 1.369x [1.187–1.527] (u16ftk1728dn_st/i8dk1728dn_gd2) | 1.391x [1.266–1.518] (u16ftk1728dn_st/i8dk1728dn_gd2) | 1.341x [1.243–1.641] (u16ftk1728dn_st/i8dk1728dn_gd2) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 2 | u16ft_st | 22.2/19.3, 22.8/18.0, 23.1/19.7 | 15.8/13.4, 16.3/12.7, 16.5/13.6 | 23.1/17.9, 23.9/17.6, 24.3/18.4 | 62.1/51.9, 64.3/48.6, 65.6/52.9 | 9.78, 10.56, 10.58 |
| 2 | u16ftk864_st | 22.0/18.5, 22.9/17.8, 25.8/21.6 | 38.3/13.0, 39.9/16.8, 40.2/15.3 | 49.2/12.7, 58.0/17.6, 56.2/15.3 | 112.3/46.8, 124.0/54.8, 126.2/54.1 | 15.76, 17.42, 18.03 |
| 2 | u16ftk1728_st | 22.1/19.0, 22.6/17.9, 27.0/23.2 | 26.9/13.0, 29.7/15.2, 29.0/15.0 | 42.6/14.1, 46.4/17.1, 44.2/16.9 | 94.9/48.0, 101.5/52.5, 102.9/58.1 | 13.28, 14.36, 15.64 |
| 2 | u16dk864_st | 67.5/22.2, 72.7/25.0, 70.1/23.5 | 22.5/12.6, 25.2/14.5, 23.5/13.1 | 35.3/12.9, 39.4/15.3, 36.7/13.6 | 128.3/50.6, 141.2/57.8, 132.1/53.2 | 20.64, 21.85, 21.19 |
| 2 | u16dk1728_st | 48.1/21.0, 52.4/22.8, 53.7/23.0 | 19.7/13.2, 21.8/14.1, 20.8/14.2 | 27.7/13.9, 30.9/15.4, 30.9/15.6 | 98.3/50.4, 107.9/54.5, 108.2/55.5 | 16.11, 17.02, 17.96 |
| 2 | u16ftk864dn_st | 22.6/19.3, 24.6/19.1, 22.7/19.1 | 25.1/13.2, 26.8/14.6, 25.5/13.5 | 34.2/16.9, 38.5/18.7, 35.2/17.2 | 84.9/51.5, 92.3/54.4, 85.7/52.1 | 12.46, 13.96, 12.63 |
| 2 | u16ftk1728dn_st | 22.7/19.4, 23.3/18.4, 25.7/21.7 | 20.1/13.2, 21.4/13.6, 20.7/14.3 | 33.0/17.4, 32.8/17.9, 32.8/18.3 | 78.3/52.0, 80.5/51.9, 83.0/59.6 | 11.37, 12.25, 13.29 |
| 2 | i8dk864_gd2 | – | 44.4/17.3, 48.0/16.6, 45.8/18.2 | 64.2/15.6, 72.6/19.0, 69.6/16.4 | 112.3/35.4, 124.2/38.5, 118.6/38.2 | 15.71, 17.02, 16.45 |
| 2 | i8dk1728_gd2 | – | 34.0/16.7, 33.9/15.5, 30.8/14.6 | 54.4/16.3, 55.9/18.7, 47.3/14.7 | 91.6/35.2, 92.5/36.7, 80.7/31.4 | 12.86, 12.96, 11.33 |
| 2 | i8dk864_st | – | 29.4/14.9, 29.1/14.8, 26.9/12.9 | 42.5/14.4, 43.7/15.4, 36.2/11.9 | 75.0/31.5, 75.2/32.4, 65.2/26.9 | 12.66, 12.43, 10.97 |
| 2 | i8dk1728_st | – | 22.5/14.1, 23.5/14.8, 21.6/13.2 | 32.1/15.1, 34.1/16.3, 30.0/13.5 | 56.9/31.0, 60.0/33.1, 53.4/28.3 | 9.87, 9.91, 9.09 |
| 2 | i8dk864dn_gd2 | – | 29.4/13.6, 31.9/14.4, 29.1/14.2 | 41.4/17.2, 44.7/18.8, 42.0/17.8 | 72.8/32.8, 80.6/35.6, 73.3/34.2 | 10.33, 11.31, 10.43 |
| 2 | i8dk1728dn_gd2 | – | 23.2/13.9, 24.4/13.7, 23.8/14.4 | 39.3/18.6, 40.1/18.0, 37.3/17.7 | 64.7/34.6, 66.7/33.7, 63.5/34.0 | 9.25, 9.51, 9.11 |

| B | config | max abs err vs ref | cos | instr identical | max abs diff vs unsplit (u16ft_st / i8d_gd2) | bit-identical to unsplit |
|---|---|---|---|---|---|---|
| 2 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 0.0e+00 | True |
| 2 | u16ftk864_st | 5.61e-07, 6.23e-07, 5.88e-07 | 1.000000000000 | True | 1.1e-06 | False |
| 2 | u16ftk1728_st | 8.06e-07, 7.07e-07, 7.20e-07 | 1.000000000000 | True | 1.0e-06 | False |
| 2 | u16dk864_st | 5.61e-07, 6.23e-07, 5.88e-07 | 1.000000000000 | True | 1.1e-06 | False |
| 2 | u16dk1728_st | 8.06e-07, 7.07e-07, 7.20e-07 | 1.000000000000 | True | 1.0e-06 | False |
| 2 | u16ftk864dn_st | 1.00e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 5.4e-07 | False |
| 2 | u16ftk1728dn_st | 1.12e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 4.8e-07 | False |
| 2 | i8d_dy | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | – | – |
| 2 | i8d_gd2 | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | – | – |
| 2 | i8dk864_gd2 | 2.25e-06, 2.33e-06, 2.15e-06 | 1.000000000000 | True | 4.0e-06 | False |
| 2 | i8dk1728_gd2 | 2.84e-06, 2.66e-06, 2.69e-06 | 1.000000000000 | True | 3.8e-06 | False |
| 2 | i8dk864_st | 2.25e-06, 2.33e-06, 2.15e-06 | 1.000000000000 | True | 4.0e-06 | False |
| 2 | i8dk1728_st | 2.84e-06, 2.66e-06, 2.69e-06 | 1.000000000000 | True | 3.8e-06 | False |
| 2 | i8dk864dn_gd2 | 3.46e-06, 3.41e-06, 3.25e-06 | 1.000000000000 | True | 2.3e-06 | False |
| 2 | i8dk1728dn_gd2 | 3.46e-06, 3.41e-06, 3.49e-06 | 1.000000000000 | True | 1.9e-06 | False |

### B4
| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 4 | u16ft_st | 15.24 [11.90–16.55] | 13.67 [13.02–15.87] | 12.86 [12.02–14.82] |
| 4 | u16ftk864_st | 23.64 [19.08–26.96] | 22.69 [21.46–24.05] | 20.30 [19.76–21.98] |
| 4 | u16ftk1728_st | 18.66 [15.76–21.69] | 18.15 [17.03–18.86] | 16.20 [15.73–17.31] |
| 4 | u16dk864_st | 23.99 [21.56–32.09] | 25.58 [24.89–28.53] | 23.06 [21.39–29.87] |
| 4 | u16dk1728_st | 20.04 [17.69–23.73] | 19.73 [18.56–21.72] | 18.42 [17.13–19.61] |
| 4 | u16ftk864dn_st | 17.24 [14.60–20.54] | 17.33 [16.75–18.44] | 16.22 [14.45–17.52] |
| 4 | u16ftk1728dn_st | 15.77 [13.01–17.04] | 15.93 [14.60–15.95] | 14.39 [12.71–17.83] |
| 4 | i8d_dy | 11.59 [10.32–12.98] | 12.78 [11.31–13.17] | 10.80 [10.30–12.06] |
| 4 | i8d_gd2 | 10.35 [9.02–11.18] | 10.34 [9.56–11.59] | 9.70 [8.91–11.56] |
| 4 | i8dk864_gd2 | 23.16 [20.28–23.63] | 22.05 [21.57–25.29] | 20.65 [19.52–23.25] |
| 4 | i8dk1728_gd2 | 16.67 [15.23–18.27] | 16.92 [16.65–19.19] | 14.89 [14.56–16.59] |
| 4 | i8dk864_st | 16.17 [13.61–17.00] | 15.41 [14.77–17.98] | 14.96 [13.76–15.73] |
| 4 | i8dk1728_st | 13.60 [11.08–14.40] | 12.70 [12.31–14.39] | 11.50 [10.58–12.37] |
| 4 | i8dk864dn_gd2 | 14.97 [12.92–15.81] | 14.76 [14.38–16.20] | 13.52 [12.72–14.82] |
| 4 | i8dk1728dn_gd2 | 12.97 [10.92–17.69] | 12.33 [12.19–13.49] | 11.84 [10.65–12.84] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 4 | u16ft/best_new | 0.967x [0.869–1.152] (u16ft_st/u16ftk1728dn_st) | 0.858x [0.817–1.002] (u16ft_st/u16ftk1728dn_st) | 0.894x [0.739–1.140] (u16ft_st/u16ftk1728dn_st) |
| 4 | u16ft/u16ftk864_st | 0.645x [0.608–0.662] (u16ft_st/u16ftk864_st) | 0.602x [0.563–0.668] (u16ft_st/u16ftk864_st) | 0.634x [0.607–0.674] (u16ft_st/u16ftk864_st) |
| 4 | u16ft/u16ftk1728_st | 0.817x [0.719–1.040] (u16ft_st/u16ftk1728_st) | 0.753x [0.703–0.856] (u16ft_st/u16ftk1728_st) | 0.794x [0.717–0.942] (u16ft_st/u16ftk1728_st) |
| 4 | u16ft/u16dk864_st | 0.635x [0.516–0.751] (u16ft_st/u16dk864_st) | 0.534x [0.465–0.620] (u16ft_st/u16dk864_st) | 0.558x [0.441–0.693] (u16ft_st/u16dk864_st) |
| 4 | u16ft/u16dk1728_st | 0.761x [0.641–0.847] (u16ft_st/u16dk1728_st) | 0.693x [0.600–0.855] (u16ft_st/u16dk1728_st) | 0.698x [0.633–0.785] (u16ft_st/u16dk1728_st) |
| 4 | u16ft/u16ftk864dn_st | 0.884x [0.742–1.088] (u16ft_st/u16ftk864dn_st) | 0.789x [0.719–0.903] (u16ft_st/u16ftk864dn_st) | 0.793x [0.711–1.025] (u16ft_st/u16ftk864dn_st) |
| 4 | u16ft/u16ftk1728dn_st | 0.967x [0.869–1.152] (u16ft_st/u16ftk1728dn_st) | 0.858x [0.817–1.002] (u16ft_st/u16ftk1728dn_st) | 0.894x [0.739–1.140] (u16ft_st/u16ftk1728dn_st) |
| 4 | best_new/best_i8d | 1.523x [1.374–1.548] (u16ftk1728dn_st/i8d_gd2) | 1.541x [1.374–1.544] (u16ftk1728dn_st/i8d_gd2) | 1.484x [1.323–1.648] (u16ftk1728dn_st/i8d_gd2) |
| 4 | u16ft/best_i8d | 1.472x [1.309–1.583] (u16ft_st/i8d_gd2) | 1.322x [1.257–1.513] (u16ft_st/i8d_gd2) | 1.326x [1.140–1.663] (u16ft_st/i8d_gd2) |
| 4 | best_i8d/best_i8dk | 0.798x [0.585–0.896] (i8d_gd2/i8dk1728dn_gd2) | 0.838x [0.731–0.874] (i8d_gd2/i8dk1728dn_gd2) | 0.843x [0.775–0.955] (i8d_gd2/i8dk1728_st) |
| 4 | best_new/best_i8dk | 1.216x [0.804–1.356] (u16ftk1728dn_st/i8dk1728dn_gd2) | 1.292x [1.114–1.303] (u16ftk1728dn_st/i8dk1728dn_gd2) | 1.251x [1.130–1.473] (u16ftk1728dn_st/i8dk1728_st) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 4 | u16ft_st | 22.9/19.3, 26.4/19.1, 22.0/18.1 | 15.9/13.6, 16.9/13.1, 15.2/12.8 | 42.5/34.0, 47.3/33.5, 42.6/34.1 | 82.6/68.6, 91.9/67.5, 82.3/68.0 | 12.78, 16.08, 12.03 |
| 4 | u16ftk864_st | 28.0/24.4, 26.6/19.3, 23.6/19.0 | 40.1/16.5, 40.7/18.3, 37.3/13.9 | 99.0/27.3, 111.2/35.6, 93.3/25.9 | 171.2/71.1, 180.9/76.3, 158.1/61.6 | 25.01, 25.45, 21.99 |
| 4 | u16ftk1728_st | 28.5/25.1, 25.6/18.1, 23.4/19.9 | 28.2/16.6, 28.1/15.6, 26.1/14.3 | 76.6/28.4, 82.0/32.5, 77.8/27.6 | 136.6/73.0, 139.5/68.2, 132.0/63.2 | 20.12, 19.76, 18.22 |
| 4 | u16dk864_st | 81.1/26.0, 78.3/28.1, 80.1/26.2 | 26.3/14.3, 27.6/16.3, 25.7/14.2 | 76.6/26.0, 80.1/31.2, 76.6/27.4 | 190.1/69.1, 189.0/79.3, 185.6/71.0 | 30.65, 29.38, 29.66 |
| 4 | u16dk1728_st | 57.7/23.9, 56.0/24.3, 49.7/21.5 | 22.6/14.9, 22.7/15.2, 20.0/13.4 | 58.0/28.0, 60.0/30.3, 53.7/25.8 | 141.5/70.7, 141.9/72.3, 126.4/63.3 | 23.22, 22.37, 20.02 |
| 4 | u16ftk864dn_st | 30.1/26.0, 25.7/18.0, 26.1/22.3 | 27.1/16.1, 25.4/14.2, 26.7/14.8 | 65.6/33.9, 69.1/33.9, 68.4/34.8 | 125.3/77.3, 124.1/67.9, 124.1/74.8 | 19.36, 18.67, 18.35 |
| 4 | u16ftk1728dn_st | 28.9/25.1, 27.2/18.8, 23.7/20.8 | 20.6/15.2, 20.1/13.5, 19.7/13.4 | 56.3/33.0, 58.4/32.3, 57.9/33.3 | 108.2/74.9, 106.4/67.1, 104.5/69.5 | 17.04, 16.88, 14.92 |
| 4 | i8dk864_gd2 | – | 51.6/20.6, 45.9/18.2, 45.6/19.3 | 136.5/32.7, 126.9/35.5, 122.7/30.2 | 188.9/56.9, 175.8/56.9, 171.3/52.3 | 26.10, 23.67, 23.34 |
| 4 | i8dk1728_gd2 | – | 35.1/18.3, 31.5/15.7, 34.2/17.7 | 99.2/30.7, 97.6/34.0, 95.0/30.6 | 137.9/51.5, 130.7/52.1, 131.6/51.0 | 19.11, 17.94, 18.27 |
| 4 | i8dk864_st | – | 29.6/15.0, 28.1/15.1, 32.9/17.5 | 76.0/24.2, 76.9/28.2, 84.5/27.2 | 108.0/41.4, 107.8/45.3, 120.6/47.4 | 17.50, 16.97, 19.71 |
| 4 | i8dk1728_st | – | 23.5/15.0, 21.6/13.6, 22.6/14.4 | 57.8/26.1, 58.7/28.1, 58.8/26.8 | 83.6/43.1, 82.8/43.6, 84.8/43.0 | 14.07, 13.45, 14.13 |
| 4 | i8dk864dn_gd2 | – | 32.5/17.0, 29.0/14.5, 31.5/14.9 | 82.3/35.8, 81.1/34.2, 78.2/35.5 | 118.4/55.3, 112.7/51.1, 111.8/52.9 | 16.41, 15.57, 15.77 |
| 4 | i8dk1728dn_gd2 | – | 24.9/16.1, 23.2/13.3, 22.7/14.0 | 69.8/34.3, 72.9/33.7, 66.2/33.5 | 96.5/53.0, 97.8/49.0, 91.6/49.3 | 13.70, 13.60, 12.92 |

| B | config | max abs err vs ref | cos | instr identical | max abs diff vs unsplit (u16ft_st / i8d_gd2) | bit-identical to unsplit |
|---|---|---|---|---|---|---|
| 4 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 0.0e+00 | True |
| 4 | u16ftk864_st | 6.57e-07, 6.23e-07, 5.88e-07 | 1.000000000000 | True | 1.1e-06 | False |
| 4 | u16ftk1728_st | 8.06e-07, 7.62e-07, 8.02e-07 | 1.000000000000 | True | 1.1e-06 | False |
| 4 | u16dk864_st | 6.57e-07, 6.23e-07, 5.88e-07 | 1.000000000000 | True | 1.1e-06 | False |
| 4 | u16dk1728_st | 8.06e-07, 7.62e-07, 8.02e-07 | 1.000000000000 | True | 1.1e-06 | False |
| 4 | u16ftk864dn_st | 1.11e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 5.4e-07 | False |
| 4 | u16ftk1728dn_st | 1.12e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 4.8e-07 | False |
| 4 | i8d_dy | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | – | – |
| 4 | i8d_gd2 | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | – | – |
| 4 | i8dk864_gd2 | 2.25e-06, 2.33e-06, 2.15e-06 | 1.000000000000 | True | 4.3e-06 | False |
| 4 | i8dk1728_gd2 | 2.84e-06, 2.68e-06, 2.71e-06 | 1.000000000000 | True | 4.3e-06 | False |
| 4 | i8dk864_st | 2.25e-06, 2.33e-06, 2.15e-06 | 1.000000000000 | True | 4.3e-06 | False |
| 4 | i8dk1728_st | 2.84e-06, 2.68e-06, 2.71e-06 | 1.000000000000 | True | 4.3e-06 | False |
| 4 | i8dk864dn_gd2 | 3.46e-06, 3.41e-06, 3.30e-06 | 1.000000000000 | True | 2.3e-06 | False |
| 4 | i8dk1728dn_gd2 | 3.46e-06, 3.41e-06, 3.49e-06 | 1.000000000000 | True | 1.9e-06 | False |

## Verdict

**not-viable.** Column-chunking fixes the cache problem it targets, but on one thread only. Single-thread, the down site's 256-row cast + 2 dots takes 747–777 µs unsplit, 385–406 µs at Kc 864 (1.9x faster) and 472–496 µs at Kc 1728 (1.6x). In the W 8 pool the gain does not survive. K-blocking every site is 0.65–0.72x (Kc 864) and 0.81–0.89x (Kc 1728) of u16ft_st at B 2, and 0.60–0.64x / 0.75–0.82x at B 4. The per-sub-tile take variant (u16dk, all scratch L2-resident) is worse still at 0.53–0.76x. The best config K-blocks only the 7.08 MB-tile down site at Kc 1728 and runs the in-2560 sites unsplit: 0.95–0.99x (B 2) and 0.86–0.97x (B 4). The bar of >= 1.3x is missed everywhere. The stage profile shows why. Under the pool, cast + dot wall worker-ms rise from 16 + 23 to 27–30 + 43–46 (B 2, K-blocked all sites) while their CPU-ms stay flat or drop (13 + 18 → 13–15 + 14–17). The extra time is off-CPU wait between the 2–8x more short numpy calls per tile, not memory stalls. Cache residency is not the binding limit on this pool. The int8d control confirms it: the same K-blocking makes int8d slower too (best K-blocked 0.80–0.90x of unsplit i8d_gd2). Against unsplit int8d the best new config is 1.53–1.59x behind at B 2 and 1.48–1.54x at B 4, and u16ft_st is 1.45–1.57x / 1.32–1.47x behind this wave. Against K-blocked int8d it is 1.34–1.39x / 1.22–1.29x behind. Error: splitting K reorders the f32 accumulation, so K-blocked outputs are not bit-identical to u16ft_st (max diff ≤ 1.1e-6; the int8 control differs by ≤ 4.3e-6 from i8d). All stay ≤ 1.12e-6 from the float64 ternary reference (cos 1.0), and the decoded trits are exact (pack round-trip asserted). Ternary keeps only the known 3.98x weight-memory win (17.46 vs 69.56 MB). The fused-take scratch shrinks 19.1 → 12.9–13.8 MB per worker and the per-sub-tile variant needs only 1.1–2.2 MB, but that buys no speed.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-27 10:55 wave reached this lane, and no transcript tool was available to read a Chief-named item, so this fire took the chained question from the 05:50 note. JuniorLLM HEAD at clone was 3f10b6c (cloudcover95, 05:54 MT, this lane's 05:50 note). No Trit Pack, Handshake or LLC commit falls in the last 6 hours. This note only adds one new file under junior_osai/vault/findings/.

## Next question

The binding limit in the W 8 thread pool is the number of short numpy calls per tile (off-CPU wall that grows with call count), not L2 residency. Single-thread K-blocking wins 1.6–1.9x but loses it all in the pool. The next test applies K-blocking to the down site only, at a fixed ~1.77 MB f32 sub-tile, while cutting calls per weight byte: 512 rows × Kc 864 and 1024 rows × Kc 432, against 256 × 1728. Separately: does replacing the per-b np.dot loop with one (Kc, B)-shaped np.matmul per sub-tile (y_tile += sub @ X[:, k0:k1].T) reduce calls enough to recover >= 1.3x over u16ft_st at B 2 / B 4, and bring u16 within 1.3x of int8d?

## Script (v16k.py)

```python
#!/usr/bin/env python3
"""V16K: K-blocked (column-chunk) u16ft for the 8-trit uint16 pack, row-major per-row np.dot: one fused u64 take per static worker range
(or per sub-tile), then cast + per-b partial np.dot on 256-row x Kc-column sub-tiles (f32 sub-tile L2-resident), accumulated into y,
vs u16ft_st (02:47 best) and int8d (unsplit, and K-blocked the same way as a control), at B2 / B4, W=8 decode layer, full bitnetd layer, ctx-free.
Base: /workspace/bench0255_1127/v16t.py (exec'd without main(); it execs v16d.py -> v16.py: same weights, P16 pack with exact trit
round-trip asserts, float64 refs, dy / gd2 schedules, r_u16ft kernel, profiling hooks).
  Kc in trit columns: 864 (108 u16 codes, f32 sub-tile 256 x 864 x 4 = 884,736 B) and 1728 (216 codes, 1,769,472 B); both multiples of 8,
  so chunks align with the 8-trit u16 groups. in 2560 sites (qkv/o/gu) split 864+864+832 / 1728+832; in 6912 (down) splits 8 x 864 / 4 x 1728.
  u16ftk{Kc}_st : static split, ONE np.take over the whole worker range (as u16ft) -> per 256-row tile, per K chunk: cast strided int8 view
                  into a contiguous (256, Kc) f32 sub-tile -> per-b np.dot -> y (first chunk) / tmp + add (later chunks)
  u16dk{Kc}_st  : static split, per 256-row x Kc sub-tile: take (strided P16 slice -> contiguous u64) -> cast -> per-b partial dot
  u16ftk{Kc}dn_st / i8dk{Kc}dn_gd2 : K-block only the down site (in 6912); in 2560 sites (2.6 MB f32 tile) run unsplit
  i8dk{Kc}_gd2  : int8d (T128 gd2 schedule) with the same K-blocking of cast + per-b partial dot; i8dk{Kc}_st with 256-row tiles static
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16k.py out.json"""
import json, os, sys
import numpy as np
_S = open("/workspace/bench0255_1127/v16t.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "v16t.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
KCS = (864, 1728); assert all(k % 8 == 0 for k in KCS)
def kspl(i, kc):
    if kc < 0: kc = -kc; kc = kc if i > D else i          # negative kc: K-block only the in 6912 site (down), in 2560 sites unsplit
    return [(c, min(i, c + kc)) for c in range(0, i, kc)]
def mk_u16ftk(kc):
    def r(PR, x, y, r0, r1, T, w, i):
        q = PR.shape[1]; n = r1 - r0; P = PRF[0]; TT = 256
        su = scr(("u16f", w), (RMAX * Q8,), np.uint64)[:n * q].reshape(n, q); s8 = su.view(np.int8).reshape(n, q * 8)
        sb = scr(("fk", w, kc), (TT * F,), np.float32); tmp = scr(("tk", w), (TT,), np.float32)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16U, PR[r0:r1], out=su)
        if P: _add(w, "take", pc() - t0, tt() - c0)
        for a in range(0, n, TT):
            m = min(TT, n - a)
            for j, (k0, k1) in enumerate(kspl(i, kc)):
                s = sb[:m * (k1 - k0)].reshape(m, k1 - k0)
                if P: t0 = pc(); c0 = tt()
                np.copyto(s, s8[a:a+m, k0:k1], casting="unsafe")
                if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
                for b in range(x.shape[0]):
                    yv = y[b, r0+a:r0+a+m]
                    if j == 0: np.dot(s, x[b, k0:k1], out=yv)
                    else: np.dot(s, x[b, k0:k1], out=tmp[:m]); np.add(yv, tmp[:m], out=yv)
                if P: _add(w, "dot", pc() - t0, tt() - c0)
    return r
def mk_u16dk(kc):
    def r(PR, x, y, r0, r1, T, w, i):
        P = PRF[0]; TT = 256; qk = kc // 8
        su = scr(("uk", w, kc), (TT * qk,), np.uint64); sb = scr(("fk", w, kc), (TT * kc,), np.float32); tmp = scr(("tk", w), (TT,), np.float32)
        for o0 in range(r0, r1, TT):
            m = min(TT, r1 - o0)
            for j, (k0, k1) in enumerate(kspl(i, kc)):
                u = su[:m * ((k1 - k0) // 8)].reshape(m, (k1 - k0) // 8); s = sb[:m * (k1 - k0)].reshape(m, k1 - k0)
                if P: t0 = pc(); c0 = tt()
                np.take(LUT16U, PR[o0:o0+m, k0 // 8:k1 // 8], out=u)
                if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
                np.copyto(s, u.view(np.int8).reshape(m, k1 - k0), casting="unsafe")
                if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
                for b in range(x.shape[0]):
                    yv = y[b, o0:o0+m]
                    if j == 0: np.dot(s, x[b, k0:k1], out=yv)
                    else: np.dot(s, x[b, k0:k1], out=tmp[:m]); np.add(yv, tmp[:m], out=yv)
                if P: _add(w, "dot", pc() - t0, tt() - c0)
    return r
def mk_8dk(kc):
    def r(Q, x, y, r0, r1, T, w, i):
        P = PRF[0]; TT = T; sb = scr(("f8k", w, kc, TT), (TT * F,), np.float32); tmp = scr(("t8k", w, TT), (TT,), np.float32)
        for o0 in range(r0, r1, TT):
            m = min(TT, r1 - o0)
            for j, (k0, k1) in enumerate(kspl(i, kc)):
                s = sb[:m * (k1 - k0)].reshape(m, k1 - k0)
                if P: t0 = pc(); c0 = tt()
                np.copyto(s, Q[o0:o0+m, k0:k1], casting="unsafe")
                if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
                for b in range(x.shape[0]):
                    yv = y[b, o0:o0+m]
                    if j == 0: np.dot(s, x[b, k0:k1], out=yv)
                    else: np.dot(s, x[b, k0:k1], out=tmp[:m]); np.add(yv, tmp[:m], out=yv)
                if P: _add(w, "dot", pc() - t0, tt() - c0)
    return r
K8 = {}
for kc in KCS: KD[f"u16ftk{kc}"] = mk_u16ftk(kc); KD[f"u16dk{kc}"] = mk_u16dk(kc); K8[f"8dk{kc}"] = mk_8dk(kc)
for kc in KCS: KD[f"u16ftk{kc}dn"] = mk_u16ftk(-kc); K8[f"8dk{kc}dn"] = mk_8dk(-kc)
_mm_v16t = mm
def mm(k, a, kind, T, mode, rec):
    if kind not in K8: return _mm_v16t(k, a, kind, T, mode, rec)
    CUR[0] = k; B = a.shape[0]; P = par.F8[k]; R = P["Q"]; i = R.shape[1]; out = R.shape[0]
    x = np.ascontiguousarray(a, np.float32); y = np.empty((B, out), np.float32)
    run_site(K8[kind], R, x, y, out, T, i, mode, rec, k, "8d")
    y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
C = {"u16ft_st": ("u16ft", 8, "st")}
for kc in KCS: C[f"u16ftk{kc}_st"] = (f"u16ftk{kc}", 8, "st")
for kc in KCS: C[f"u16dk{kc}_st"] = (f"u16dk{kc}", 256, "st")
for kc in KCS: C[f"u16ftk{kc}dn_st"] = (f"u16ftk{kc}dn", 8, "st")
C.update({"i8d_dy": ("8d", 128, "dy"), "i8d_gd2": ("8d", 128, "gd2")})
for kc in KCS: C[f"i8dk{kc}_gd2"] = (f"8dk{kc}", 128, "gd2")
for kc in KCS: C[f"i8dk{kc}_st"] = (f"8dk{kc}", 256, "st")
for kc in KCS: C[f"i8dk{kc}dn_gd2"] = (f"8dk{kc}dn", 128, "gd2")
U = [c for c in C if C[c][0] in KD]; BASE = "u16ft_st"; NEW = [c for c in U if c != BASE]; I8 = ["i8d_dy", "i8d_gd2"]; I8K = [c for c in C if c.startswith("i8dk")]
def micro(seed):
    """single-thread, site down (in 6912), 256 rows, B 2: u16ft per-tile (cast full 256 x 6912 f32 tile 7.08 MB + 2 dots) vs K-blocked cast+dot."""
    k = "down"; PR = P16[k]["PR"][:256]; q = PR.shape[1]; i = P16[k]["inp"]; su = np.empty((256, q), np.uint64); s8 = su.view(np.int8).reshape(256, q * 8)
    np.take(LUT16U, PR, out=su); x = np.random.default_rng(seed).standard_normal((2, i)).astype(np.float32); y = np.empty((2, 256), np.float32)
    s = np.empty((256, i), np.float32); out = dict(site=k, in_=i)
    def t_(f, n=41):
        f(); v = []
        for _ in range(n): t0 = pc(); f(); v.append((pc() - t0) * 1e6)
        return dict(med=float(np.median(v)), lo=float(min(v)), hi=float(max(v)))
    def full():
        np.copyto(s, s8, casting="unsafe")
        for b in range(2): np.dot(s, x[b], out=y[b])
    out["full_cast_dot"] = t_(full); yr = y.copy()
    for kc in KCS:
        sb = np.empty(256 * kc, np.float32); tmp = np.empty(256, np.float32)
        def kb():
            for j, (k0, k1) in enumerate(kspl(i, kc)):
                t = sb[:256 * (k1 - k0)].reshape(256, k1 - k0); np.copyto(t, s8[:, k0:k1], casting="unsafe")
                for b in range(2):
                    if j == 0: np.dot(t, x[b, k0:k1], out=y[b])
                    else: np.dot(t, x[b, k0:k1], out=tmp); y[b] += tmp
        out[f"k{kc}_cast_dot"] = t_(kb); out[f"k{kc}_maxdiff_vs_full"] = float(np.abs(y - yr).max())
    return out
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, RMAX=RMAX, time={}, prof={}, err={})
    W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES); W16 = sum(P16[k]["PR"].nbytes + P16[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(u16=W16, int8=W8, int8_over_u16=W8 / W16, lut_u64=LUT16U.nbytes, scratch_u16ft_per_worker=RMAX * Q8 * 8 + 256 * F * 4,
                        **{f"scratch_u16ftk{kc}_per_worker": RMAX * Q8 * 8 + 256 * kc * 4 for kc in KCS},
                        **{f"scratch_u16dk{kc}_per_worker": 256 * kc + 256 * kc * 4 for kc in KCS}, **{f"subtile_f32_{kc}": 256 * kc * 4 for kc in KCS})
    print("bytes", res["bytes"], flush=True)
    res["micro"] = micro(seed); print("micro", {k: (round(v["med"], 1) if isinstance(v, dict) else v) for k, v in res["micro"].items()}, flush=True)
    names = list(C)
    for B in (2, 4):
        xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0].startswith("8d") else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in U: e[c]["vs_u16ft_mab"] = float(np.abs(outs[c] - outs["u16ft_st"]).max()); e[c]["bitident_u16ft"] = bool(np.array_equal(outs[c], outs["u16ft_st"]))
        for c in I8K: e[c]["vs_i8d_mab"] = float(np.abs(outs[c] - outs["i8d_gd2"]).max()); e[c]["bitident_i8d"] = bool(np.array_equal(outs[c], outs["i8d_gd2"]))
        res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.12f}", v["instr_identical"], v.get("bitident_u16ft", v.get("bitident_i8d"))) for c, v in e.items()}, flush=True)
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
        bn = min(NEW, key=lambda c: o[c]["med"]); bi = min(I8, key=lambda c: o[c]["med"]); bk = min(I8K, key=lambda c: o[c]["med"])
        pr_(BASE, bn, "u16ft/best_new")
        for c in NEW: pr_(BASE, c, f"u16ft/{c}")
        pr_(bn, bi, "best_new/best_i8d"); pr_(BASE, bi, "u16ft/best_i8d"); pr_(bi, bk, "best_i8d/best_i8dk"); pr_(bn, bk, "best_new/best_i8dk")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:15s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
        pr = {}; PRF[0] = True
        for c in U + I8K:
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
            print(f"B{B} prof {c:15s} " + "  ".join(f"{st} {v['wall']:.1f}/{v['cpu']:.1f}ms" for st, v in pr[c].items()), flush=True)
        PRF[0] = False; res["prof"][f"B{B}"] = pr
    json.dump(res, open(sys.argv[1], "w"), indent=1)
main()
```

close: 5a8104ba2434ee28
