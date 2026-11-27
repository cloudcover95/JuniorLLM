---
local: true
date: 2026-11-27
wave: "2026-11-27 14:55 slot (fired 14:47 MT, ran ~14:48–14:56 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (down-site-only K-blocked u16ft at a fixed 1.77 MB f32 sub-tile, shapes 256x1728 / 512x864 / 1024x432, each with per-b np.dot or ONE np.matmul per sub-tile, on the W 8 decode layer, full bitnetd layer, ctx-free. B 2: best new 0.96–1.03x of u16ft_st (per-b dot; the matmul variants are 0.72–0.80x). B 4: one matmul per sub-tile wins, best 1.13–1.15x (256x1728 mm 1.09–1.13x, 512x864 mm 1.08–1.15x), and it halves dot wall worker-ms 42 -> 30. The bar of >= 1.3x vs u16ft_st is missed at both B. Gap to int8d: 1.49–1.54x at B 2 and 1.20x at B 4 (u16ft_st 1.35–1.38x), the first time a u16 path lands within 1.3x of int8d, but ternary is still slower and int8d has no matmul control yet. Error <= 1.16e-6 vs the float64 ternary reference (cos 1.0); not bit-identical to u16ft_st (<= 1.5e-6, reordered f32 sums); trits exact. Memory unchanged: 17.46 vs 69.56 MB int8 (3.98x, the known weight-memory win). No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, uint16, eight-trit-pack, base-3, 2-bit, np-take, fused-take, static-split, k-blocking, column-chunk, np-matmul, call-count, l2, int8, numpy, not-viable]
---

# Down-only K-blocked u16ft with one matmul per sub-tile: 1.13–1.15x at B 4, ~1.0x at B 2, int8d gap down to 1.2x at B 4

The 10:55 note ([[2026-11-27_1055_column-chunk-k864-k1728-u16ft-fused-take-per-worker-range-t256-vs-u16ft-st-and-int8d-split-and-unsplit-w8-pool-b2-b4-full-bitnetd-layer-slower-0p53-0p99x-single-thread-1p9x-gap-int8d-1p5x]]) found that K-blocking u16ft wins 1.6–1.9x on one thread but loses it in the W 8 pool. Cast + dot wall worker-ms grow with the number of short numpy calls while CPU-ms stay flat. It asked whether K-blocking only the down site at a fixed ~1.77 MB f32 sub-tile, with fewer calls per weight byte (512 × 864 and 1024 × 432 vs 256 × 1728), and one (B, Kc) @ (Kc, m) np.matmul per sub-tile in place of the per-b np.dot loop, recovers >= 1.3x over u16ft_st at B 2 / B 4 and brings u16 within 1.3x of int8d.

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 2 and B 4: does a down-only K-blocked u16ft at a fixed 1.77 MB f32 sub-tile (256 × 1728, 512 × 864, 1024 × 432), with per-b np.dot or one np.matmul per sub-tile, beat u16ft_st by >= 1.3x? Where does it land against int8d?

## Method

- Box: 8 CPUs (Xeon, L2 2 MiB per core), numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents (load avg 0.7–0.9 at start), so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1055_1127/v16k.py exec'd without main() (→ v16t.py → v16d.py → v16.py: same weights, P16 8-trit uint16 pack with exact trit round-trip asserts, float64 references, schedules, the r_u16ft kernel, profiling hooks).
- Kernels (static split, ONE fused u64 LUT take per worker range, as u16ft). The in-2560 sites (qkv / o / gu) always run unsplit 256-row tiles. Only the in-6912 down site is K-blocked:
  - **u16m{R}x{Kc}dot_st**: per R-row tile, per Kc chunk: cast the strided int8 view into a contiguous (R, Kc) f32 sub-tile (always 1,769,472 B), then per-b np.dot (chunk 0 into y, later chunks into tmp + np.add).
  - **u16m{R}x{Kc}mm_st**: the same, but ONE np.matmul per sub-tile: tmp(B, R) = X[:, k0:k1] @ sub.T, then y[:, rows] = / += tmp.
  - **u16ftmm_st**: no K-blocking. Every site uses 256-row tiles with one matmul per tile.
  - Baselines: **u16ft_st** (02:47 best), **u16ftk1728dn_st** (10:55 best, rerun), **i8d_dy / i8d_gd2** (int8d T 128).
- Microbench (single thread, down site, 1024 rows, B 2 and B 4): cast + dots over the whole 1024 × 6912 block for each shape, dot vs mm, plus the unsplit 256-row tile (full256 / full256mm). 31 reps, median [min–max].
- Error: max abs error and cos vs the float64 ternary (u16) or int8 reference. Each run also checks that instrumented and plain output are identical, and records max abs diff and np.array_equal vs u16ft_st.
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round. Reported as median of round medians, round min–max and per-round ratios. Then 15 instrumented layers per config give stage worker-ms (wall/cpu summed over workers) and per-site span. Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench1455_1127 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16m.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Bytes: weights u16 17,458,176 B vs int8 69,559,296 B (3.98x); LUT u64 52,488 B; f32 sub-tile 1,769,472 B for every shape; per-worker scratch 19,077,120 B (unchanged). Below: the microbench, layer ms (median of round medians [round min–max], runs 1 / 2 / 3), ratios, stage worker-ms (wall/cpu, runs 1, 2, 3) and error.

bytes {'u16': 17458176, 'int8': 69559296, 'int8_over_u16': 3.9843392574344536, 'lut_u64': 52488, 'subtile_f32': 1769472, 'scratch_per_worker': 19077120}

Single-thread microbench, down site (in 6912), 1024 rows: cast + dots over the whole block per shape, µs median [min–max], runs 1 / 2 / 3 (maxdiff vs full256 per-b dot):

| op | run 1 | run 2 | run 3 | max diff |
|---|---|---|---|---|
| B2_full256 | 3027 [2867–3221] | 2993 [2876–3192] | 2925 [2731–3102] | 0.0e+00 |
| B2_full256mm | 3465 [3196–3636] | 3362 [3175–3608] | 3302 [3169–3953] | 1.3e-04 |
| B2_256x1728dot | 1887 [1741–2055] | 1769 [1737–1933] | 1829 [1791–2052] | 1.0e-04 |
| B2_256x1728mm | 1489 [1468–1737] | 1508 [1481–1829] | 1562 [1495–1787] | 1.0e-04 |
| B2_512x864dot | 1932 [1878–2724] | 1919 [1879–2072] | 2004 [1923–2166] | 1.1e-04 |
| B2_512x864mm | 1664 [1629–1867] | 1676 [1607–2042] | 1688 [1623–1797] | 1.1e-04 |
| B2_1024x432dot | 2379 [2264–2782] | 2697 [2189–12722] | 2285 [2233–2515] | 1.0e-04 |
| B2_1024x432mm | 3822 [3752–3945] | 3796 [3618–4697] | 3817 [3640–4123] | 1.2e-04 |
| B4_full256 | 4802 [4705–4986] | 4933 [4799–5153] | 4769 [4612–4946] | 0.0e+00 |
| B4_full256mm | 3406 [3233–3584] | 3438 [3247–4439] | 3456 [3244–3623] | 1.3e-04 |
| B4_256x1728dot | 2543 [2510–2785] | 2790 [2616–4673] | 2673 [2572–2943] | 1.1e-04 |
| B4_256x1728mm | 3272 [3201–3574] | 3384 [3255–4101] | 3566 [3264–4235] | 1.1e-04 |
| B4_512x864dot | 2726 [2681–2881] | 2842 [2779–3820] | 2818 [2738–4055] | 1.2e-04 |
| B4_512x864mm | 3565 [3404–4524] | 3631 [3414–4402] | 3617 [3342–4332] | 1.5e-04 |
| B4_1024x432dot | 3246 [3117–3529] | 3771 [3255–4465] | 3189 [3088–4463] | 1.1e-04 |
| B4_1024x432mm | 3996 [3784–16025] | 3803 [3737–4358] | 3814 [3663–4907] | 1.5e-04 |

### B2

| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | u16ft_st | 10.36 [8.81–13.98] | 9.93 [8.83–11.62] | 9.77 [8.72–13.69] |
| 2 | u16ftk1728dn_st | 10.42 [9.71–10.91] | 10.39 [9.60–13.95] | 10.46 [9.63–13.95] |
| 2 | u16ftmm_st | 13.66 [12.72–14.85] | 13.12 [12.48–17.59] | 13.59 [12.65–17.98] |
| 2 | u16m256x1728dot_st | 10.15 [9.95–11.84] | 10.36 [9.61–14.10] | 9.99 [9.74–13.87] |
| 2 | u16m256x1728mm_st | 12.91 [12.39–13.89] | 13.43 [12.67–14.80] | 12.19 [11.87–23.27] |
| 2 | u16m512x864dot_st | 10.06 [9.66–11.72] | 10.54 [9.62–13.12] | 9.75 [9.52–13.07] |
| 2 | u16m512x864mm_st | 13.13 [12.35–15.69] | 12.83 [12.08–14.70] | 12.98 [12.10–16.90] |
| 2 | u16m1024x432dot_st | 11.37 [11.00–15.05] | 11.75 [10.99–15.70] | 11.30 [10.89–12.12] |
| 2 | u16m1024x432mm_st | 15.03 [13.48–16.71] | 13.87 [13.47–16.75] | 13.61 [13.26–14.02] |
| 2 | i8d_dy | 7.46 [7.26–8.39] | 7.45 [6.69–8.16] | 7.26 [6.86–7.75] |
| 2 | i8d_gd2 | 6.70 [6.36–7.79] | 6.71 [6.47–7.53] | 6.57 [6.49–7.87] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | u16ft/best_new | 1.030x [0.853–1.389] (u16ft_st/u16m512x864dot_st) | 0.958x [0.704–1.209] (u16ft_st/u16m256x1728dot_st) | 1.002x [0.894–1.047] (u16ft_st/u16m512x864dot_st) |
| 2 | u16ft/u16ftmm_st | 0.758x [0.692–1.081] (u16ft_st/u16ftmm_st) | 0.757x [0.565–0.931] (u16ft_st/u16ftmm_st) | 0.719x [0.485–0.936] (u16ft_st/u16ftmm_st) |
| 2 | u16ft/u16m256x1728dot_st | 1.021x [0.866–1.380] (u16ft_st/u16m256x1728dot_st) | 0.958x [0.704–1.209] (u16ft_st/u16m256x1728dot_st) | 0.978x [0.852–1.305] (u16ft_st/u16m256x1728dot_st) |
| 2 | u16ft/u16m256x1728mm_st | 0.802x [0.711–1.128] (u16ft_st/u16m256x1728mm_st) | 0.739x [0.657–0.785] (u16ft_st/u16m256x1728mm_st) | 0.802x [0.588–1.070] (u16ft_st/u16m256x1728mm_st) |
| 2 | u16ft/u16m512x864dot_st | 1.030x [0.853–1.389] (u16ft_st/u16m512x864dot_st) | 0.942x [0.885–1.014] (u16ft_st/u16m512x864dot_st) | 1.002x [0.894–1.047] (u16ft_st/u16m512x864dot_st) |
| 2 | u16ft/u16m512x864mm_st | 0.789x [0.661–1.115] (u16ft_st/u16m512x864mm_st) | 0.774x [0.693–0.914] (u16ft_st/u16m512x864mm_st) | 0.753x [0.680–0.975] (u16ft_st/u16m512x864mm_st) |
| 2 | u16ft/u16m1024x432dot_st | 0.911x [0.689–1.271] (u16ft_st/u16m1024x432dot_st) | 0.845x [0.618–0.999] (u16ft_st/u16m1024x432dot_st) | 0.864x [0.800–1.129] (u16ft_st/u16m1024x432dot_st) |
| 2 | u16ft/u16m1024x432mm_st | 0.689x [0.620–1.022] (u16ft_st/u16m1024x432mm_st) | 0.716x [0.579–0.838] (u16ft_st/u16m1024x432mm_st) | 0.718x [0.658–1.006] (u16ft_st/u16m1024x432mm_st) |
| 2 | u16ft/u16ftk1728dn_st | 0.994x [0.821–1.440] (u16ft_st/u16ftk1728dn_st) | 0.956x [0.712–1.211] (u16ft_st/u16ftk1728dn_st) | 0.935x [0.887–1.011] (u16ft_st/u16ftk1728dn_st) |
| 2 | best_new/best_i8d | 1.502x [1.307–1.762] (u16m512x864dot_st/i8d_gd2) | 1.544x [1.432–1.872] (u16m256x1728dot_st/i8d_gd2) | 1.485x [1.438–1.973] (u16m512x864dot_st/i8d_gd2) |
| 2 | u16ft/best_i8d | 1.547x [1.299–2.057] (u16ft_st/i8d_gd2) | 1.479x [1.319–1.731] (u16ft_st/i8d_gd2) | 1.488x [1.290–1.959] (u16ft_st/i8d_gd2) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 2 | u16ft_st | 21.2/18.2, 20.7/18.3, 21.3/18.5 | 15.5/12.9, 14.8/12.9, 15.3/13.2 | 23.1/17.6, 22.5/17.5, 23.0/17.5 | 61.4/50.1, 59.8/49.8, 61.3/50.1 | 9.39, 9.14, 9.34 |
| 2 | u16ftk1728dn_st | 22.2/18.5, 21.5/18.3, 21.0/18.0 | 18.8/12.7, 18.2/12.3, 18.8/12.5 | 30.3/16.6, 28.8/16.0, 29.3/16.2 | 74.1/49.9, 70.0/48.6, 71.3/48.7 | 11.14, 10.34, 10.31 |
| 2 | u16ftmm_st | 24.2/20.6, 22.1/18.8, 23.4/18.9 | 23.8/14.9, 23.5/13.5, 23.3/13.5 | 31.9/25.6, 29.8/24.1, 30.3/24.3 | 83.0/63.1, 76.7/58.3, 78.7/58.1 | 14.08, 12.42, 13.44 |
| 2 | u16m256x1728dot_st | 22.5/19.4, 21.4/18.4, 20.8/18.0 | 19.0/12.6, 18.5/12.3, 18.2/12.3 | 31.9/17.0, 29.9/16.3, 29.8/16.3 | 75.6/50.7, 72.4/48.7, 71.2/48.4 | 11.38, 10.55, 10.35 |
| 2 | u16m256x1728mm_st | 32.8/28.1, 21.8/18.6, 21.6/18.5 | 31.0/16.5, 26.8/12.2, 28.0/12.3 | 37.5/28.2, 28.3/20.6, 29.1/21.1 | 103.6/75.5, 78.3/52.9, 80.6/53.4 | 17.49, 12.68, 13.01 |
| 2 | u16m512x864dot_st | 23.9/20.0, 21.5/18.5, 21.6/18.1 | 20.1/13.0, 18.7/12.3, 19.2/12.4 | 28.3/16.6, 27.9/16.1, 28.0/16.0 | 74.3/51.3, 69.1/48.3, 70.9/48.2 | 11.16, 10.31, 10.35 |
| 2 | u16m512x864mm_st | 23.1/19.0, 22.1/18.4, 21.8/18.4 | 26.0/12.7, 24.8/12.2, 25.1/12.3 | 29.8/21.3, 30.3/20.8, 30.6/20.7 | 80.7/54.6, 79.1/53.0, 79.0/53.1 | 13.12, 12.71, 13.05 |
| 2 | u16m1024x432dot_st | 21.3/18.4, 21.8/18.4, 21.7/18.3 | 22.1/13.1, 22.1/12.6, 21.7/12.9 | 37.8/17.2, 37.9/16.7, 37.8/17.0 | 83.7/50.6, 83.6/49.9, 83.3/50.1 | 12.18, 12.04, 11.95 |
| 2 | u16m1024x432mm_st | 22.4/19.0, 20.8/17.4, 22.0/18.4 | 29.8/13.0, 28.6/12.5, 29.8/12.9 | 35.8/21.6, 34.0/20.7, 34.8/21.5 | 90.2/55.4, 85.3/52.3, 88.2/54.4 | 14.27, 13.41, 13.74 |

| B | config | max abs err vs ref | cos | instr identical | max abs diff vs u16ft_st | bit-identical to u16ft_st |
|---|---|---|---|---|---|---|
| 2 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 0.0e+00 | True |
| 2 | u16ftk1728dn_st | 1.12e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 4.8e-07 | False |
| 2 | u16ftmm_st | 1.07e-06, 1.16e-06, 1.04e-06 | 1.000000000000 | True | 1.5e-06 | False |
| 2 | u16m256x1728dot_st | 1.12e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 4.8e-07 | False |
| 2 | u16m256x1728mm_st | 9.84e-07, 9.10e-07, 9.99e-07 | 1.000000000000 | True | 1.4e-06 | False |
| 2 | u16m512x864dot_st | 1.00e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 5.4e-07 | False |
| 2 | u16m512x864mm_st | 8.95e-07, 9.40e-07, 9.99e-07 | 1.000000000000 | True | 1.4e-06 | False |
| 2 | u16m1024x432dot_st | 1.00e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 4.8e-07 | False |
| 2 | u16m1024x432mm_st | 9.55e-07, 9.10e-07, 1.00e-06 | 1.000000000000 | True | 1.4e-06 | False |
| 2 | i8d_dy | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | – | – |
| 2 | i8d_gd2 | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | – | – |

### B4

| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 4 | u16ft_st | 12.22 [11.05–17.38] | 12.49 [11.29–13.44] | 12.64 [11.19–13.25] |
| 4 | u16ftk1728dn_st | 13.40 [12.63–14.83] | 13.19 [12.34–14.00] | 13.16 [12.89–14.03] |
| 4 | u16ftmm_st | 12.25 [11.66–13.79] | 12.77 [11.56–14.00] | 12.86 [11.63–14.94] |
| 4 | u16m256x1728dot_st | 12.94 [12.58–14.23] | 12.81 [12.49–13.83] | 13.25 [12.58–18.19] |
| 4 | u16m256x1728mm_st | 10.86 [10.16–11.90] | 11.25 [10.09–11.97] | 11.62 [10.46–14.57] |
| 4 | u16m512x864dot_st | 12.83 [12.31–13.11] | 12.73 [12.20–13.33] | 13.35 [12.36–17.88] |
| 4 | u16m512x864mm_st | 11.29 [10.61–11.49] | 10.98 [10.11–12.44] | 10.96 [10.56–12.25] |
| 4 | u16m1024x432dot_st | 15.04 [14.25–15.82] | 14.52 [13.78–15.96] | 15.40 [14.21–16.53] |
| 4 | u16m1024x432mm_st | 12.15 [11.24–13.30] | 11.64 [11.27–12.66] | 12.99 [11.11–15.53] |
| 4 | i8d_dy | 9.98 [9.81–11.66] | 9.99 [9.34–11.41] | 10.20 [9.63–12.30] |
| 4 | i8d_gd2 | 9.06 [8.86–10.24] | 9.14 [8.70–9.24] | 9.17 [8.70–10.74] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 4 | u16ft/best_new | 1.126x [1.060–1.531] (u16ft_st/u16m256x1728mm_st) | 1.138x [0.939–1.221] (u16ft_st/u16m512x864mm_st) | 1.154x [0.938–1.255] (u16ft_st/u16m512x864mm_st) |
| 4 | u16ft/u16ftmm_st | 0.998x [0.876–1.343] (u16ft_st/u16ftmm_st) | 0.978x [0.823–1.052] (u16ft_st/u16ftmm_st) | 0.983x [0.846–1.139] (u16ft_st/u16ftmm_st) |
| 4 | u16ft/u16m256x1728dot_st | 0.945x [0.849–1.251] (u16ft_st/u16m256x1728dot_st) | 0.975x [0.899–0.998] (u16ft_st/u16m256x1728dot_st) | 0.954x [0.728–0.981] (u16ft_st/u16m256x1728dot_st) |
| 4 | u16ft/u16m256x1728mm_st | 1.126x [1.060–1.531] (u16ft_st/u16m256x1728mm_st) | 1.110x [0.962–1.209] (u16ft_st/u16m256x1728mm_st) | 1.088x [0.910–1.209] (u16ft_st/u16m256x1728mm_st) |
| 4 | u16ft/u16m512x864dot_st | 0.953x [0.893–1.326] (u16ft_st/u16m512x864dot_st) | 0.981x [0.864–1.055] (u16ft_st/u16m512x864dot_st) | 0.947x [0.741–1.045] (u16ft_st/u16m512x864dot_st) |
| 4 | u16ft/u16m512x864mm_st | 1.083x [1.001–1.639] (u16ft_st/u16m512x864mm_st) | 1.138x [0.939–1.221] (u16ft_st/u16m512x864mm_st) | 1.154x [0.938–1.255] (u16ft_st/u16m512x864mm_st) |
| 4 | u16ft/u16m1024x432dot_st | 0.813x [0.776–1.199] (u16ft_st/u16m1024x432dot_st) | 0.860x [0.787–0.894] (u16ft_st/u16m1024x432dot_st) | 0.821x [0.712–0.932] (u16ft_st/u16m1024x432dot_st) |
| 4 | u16ft/u16m1024x432mm_st | 1.006x [0.966–1.307] (u16ft_st/u16m1024x432mm_st) | 1.073x [0.944–1.131] (u16ft_st/u16m1024x432mm_st) | 0.973x [0.832–1.193] (u16ft_st/u16m1024x432mm_st) |
| 4 | u16ft/u16ftk1728dn_st | 0.912x [0.825–1.172] (u16ft_st/u16ftk1728dn_st) | 0.947x [0.837–1.012] (u16ft_st/u16ftk1728dn_st) | 0.961x [0.797–0.989] (u16ft_st/u16ftk1728dn_st) |
| 4 | best_new/best_i8d | 1.199x [1.109–1.232] (u16m256x1728mm_st/i8d_gd2) | 1.201x [1.095–1.346] (u16m512x864mm_st/i8d_gd2) | 1.195x [1.020–1.409] (u16m512x864mm_st/i8d_gd2) |
| 4 | u16ft/best_i8d | 1.349x [1.240–1.698] (u16ft_st/i8d_gd2) | 1.367x [1.224–1.480] (u16ft_st/i8d_gd2) | 1.379x [1.177–1.450] (u16ft_st/i8d_gd2) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 4 | u16ft_st | 22.3/19.1, 22.2/19.5, 21.5/18.8 | 15.3/13.1, 15.3/13.2, 15.3/12.9 | 41.8/33.5, 42.1/33.9, 42.2/33.3 | 79.9/66.6, 81.0/68.3, 80.6/66.7 | 12.34, 11.84, 11.94 |
| 4 | u16ftk1728dn_st | 21.3/18.7, 21.3/18.3, 21.8/19.0 | 17.9/12.6, 17.2/12.3, 18.6/12.7 | 51.4/30.3, 50.5/29.6, 51.4/30.6 | 92.2/63.6, 91.2/62.4, 95.0/64.1 | 13.21, 13.37, 13.77 |
| 4 | u16ftmm_st | 21.8/18.6, 23.8/19.6, 22.0/18.3 | 19.0/13.2, 20.7/13.7, 20.3/13.4 | 29.8/25.8, 31.3/27.1, 29.1/25.6 | 72.8/59.0, 76.6/61.9, 73.3/59.0 | 11.29, 12.66, 12.16 |
| 4 | u16m256x1728dot_st | 21.3/18.2, 21.7/18.7, 20.6/18.0 | 17.3/12.3, 18.2/12.4, 17.2/12.2 | 51.1/30.2, 53.5/30.6, 50.7/30.0 | 92.1/62.2, 95.1/63.3, 89.8/61.6 | 13.11, 13.52, 12.76 |
| 4 | u16m256x1728mm_st | 21.6/18.2, 22.0/18.6, 21.4/18.1 | 19.8/12.1, 21.8/12.4, 20.8/12.2 | 29.6/25.7, 30.3/26.1, 29.6/25.5 | 73.1/58.1, 76.2/58.6, 73.0/57.7 | 11.10, 11.67, 11.15 |
| 4 | u16m512x864dot_st | 21.5/18.1, 24.3/19.0, 21.2/18.1 | 18.2/12.3, 18.9/13.0, 18.2/12.3 | 50.2/29.6, 51.1/30.4, 49.9/29.5 | 91.3/61.5, 97.5/64.2, 91.3/61.8 | 13.12, 15.11, 13.19 |
| 4 | u16m512x864mm_st | 22.4/18.8, 23.2/19.6, 21.6/17.3 | 19.3/12.6, 19.4/13.1, 18.2/12.3 | 33.3/27.5, 34.2/28.6, 32.2/26.8 | 76.7/60.4, 79.0/62.9, 73.5/58.4 | 12.11, 12.48, 11.05 |
| 4 | u16m1024x432dot_st | 21.4/18.2, 24.6/19.8, 21.3/18.3 | 20.6/12.9, 22.2/13.8, 20.5/12.9 | 67.1/32.1, 70.6/33.2, 64.4/31.7 | 111.6/64.9, 120.6/68.5, 107.8/65.0 | 15.64, 16.96, 15.48 |
| 4 | u16m1024x432mm_st | 21.0/17.6, 22.8/19.4, 21.2/17.9 | 22.5/12.7, 23.6/13.5, 22.6/12.7 | 36.1/27.5, 39.4/29.2, 36.4/27.1 | 81.8/59.9, 87.9/64.5, 82.6/59.6 | 11.88, 12.80, 12.36 |

| B | config | max abs err vs ref | cos | instr identical | max abs diff vs u16ft_st | bit-identical to u16ft_st |
|---|---|---|---|---|---|---|
| 4 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 0.0e+00 | True |
| 4 | u16ftk1728dn_st | 1.12e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 4.8e-07 | False |
| 4 | u16ftmm_st | 1.14e-06, 1.16e-06, 1.14e-06 | 1.000000000000 | True | 1.7e-06 | False |
| 4 | u16m256x1728dot_st | 1.12e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 4.8e-07 | False |
| 4 | u16m256x1728mm_st | 1.09e-06, 9.86e-07, 1.02e-06 | 1.000000000000 | True | 1.5e-06 | False |
| 4 | u16m512x864dot_st | 1.11e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 5.4e-07 | False |
| 4 | u16m512x864mm_st | 1.11e-06, 1.05e-06, 1.18e-06 | 1.000000000000 | True | 1.7e-06 | False |
| 4 | u16m1024x432dot_st | 1.00e-06, 8.91e-07, 1.04e-06 | 1.000000000000 | True | 4.8e-07 | False |
| 4 | u16m1024x432mm_st | 1.11e-06, 1.07e-06, 1.17e-06 | 1.000000000000 | True | 1.7e-06 | False |
| 4 | i8d_dy | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | – | – |
| 4 | i8d_gd2 | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | – | – |


## Verdict

**not-viable.** Cutting numpy calls per weight byte helps only where batching gives matmul real work. At B 4, one np.matmul per down-site sub-tile beats u16ft_st by 1.09–1.13x (256 × 1728) and 1.08–1.15x (512 × 864); the best of each run is 1.13–1.15x. The stage profile shows where it comes from: dot wall worker-ms drop from 42 to 30 (CPU 33 → 26), while take (~21) and cast (~18–20) stay put. At B 2 the matmul variants lose (0.72–0.80x), and the best per-b-dot shapes only tie u16ft_st (0.96–1.03x). Taller, narrower sub-tiles are worse at the same byte size: 1024 × 432 is 0.81–0.91x with dot and 0.69–1.07x with mm, so fewer, longer calls do not help when every call gets shorter in K. The all-site unsplit matmul (u16ftmm_st) is 0.72–0.76x at B 2 and 0.98–1.00x at B 4. On one thread the ordering differs again. Over 1024 down rows at B 2, 256 × 1728 mm takes 1489–1562 µs vs 2925–3027 µs unsplit (1.9–2.0x faster). At B 4, per-b dot is fastest single-thread (2543–2790 µs vs 4769–4933 µs) and mm is slower (3272–3566 µs), yet mm wins in the pool. So the pool cost is per call, not per flop, and the 10:55 diagnosis holds. The bar of >= 1.3x vs u16ft_st is missed at both B. Against int8d (i8d_gd2), the best new config is 1.49–1.54x behind at B 2 and 1.20x behind at B 4 (u16ft_st: 1.48–1.55x and 1.35–1.38x). This is the first u16 path within 1.3x of int8d, but ternary is still slower, and int8d did not get the same one-matmul-per-tile treatment. Error: all u16 variants stay <= 1.16e-6 from the float64 ternary reference (cos 1.0). They are not bit-identical to u16ft_st (max diff <= 1.5e-6, reordered f32 sums), and the trits are exact (pack round-trip asserted). Ternary keeps only the known 3.98x weight-memory win (17.46 vs 69.56 MB).

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-27 14:55 wave reached this lane, and the transcript search found no Chief-named item, so this fire took the chained question from the 10:55 note. JuniorLLM HEAD at clone was 5213403 (cloudcover95, 10:56 MT, this lane's 10:55 note). The GitHub commit list since 08:48 MT holds only that commit: no Trit Pack, Handshake or LLC commit in the last 6 hours. This note only adds one new file under junior_osai/vault/findings/.

## Next question

At B 4 the one-matmul sub-tile path takes the dot stage from 42 to 30 worker-ms. take + cast (~40 worker-ms) are now the larger share. Two tests follow. First, a fair control: int8d with one matmul per 256-row tile (i8dmm), down-only K-blocked the same way, at B 4 and B 8. Second, a B-adaptive u16 kernel: per-b dot at B <= 2, down-only 512 × 864 / 256 × 1728 matmul at B >= 4. Does the adaptive kernel reach >= 1.3x over u16ft_st at B 8, and does it stay within 1.3x of i8dmm, or does int8d gain the same amount and reopen the gap?

## Script (v16m.py)

```python
#!/usr/bin/env python3
"""V16M: down-site-only K-blocked u16ft at a fixed ~1.77 MB f32 sub-tile with fewer numpy calls per weight byte:
sub-tile shapes 256x1728 (10:55 best), 512x864, 1024x432, each with per-b np.dot (dot) or ONE np.matmul per sub-tile (mm:
tmp(B,m) = X[:,k0:k1] @ sub.T, then y[:, rows] (+)= tmp), vs u16ft_st, u16ft with one matmul per 256-row tile (u16ftmm_st, all sites
unsplit), and int8d (i8d_gd2, i8d_dy). B2 / B4, W 8 decode layer, full bitnetd layer, ctx-free.
Base: /workspace/bench1055_1127/v16k.py exec'd without main() (-> v16t.py -> v16d.py -> v16.py: same weights, P16 pack with exact trit
round-trip asserts, float64 refs, schedules, profiling hooks). in-2560 sites always run unsplit 256-row tiles (as u16ft).
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16m.py out.json"""
import json, os, sys
import numpy as np
_S = open("/workspace/bench1055_1127/v16k.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "v16k.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
SHAPES = ((256, 1728), (512, 864), (1024, 432)); assert all(r * k == 256 * 1728 and k % 8 == 0 for r, k in SHAPES)
def mk_m(R, KC, MM):
    def r(PR, x, y, r0, r1, T, w, i):
        q = PR.shape[1]; n = r1 - r0; P = PRF[0]; B = x.shape[0]
        if i > D: TT, kc = R, KC
        else: TT, kc = 256, i
        su = scr(("u16f", w), (RMAX * Q8,), np.uint64)[:n * q].reshape(n, q); s8 = su.view(np.int8).reshape(n, q * 8)
        sb = scr(("fm", w), (256 * F,), np.float32); tmp = scr(("tm", w), (4 * 1024,), np.float32)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16U, PR[r0:r1], out=su)
        if P: _add(w, "take", pc() - t0, tt() - c0)
        for a in range(0, n, TT):
            m = min(TT, n - a)
            for j, k0 in enumerate(range(0, i, kc)):
                k1 = min(i, k0 + kc); s = sb[:m * (k1 - k0)].reshape(m, k1 - k0)
                if P: t0 = pc(); c0 = tt()
                np.copyto(s, s8[a:a+m, k0:k1], casting="unsafe")
                if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
                yv = y[:, r0+a:r0+a+m]
                if MM:
                    t = tmp[:B * m].reshape(B, m); np.matmul(x[:, k0:k1], s.T, out=t)
                    if j == 0: yv[...] = t
                    else: np.add(yv, t, out=yv)
                else:
                    for b in range(B):
                        if j == 0: np.dot(s, x[b, k0:k1], out=y[b, r0+a:r0+a+m])
                        else: tv = tmp[:m]; np.dot(s, x[b, k0:k1], out=tv); np.add(y[b, r0+a:r0+a+m], tv, out=y[b, r0+a:r0+a+m])
                if P: _add(w, "dot", pc() - t0, tt() - c0)
    return r
C = {"u16ft_st": ("u16ft", 8, "st"), "u16ftk1728dn_st": ("u16ftk1728dn", 8, "st")}
KD["u16ftmm"] = mk_m(256, 10**9, True); C["u16ftmm_st"] = ("u16ftmm", 8, "st")
for R_, K_ in SHAPES:
    for mmf in (False, True):
        nm = f"u16m{R_}x{K_}{'mm' if mmf else 'dot'}"; KD[nm] = mk_m(R_, K_, mmf); C[nm + "_st"] = (nm, 8, "st")
C.update({"i8d_dy": ("8d", 128, "dy"), "i8d_gd2": ("8d", 128, "gd2")})
U = [c for c in C if C[c][0] in KD]; BASE = "u16ft_st"; NEW = [c for c in U if c not in (BASE, "u16ftk1728dn_st")]; I8 = ["i8d_dy", "i8d_gd2"]
def micro(seed):
    """single thread, down site (in 6912), 1024 rows, B 2 and B 4: cast + dots over the whole 1024 x 6912 block, per shape, dot vs mm."""
    k = "down"; PR = P16[k]["PR"][:1024]; q = PR.shape[1]; i = P16[k]["inp"]; su = np.empty((1024, q), np.uint64); s8 = su.view(np.int8).reshape(1024, q * 8)
    np.take(LUT16U, PR, out=su); out = dict(site=k, in_=i, rows=1024)
    def t_(f, n=31):
        f(); v = []
        for _ in range(n): t0 = pc(); f(); v.append((pc() - t0) * 1e6)
        return dict(med=float(np.median(v)), lo=float(min(v)), hi=float(max(v)))
    sb = np.empty(256 * F, np.float32); tmp = np.empty(4 * 1024, np.float32)
    for B in (2, 4):
        x = np.random.default_rng(seed + B).standard_normal((B, i)).astype(np.float32); y = np.empty((B, 1024), np.float32); ref = None
        for nm, R_, K_, MM in [("full256", 256, i, False), ("full256mm", 256, i, True)] + [(f"{a}x{b}{'mm' if f else 'dot'}", a, b, f) for a, b in SHAPES for f in (False, True)]:
            def g():
                for a0 in range(0, 1024, R_):
                    for j, k0 in enumerate(range(0, i, K_)):
                        k1 = min(i, k0 + K_); s = sb[:R_ * (k1 - k0)].reshape(R_, k1 - k0); np.copyto(s, s8[a0:a0+R_, k0:k1], casting="unsafe"); yv = y[:, a0:a0+R_]
                        if MM:
                            t = tmp[:B * R_].reshape(B, R_); np.matmul(x[:, k0:k1], s.T, out=t)
                            if j == 0: yv[...] = t
                            else: yv += t
                        else:
                            for b in range(B):
                                if j == 0: np.dot(s, x[b, k0:k1], out=y[b, a0:a0+R_])
                                else: tv = tmp[:R_]; np.dot(s, x[b, k0:k1], out=tv); y[b, a0:a0+R_] += tv
            out[f"B{B}_{nm}"] = t_(g)
            if ref is None: ref = y.copy()
            out[f"B{B}_{nm}_maxdiff"] = float(np.abs(y - ref).max())
    return out
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, RMAX=RMAX, time={}, prof={}, err={})
    W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES); W16 = sum(P16[k]["PR"].nbytes + P16[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(u16=W16, int8=W8, int8_over_u16=W8 / W16, lut_u64=LUT16U.nbytes, subtile_f32=256 * 1728 * 4, scratch_per_worker=RMAX * Q8 * 8 + 256 * F * 4)
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
        for c in NEW + ["u16ftk1728dn_st"]: pr_(BASE, c, f"u16ft/{c}")
        pr_(bn, bi, "best_new/best_i8d"); pr_(BASE, bi, "u16ft/best_i8d")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:20s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:28s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
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
            print(f"B{B} prof {c:20s} " + "  ".join(f"{st} {v['wall']:.1f}/{v['cpu']:.1f}ms" for st, v in pr[c].items()), flush=True)
        PRF[0] = False; res["prof"][f"B{B}"] = pr
    json.dump(res, open(sys.argv[1], "w"), indent=1)
main()
```

close: cbf78c26e746457d
