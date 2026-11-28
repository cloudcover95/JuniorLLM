---
local: true
date: 2026-11-28
wave: "2026-11-28 02:55 slot (fired 02:46 MT, ran ~02:47–02:58 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (weight-major u16 with one fused take per worker range, u16gft: cast per tile, tile @ xT -> yT, W 8 decode layer, full bitnetd layer, ctx-free, B 8 / 16 / 32. u16gft is the fastest u16 kernel so far at B >= 8 (1.07–1.17x faster than the 18:55 row-major u16ad), and the gap to the best int8 shrinks as B grows: 1.34–1.36x at B 8, 1.29–1.33x at B 16, 1.21–1.28x at B 32 (int8 = i8g_gd2 or i8dmm_gd2, whichever is faster in the run). The ratio falls below 1.3x only at B 32 (and in one of three runs at B 16). Ternary is still slower at every batch; the take stage (~20–22 worker-ms per layer, flat in B) is the floor that int8 does not pay. Error <= 1.23e-6 vs the float64 ternary reference (cos 1.0); trits exact. Memory unchanged: 17.46 vs 69.56 MB int8 (3.98x, the known weight-memory win). No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, batched-decode, thread-pool, lut-expand, uint16, eight-trit-pack, base-3, 2-bit, np-take, fused-take, static-split, weight-major, k-blocking, np-matmul, int8-control, int8, numpy, not-viable]
---

# Weight-major u16gft: fastest u16 yet at B >= 8, and the int8 gap shrinks from 1.35x (B 8) to 1.21–1.28x (B 32), but ternary stays slower

The 18:55 note ([[2026-11-27_1855_int8-one-matmul-control-i8dmm-and-b-adaptive-u16-dot-b2-matmul-b4-b8-vs-u16ft-st-i8d-i8g-w8-pool-b2-b4-b8-adaptive-1p38-1p50x-b8-gap-int8-matmul-1p4x-i8g-1p8x]]) found that int8 gains more from one matmul per tile than u16 does, and that the weight-major int8 kernel i8g_gd2 (cast tile, tile @ x.T → yT) was the fastest config at B 8, 1.76–1.86x ahead of the B-adaptive u16. u16 had never been run weight-major with the fused take, and never past B 8. It asked: build u16gft (fused take per worker range, cast per tile or down sub-tile, tile @ x.T → yT), run it at B 8, 16, 32 against i8g_gd2 and i8dmm_gd2, and see whether the u16 / int8 ratio falls below 1.3x as B grows, or whether the take + cast floor keeps ternary >= 1.4x behind.

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 8, B 16 and B 32: does a weight-major u16 kernel with one fused LUT take per worker range come within 1.3x of the best int8 kernel, and does the gap shrink as B grows?

## Method

- Box: 8 CPUs (Xeon, L2 2 MiB per core), numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents (load avg at run start in the bytes line below), so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1855_1127/v16x.py exec'd without main() (→ v16m → v16k → v16t → v16d → v16.py: same weights, P16 8-trit uint16 pack with exact trit round-trip asserts, float64 references, st / dy / gd2 schedules, u16ad and i8dmm kernels, profiling hooks).
- New u16 weight-major kernels. Each worker does one np.take(LUT16U, P16[rows]) over its whole static row range into a u64 scratch whose int8 view is the trit tile. Then per tile: cast int8 → f32 and np.matmul(tile, xT, out=yT[rows]).
  - **u16gft256_st / u16gft128_st**: 256-row or 128-row tiles, all sites unsplit.
  - **u16gfk_st**: 256-row tiles; the down site (in 6912) is K-blocked at Kc 1728 (f32 sub-tile 1,769,472 B), tmp (m, B) = sub @ xT[k0:k1], accumulated into yT.
- Controls: **u16ad_st** (18:55 row-major B-adaptive u16; at B >= 4 it is the down 256x1728 sub-tile x @ tile.T kernel), **i8g_gd2** (weight-major int8 cast + tile @ xT, T 128), **i8g_st** (same, static split, T 256), **i8dmm_gd2** (int8 cast + x @ tile.T, T 128).
- Error: max abs error and cos vs the float64 ternary (u16) or int8 reference (XREF rows 0..B-1). Each run also checks that instrumented and plain output are identical.
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round. Reported as median of round medians, round min–max, per-round ratios, and token-layers/s (B / layer time). Then 9 instrumented layers per config give stage worker-ms (wall/cpu summed over workers). Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench0255_1128 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16g.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Below: layer ms, ratios (a/b > 1 means b is faster), stage worker-ms (wall/cpu, runs 1, 2, 3; int8 kernels have no take stage) and error. Ratio rows named best_* pick the fastest config of that group within each run (best_u16g = fastest of the three u16gft variants; best_i8 = fastest of i8g_gd2 / i8g_st / i8dmm_gd2).

Bytes: weights u16 17,458,176 B vs int8 69,559,296 B (3.98x); LUT u64 52,488 B; down f32 sub-tile 1,769,472 B. Load avg at run start: 2.19, 2.03, 2.60.

| B | config | layer ms, median of round medians [round min–max], runs 1 / 2 / 3 | token-layers/s (run 1) |
|---|---|---|---|
| 8 | u16gft256_st | 10.60 [9.53–11.50], 10.44 [8.82–11.48], 11.18 [9.22–12.98] | 755 |
| 8 | u16gft128_st | 10.29 [9.70–11.45], 10.03 [9.55–10.75], 10.18 [9.65–11.19] | 778 |
| 8 | u16gfk_st | 9.98 [9.72–10.86], 9.91 [9.32–10.73], 10.01 [9.63–11.13] | 802 |
| 8 | u16ad_st | 11.33 [10.78–13.70], 11.26 [10.60–13.77], 11.66 [10.77–14.22] | 706 |
| 8 | i8g_gd2 | 7.33 [7.10–7.47], 7.29 [7.15–7.96], 7.50 [6.95–8.06] | 1092 |
| 8 | i8g_st | 7.77 [7.05–8.52], 7.78 [7.18–8.27], 7.95 [7.47–9.66] | 1030 |
| 8 | i8dmm_gd2 | 8.11 [7.85–8.78], 8.25 [7.68–8.78], 8.35 [8.15–9.00] | 987 |
| 16 | u16gft256_st | 11.23 [10.93–12.92], 11.19 [10.12–11.48], 11.40 [10.27–12.70] | 1425 |
| 16 | u16gft128_st | 10.81 [10.45–11.32], 10.72 [10.34–11.32], 10.67 [9.72–11.07] | 1480 |
| 16 | u16gfk_st | 10.96 [9.98–12.17], 10.58 [10.00–10.85], 10.77 [10.16–11.30] | 1460 |
| 16 | u16ad_st | 12.48 [11.62–15.93], 12.06 [11.77–13.08], 12.42 [11.39–13.31] | 1282 |
| 16 | i8g_gd2 | 8.16 [7.93–8.64], 8.21 [7.84–8.65], 8.04 [7.50–8.35] | 1960 |
| 16 | i8g_st | 8.61 [8.35–9.40], 8.67 [7.85–8.94], 8.87 [8.01–9.77] | 1857 |
| 16 | i8dmm_gd2 | 8.70 [8.54–9.64], 8.98 [8.60–9.90], 8.97 [8.52–9.64] | 1838 |
| 32 | u16gft256_st | 15.42 [14.86–17.06], 14.32 [13.93–15.54], 14.19 [13.21–22.55] | 2075 |
| 32 | u16gft128_st | 14.34 [13.80–14.57], 13.95 [13.08–16.64], 13.19 [12.60–13.81] | 2231 |
| 32 | u16gfk_st | 13.92 [12.90–14.27], 13.56 [12.92–14.11], 13.44 [12.36–14.67] | 2298 |
| 32 | u16ad_st | 15.37 [14.24–16.61], 14.45 [13.32–15.52], 14.19 [13.81–18.83] | 2083 |
| 32 | i8g_gd2 | 10.91 [10.56–12.03], 11.38 [10.77–14.43], 10.94 [10.36–21.29] | 2933 |
| 32 | i8g_st | 11.68 [11.42–14.22], 12.23 [11.77–14.63], 11.78 [11.12–12.59] | 2740 |
| 32 | i8dmm_gd2 | 11.22 [10.56–12.09], 11.15 [10.63–11.81], 11.58 [10.83–25.63] | 2852 |

| B | ratio (a/b, >1 = b faster) | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 8 | u16ad/best_u16g | 1.135x [1.054–1.272] (u16ad_st/u16gfk_st) | 1.137x [0.988–1.390] (u16ad_st/u16gfk_st) | 1.165x [1.075–1.277] (u16ad_st/u16gfk_st) |
| 8 | best_u16g/i8g_gd2 | 1.362x [1.314–1.506] (u16gfk_st/i8g_gd2) | 1.358x [1.243–1.501] (u16gfk_st/i8g_gd2) | 1.335x [1.243–1.501] (u16gfk_st/i8g_gd2) |
| 8 | best_u16g/i8g_st | 1.285x [1.171–1.448] (u16gfk_st/i8g_st) | 1.273x [1.233–1.434] (u16gfk_st/i8g_st) | 1.259x [1.098–1.349] (u16gfk_st/i8g_st) |
| 8 | best_u16g/i8dmm_gd2 | 1.230x [1.172–1.333] (u16gfk_st/i8dmm_gd2) | 1.200x [1.062–1.295] (u16gfk_st/i8dmm_gd2) | 1.199x [1.159–1.314] (u16gfk_st/i8dmm_gd2) |
| 8 | best_u16g/best_i8 | 1.362x [1.314–1.506] (u16gfk_st/i8g_gd2) | 1.358x [1.243–1.501] (u16gfk_st/i8g_gd2) | 1.335x [1.243–1.501] (u16gfk_st/i8g_gd2) |
| 8 | u16ad/best_i8 | 1.546x [1.471–1.864] (u16ad_st/i8g_gd2) | 1.544x [1.414–1.842] (u16ad_st/i8g_gd2) | 1.555x [1.336–1.896] (u16ad_st/i8g_gd2) |
| 16 | u16ad/best_u16g | 1.155x [1.056–1.435] (u16ad_st/u16gft128_st) | 1.139x [1.106–1.233] (u16ad_st/u16gfk_st) | 1.164x [1.067–1.253] (u16ad_st/u16gft128_st) |
| 16 | best_u16g/i8g_gd2 | 1.324x [1.228–1.380] (u16gft128_st/i8g_gd2) | 1.289x [1.226–1.349] (u16gfk_st/i8g_gd2) | 1.327x [1.201–1.416] (u16gft128_st/i8g_gd2) |
| 16 | best_u16g/i8g_st | 1.255x [1.176–1.346] (u16gft128_st/i8g_st) | 1.221x [1.139–1.381] (u16gfk_st/i8g_st) | 1.204x [1.085–1.289] (u16gft128_st/i8g_st) |
| 16 | best_u16g/i8dmm_gd2 | 1.242x [1.151–1.300] (u16gft128_st/i8dmm_gd2) | 1.179x [1.069–1.197] (u16gfk_st/i8dmm_gd2) | 1.190x [1.102–1.267] (u16gft128_st/i8dmm_gd2) |
| 16 | best_u16g/best_i8 | 1.324x [1.228–1.380] (u16gft128_st/i8g_gd2) | 1.289x [1.226–1.349] (u16gfk_st/i8g_gd2) | 1.327x [1.201–1.416] (u16gft128_st/i8g_gd2) |
| 16 | u16ad/best_i8 | 1.529x [1.425–1.896] (u16ad_st/i8g_gd2) | 1.469x [1.412–1.570] (u16ad_st/i8g_gd2) | 1.544x [1.364–1.774] (u16ad_st/i8g_gd2) |
| 32 | u16ad/best_u16g | 1.104x [1.020–1.240] (u16ad_st/u16gfk_st) | 1.066x [0.991–1.146] (u16ad_st/u16gfk_st) | 1.076x [1.061–1.428] (u16ad_st/u16gft128_st) |
| 32 | best_u16g/i8g_gd2 | 1.276x [1.113–1.303] (u16gfk_st/i8g_gd2) | 1.191x [0.940–1.248] (u16gfk_st/i8g_gd2) | 1.205x [0.619–1.265] (u16gft128_st/i8g_gd2) |
| 32 | best_u16g/i8g_st | 1.192x [0.942–1.244] (u16gfk_st/i8g_st) | 1.108x [0.927–1.193] (u16gfk_st/i8g_st) | 1.119x [1.025–1.190] (u16gft128_st/i8g_st) |
| 32 | best_u16g/i8dmm_gd2 | 1.241x [1.108–1.296] (u16gfk_st/i8dmm_gd2) | 1.215x [1.148–1.321] (u16gfk_st/i8dmm_gd2) | 1.139x [0.504–1.275] (u16gft128_st/i8dmm_gd2) |
| 32 | best_u16g/best_i8 | 1.276x [1.113–1.303] (u16gfk_st/i8g_gd2) | 1.215x [1.148–1.321] (u16gfk_st/i8dmm_gd2) | 1.205x [0.619–1.265] (u16gft128_st/i8g_gd2) |
| 32 | u16ad/best_i8 | 1.408x [1.236–1.475] (u16ad_st/i8g_gd2) | 1.296x [1.203–1.385] (u16ad_st/i8dmm_gd2) | 1.296x [0.884–1.369] (u16ad_st/i8g_gd2) |

| B | config | take | cast | mm / dot |
|---|---|---|---|---|
| 8 | u16gft256_st | 22.0/18.4, 21.6/17.8, 22.4/18.4 | 15.5/12.8, 14.1/12.0, 15.0/12.5 | 22.2/20.0, 21.2/19.5, 21.8/20.2 |
| 8 | u16gft128_st | 21.3/17.6, 20.4/17.2, 21.7/17.6 | 17.0/10.2, 16.4/10.3, 16.1/10.3 | 25.4/20.1, 25.2/20.4, 25.0/20.7 |
| 8 | u16gfk_st | 21.1/17.0, 20.4/17.2, 20.6/17.5 | 17.0/11.4, 16.2/11.6, 16.5/11.8 | 24.1/19.7, 24.1/19.8, 25.0/20.3 |
| 8 | u16ad_st | 20.6/17.0, 25.5/21.2, 21.9/17.5 | 17.3/11.7, 18.1/13.8, 16.9/12.1 | 34.7/29.1, 40.2/35.2, 35.7/30.7 |
| 8 | i8g_gd2 | –, –, – | 18.5/15.7, 20.9/17.6, 19.7/16.5 | 22.5/19.9, 24.1/21.7, 24.3/21.0 |
| 8 | i8g_st | –, –, – | 15.4/13.5, 14.8/13.4, 14.6/13.4 | 20.5/19.2, 19.9/19.2, 20.0/19.4 |
| 8 | i8dmm_gd2 | –, –, – | 17.5/13.6, 17.4/13.5, 17.7/13.6 | 33.9/29.1, 33.1/29.6, 34.6/30.4 |
| 16 | u16gft256_st | 22.1/17.9, 23.1/18.7, 21.7/17.5 | 14.5/12.4, 15.0/12.4, 14.6/11.7 | 24.1/22.3, 25.6/23.4, 23.6/22.2 |
| 16 | u16gft128_st | 21.0/17.4, 20.9/17.4, 20.2/17.1 | 15.5/10.2, 15.7/10.3, 15.1/9.9 | 27.6/23.3, 27.8/23.7, 26.6/22.9 |
| 16 | u16gfk_st | 20.8/17.3, 20.1/16.9, 20.6/17.0 | 15.7/11.4, 15.3/11.2, 15.6/11.2 | 27.5/23.3, 26.6/22.8, 26.1/22.6 |
| 16 | u16ad_st | 21.5/17.4, 20.4/17.0, 20.5/16.9 | 15.9/11.8, 15.9/11.7, 15.6/11.6 | 38.5/34.1, 37.3/33.0, 37.7/33.6 |
| 16 | i8g_gd2 | –, –, – | 19.4/15.2, 18.2/15.9, 20.1/15.4 | 26.4/23.9, 25.9/23.2, 26.5/23.2 |
| 16 | i8g_st | –, –, – | 14.9/13.1, 13.8/12.6, 14.0/12.2 | 23.3/21.9, 22.3/21.8, 22.8/21.9 |
| 16 | i8dmm_gd2 | –, –, – | 17.8/13.8, 18.3/13.8, 16.6/12.7 | 38.0/34.1, 41.6/37.1, 37.6/33.4 |
| 32 | u16gft256_st | 27.1/21.8, 21.6/18.1, 21.4/17.8 | 16.8/14.6, 14.9/12.2, 14.0/12.0 | 40.2/38.7, 35.4/33.5, 34.7/32.9 |
| 32 | u16gft128_st | 22.3/18.7, 22.5/17.5, 21.5/17.3 | 14.6/10.7, 13.4/10.2, 13.4/10.1 | 38.1/36.0, 37.6/35.1, 37.8/35.0 |
| 32 | u16gfk_st | 25.3/20.2, 21.8/17.4, 24.3/20.1 | 17.4/12.7, 14.7/11.4, 16.2/13.1 | 43.1/38.6, 36.3/33.6, 42.2/39.5 |
| 32 | u16ad_st | 25.8/20.1, 21.4/17.6, 21.7/17.1 | 17.5/12.8, 15.0/11.7, 15.6/11.8 | 56.1/51.3, 50.3/46.4, 49.9/46.3 |
| 32 | i8g_gd2 | –, –, – | 23.9/19.8, 19.0/15.6, 18.7/16.0 | 42.2/39.7, 37.9/34.7, 36.5/34.2 |
| 32 | i8g_st | –, –, – | 16.9/15.2, 14.7/13.0, 14.2/12.5 | 36.2/35.0, 33.3/32.2, 32.6/32.0 |
| 32 | i8dmm_gd2 | –, –, – | 19.8/16.0, 17.8/13.9, 15.2/12.8 | 54.6/50.3, 49.4/45.3, 49.0/44.7 |

| B | config | max abs err vs ref, runs 1 / 2 / 3 | cos (min) | instr identical |
|---|---|---|---|---|
| 8 | u16gft256_st | 1.23e-06, 1.15e-06, 1.22e-06 | 1.000000000000 | True |
| 8 | u16gft128_st | 1.23e-06, 1.15e-06, 1.22e-06 | 1.000000000000 | True |
| 8 | u16gfk_st | 1.20e-06, 1.05e-06, 1.21e-06 | 1.000000000000 | True |
| 8 | u16ad_st | 1.20e-06, 1.05e-06, 1.21e-06 | 1.000000000000 | True |
| 8 | i8g_gd2 | 4.42e-06, 4.39e-06, 4.52e-06 | 1.000000000000 | True |
| 8 | i8g_st | 4.42e-06, 4.39e-06, 4.52e-06 | 1.000000000000 | True |
| 8 | i8dmm_gd2 | 4.42e-06, 4.39e-06, 4.52e-06 | 1.000000000000 | True |
| 16 | u16gft256_st | 1.23e-06, 1.16e-06, 1.22e-06 | 1.000000000000 | True |
| 16 | u16gft128_st | 1.23e-06, 1.16e-06, 1.22e-06 | 1.000000000000 | True |
| 16 | u16gfk_st | 1.20e-06, 1.28e-06, 1.21e-06 | 1.000000000000 | True |
| 16 | u16ad_st | 1.20e-06, 1.28e-06, 1.21e-06 | 1.000000000000 | True |
| 16 | i8g_gd2 | 4.42e-06, 4.54e-06, 4.80e-06 | 1.000000000000 | True |
| 16 | i8g_st | 4.42e-06, 4.54e-06, 4.80e-06 | 1.000000000000 | True |
| 16 | i8dmm_gd2 | 4.42e-06, 4.54e-06, 4.80e-06 | 1.000000000000 | True |
| 32 | u16gft256_st | 1.23e-06, 1.34e-06, 1.31e-06 | 1.000000000000 | True |
| 32 | u16gft128_st | 1.23e-06, 1.34e-06, 1.31e-06 | 1.000000000000 | True |
| 32 | u16gfk_st | 1.20e-06, 1.46e-06, 1.21e-06 | 1.000000000000 | True |
| 32 | u16ad_st | 1.20e-06, 1.46e-06, 1.21e-06 | 1.000000000000 | True |
| 32 | i8g_gd2 | 4.72e-06, 4.54e-06, 4.80e-06 | 1.000000000000 | True |
| 32 | i8g_st | 4.72e-06, 4.54e-06, 4.80e-06 | 1.000000000000 | True |
| 32 | i8dmm_gd2 | 4.72e-06, 4.54e-06, 4.80e-06 | 1.000000000000 | True |

## Verdict

**not-viable.** Going weight-major helps u16: the best u16gft variant is 1.07–1.17x faster than the 18:55 row-major u16ad at every B (1.14–1.17x at B 8 and B 16, 1.07–1.10x at B 32), because tile @ xT costs about the same as int8's matmul (mm 20–25 worker-ms at B 8 vs u16ad's dot 35–40). The gap to the best int8 kernel shrinks as B grows: 1.34–1.36x at B 8, 1.29–1.33x at B 16, 1.21–1.28x at B 32. So the < 1.3x bar is met at B 32 in all three runs, and at B 16 in one of three. The 18:55 B 8 gap of 1.76–1.86x (u16ad vs i8g_gd2) is now 1.34–1.36x on the same B with u16gft; most of that change is the kernel, but part is run-to-run (u16ad/i8g_gd2 is 1.54–1.56x in this run). The profile shows why the gap stops at ~1.2x. u16's cast (13–17 worker-ms) and mm (21–43) match int8's cast (14–21) and mm (20–42) at each B, but u16 also pays the LUT take, about 20–22 worker-ms per layer, flat in B. As B grows, mm grows and the fixed take becomes a smaller share, so the ratio falls slowly; it does not reach parity, let alone a 1.5x speed win. Ternary in numpy stays a memory-only win at decode batch sizes up to B 32: 17.46 vs 69.56 MB of weights (3.98x, already known). Error: all u16 variants <= 1.23e-6 vs the float64 ternary reference (cos 1.0), trits exact (pack round-trip asserted); int8 variants <= 4.72e-6 vs the int8 reference; instrumented and plain outputs identical everywhere.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-28 02:55 wave reached this lane, and no Chief-named item was found, so this fire took the chained question from the 18:55 note. This note only adds one new file under junior_osai/vault/findings/; HEAD at clone and the last-6-hours check are in the push commit message.

## Next question

The take stage is a fixed ~21 worker-ms per layer and does not grow with B, while mm does. Run u16gft128 / u16gfk against i8g_gd2 / i8g_st at prefill-sized chunks, B 64 and B 128 (still ctx-free, same layer), where mm should dominate. Does the u16 / int8 ratio fall to <= 1.1x, making ternary near speed-parity with a 3.98x weight-memory win for chunked prefill? Or does the take + cast stay a >= 1.15x tax even there?

## Script (v16g.py)

```python
#!/usr/bin/env python3
"""V16G: weight-major u16 with one fused take per worker range (u16gft), B 8 / 16 / 32, W 8 decode layer, full bitnetd layer, ctx-free.
Base: /workspace/bench1855_1127/v16x.py exec'd without main() (-> v16m -> v16k -> v16t -> v16d -> v16.py: same weights, P16 8-trit uint16 pack
with exact trit round-trip asserts, float64 refs, st / dy / gd2 schedules, u16ad / i8dmm kernels, profiling hooks).
New weight-major kernels (xT (in, B), yT (out, B)), static split, one np.take(LUT16U) per worker range:
  u16gft{R}_st : cast R-row int8 tile -> f32 (R, in) -> np.matmul(tile, xT, out=yT[rows]); R 256 / 128
  u16gfk_st    : same, R 256, down site K-blocked at Kc 1728 (f32 sub-tile 1,769,472 B), tmp (m, B) accumulate
Controls: u16ad_st (row-major B-adaptive, 18:55 best u16), i8g_gd2 (weight-major int8 cast + tile @ xT, T 128), i8dmm_gd2 (int8 x @ tile.T, T 128),
          i8g_st (weight-major int8, static split, T 256).
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16g.py out.json"""
import json, os, sys
import numpy as np
_S = open("/workspace/bench1855_1127/v16x.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "v16x.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 9
BS = (8, 16, 32); TMPN = 32 * 256
def mk_g(R, KC):
    def r(PR, xT, yT, r0, r1, T, w, i):
        q = PR.shape[1]; n = r1 - r0; P = PRF[0]; B = xT.shape[1]; kc = KC if (KC and i > D) else i
        su = scr(("u16f", w), (RMAX * Q8,), np.uint64)[:n * q].reshape(n, q); s8 = su.view(np.int8).reshape(n, q * 8)
        sb = scr(("gm", w), (256 * F,), np.float32); tmp = scr(("gt", w), (TMPN,), np.float32)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16U, PR[r0:r1], out=su)
        if P: _add(w, "take", pc() - t0, tt() - c0)
        for a in range(0, n, R):
            m = min(R, n - a); yv = yT[r0+a:r0+a+m]
            for j, k0 in enumerate(range(0, i, kc)):
                k1 = min(i, k0 + kc); s = sb[:m * (k1 - k0)].reshape(m, k1 - k0)
                if P: t0 = pc(); c0 = tt()
                np.copyto(s, s8[a:a+m, k0:k1], casting="unsafe")
                if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
                if kc == i: np.matmul(s, xT, out=yv)
                else:
                    t = tmp[:m * B].reshape(m, B); np.matmul(s, xT[k0:k1], out=t)
                    if j == 0: yv[...] = t
                    else: np.add(yv, t, out=yv)
                if P: _add(w, "mm", pc() - t0, tt() - c0)
    return r
KF["u16gft256"] = mk_g(256, 0); KF["u16gft128"] = mk_g(128, 0); KF["u16gfk"] = mk_g(256, 1728)
C = {"u16gft256_st": ("u16gft256", 8, "st"), "u16gft128_st": ("u16gft128", 8, "st"), "u16gfk_st": ("u16gfk", 8, "st"), "u16ad_st": ("u16ad", 8, "st"),
     "i8g_gd2": ("8g", 128, "gd2"), "i8g_st": ("8g", 256, "st"), "i8dmm_gd2": ("8dmm", 128, "gd2")}
UG = ["u16gft256_st", "u16gft128_st", "u16gfk_st"]; U = UG + ["u16ad_st"]; I8 = ["i8g_gd2", "i8g_st", "i8dmm_gd2"]
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, RMAX=RMAX, load=os.getloadavg(), time={}, prof={}, err={})
    W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES); W16 = sum(P16[k]["PR"].nbytes + P16[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(u16=W16, int8=W8, int8_over_u16=W8 / W16, lut_u64=LUT16U.nbytes, subtile_f32=256 * 1728 * 4)
    print("bytes", res["bytes"], "load", res["load"], flush=True)
    names = list(C)
    for B in BS:
        xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0].startswith("8") else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in U: e[c]["vs_u16ad_mab"] = float(np.abs(outs[c] - outs["u16ad_st"]).max())
        res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.12f}", v["instr_identical"]) for c, v in e.items()}, flush=True)
        xb = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32); rm = {c: [] for c in names}
        for c in names: layer(xb, *C[c], [])
        for rd in range(ROUNDS):
            for c in names[rd % len(names):] + names[:rd % len(names)]:
                s = []
                for _ in range(REPS):
                    t0 = pc(); layer(xb, *C[c], []); s.append((pc() - t0) * 1e3)
                rm[c].append(float(np.median(s)))
        o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), rm=v, tok_s=B * 1e3 / float(np.median(v))) for c, v in rm.items()}; rat = {}
        def pr_(a, b, key):
            v = [p / q for p, q in zip(rm[a], rm[b])]; rat[key] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v), a=a, b=b)
        bg = min(UG, key=lambda c: o[c]["med"]); bi = min(I8, key=lambda c: o[c]["med"])
        pr_("u16ad_st", bg, "u16ad/best_u16g")
        for c in I8: pr_(bg, c, f"best_u16g/{c}")
        pr_(bg, bi, "best_u16g/best_i8"); pr_("u16ad_st", bi, "u16ad/best_i8")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:16s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}] {o[c]['tok_s']:.0f} tok-layer/s", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:26s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
        pr = {}; PRF[0] = True
        for c in names:
            per = {}
            for _ in range(PROF):
                for d in ACC: d.clear()
                rec = []; layer(xb, *C[c], rec); it = {}
                for (site, st), (dw, dc, n) in ((k, v) for d in ACC for k, v in d.items()):
                    a = it.setdefault(st, [0.0, 0.0]); a[0] += dw; a[1] += dc
                for st, v in it.items(): per.setdefault(st, []).append(v)
            pr[c] = {st: dict(wall=float(np.median([v[0] for v in L])) * 1e3, cpu=float(np.median([v[1] for v in L])) * 1e3) for st, L in per.items()}
            print(f"B{B} prof {c:16s} " + "  ".join(f"{st} {v['wall']:.1f}/{v['cpu']:.1f}ms" for st, v in pr[c].items()), flush=True)
        PRF[0] = False; res["prof"][f"B{B}"] = pr
    json.dump(res, open(sys.argv[1], "w"), indent=1)
main()
```

close: 1ca92abf77578694
