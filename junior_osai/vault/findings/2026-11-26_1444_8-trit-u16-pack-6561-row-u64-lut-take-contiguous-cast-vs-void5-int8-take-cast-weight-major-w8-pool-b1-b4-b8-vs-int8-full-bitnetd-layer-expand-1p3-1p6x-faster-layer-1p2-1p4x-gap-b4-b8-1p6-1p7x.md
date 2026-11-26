---
local: true
date: 2026-11-26
wave: "2026-11-26 14:55 slot (fired 14:44 MT, ran ~14:44–15:00 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (an 8-trit base-3 pack in uint16 (2 bit/weight) with a 6561-row u64 LUT makes the take output the contiguous int8 tile, so the cast stays the cheap contiguous one. On the W 8 decode layer, full bitnetd layer, ctx-free, the expand stage (take + cast) is 1.35–1.64x faster than void5 take5 + cast at B 4 and 1.35–1.42x at B 8 (gd2; bar was >= 1.3x: met), and 1.40–1.65x / 1.44–1.55x when the uint16 index is first widened to intp (u16x). Plain layer best void5 / best u16 = 1.37–1.42x (B 4), 1.30–1.44x (B 8), 1.22–1.27x (B 1); outputs bit-identical to void5. This is the new best weight-major ternary path, but t5-family vs int8 best-vs-best stays 1.59–1.76x slower at B 4 / B 8 (was 1.96–2.09x / 1.94–2.02x with void5) and tmL stays best at B 1 (1.76–1.83x slower than int8d). Memory: 17.46 MB vs 13.99 MB t5 (int8 / u16 = 3.98x, down from 4.97x); LUT 52,488 B; expand scratch unchanged (70.8 MB). No 1.5x speed or memory win vs int8 beyond the known weight-memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, u64, uint16, eight-trit-pack, base-3, 2-bit, void5, np-take, contiguous-cast, weight-major, int8, numpy, not-viable]
---

# 8-trit uint16 pack + 6561-row u64 LUT: the take writes the contiguous int8 tile, expand is 1.3–1.6x faster than void5 at B 4 / B 8, layer 1.3–1.4x faster, still 1.6–1.7x behind int8

The 10:49 note ([[2026-11-26_1049_u64-padded-t5-lut-take8-strided-cast-vs-void5-int8-take-cast-weight-major-t5vg-w8-pool-b1-b4-b8-vs-int8-full-bitnetd-layer-take8-1p4x-faster-but-strided-cast-4p5x-slower-expand-0p6x-gap-b4-2x-b8-2x]]) found that an 8-byte u64 take is ~1.4x faster than the 5-byte void5 take, but slicing 5 of its 8 bytes forces a strided cast that is ~4.5x the contiguous one. It asked whether all 8 bytes can carry real trits: pack 8 trits per uint16 (3^8 = 6,561 codes, 2 bit/weight), gather from a 6,561 × u64 table, and the take output is exactly the contiguous int8 tile.

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 4 and B 8 (B 1 for reference), does np.take(LUT16U 6561 × u64, P16 uint16 tile, out=u64 scratch) + one contiguous int8 → f32 cast beat the current void5 take5 + cast (t5vg) by >= 1.3x for the expand stage? How much of the t5 vs int8 gap does it close, and what does giving up 4.97x → 4.0x weight memory buy?

## Method

- Box: 8 CPUs, numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network. The box is shared with other agents, so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1855_1125/gd.py (exec'd without main(); it execs bench0545_1125/xb5.py: same weights, P5 base-3 pack with exact trit round-trip assert, float64 references, dy_T / gd2 schedules), same harness as the 10:49 v8.py.
- P16 per site: trits decoded from the P5 pack, re-packed 8 per uint16 with digits {0, 1, 2→−1} × 3^j; exact round-trip of every trit asserted; all input widths (2560, 6912) are multiples of 8, so the tile is exactly (n, in) with no pad.
- Kernels (weight-major, T 256, same np.matmul tile (n, in) @ x.T):
  - **t5vg** (baseline): np.take(LUT5V 243 × V5, P5 tile, out=V5 scratch) → contiguous int8 view → f32 tile.
  - **u16g** (this question): np.take(LUT16U 6561 × u64, P16 uint16 tile, out=u64 scratch) → contiguous int8 view (n, in) → f32 tile.
  - **u16x** (extra): same, but np.copyto the uint16 tile into an intp scratch first (stage "idx"), so np.take gets intp indices and the index conversion is timed on its own.
  - Baselines: int8g_T128 (cast + weight-major matmul); at B 1 also int8d_T128 (cast + per-row np.dot) and t5 tmL_T384 (T-MAC, B 1 best).
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round; median of round medians, round min–max, and per-round ratios. Then 15 instrumented layers per config give per-stage worker time (perf_counter wall / thread_time CPU) summed over all workers and sites. 3 runs (XSEED 1, 2, 3).
- Error: max abs error and cosine vs the float64 ternary reference (int8 paths vs the float64 int8 reference); bit-identity to t5vg_dy checked.

Command (in /workspace/bench1455_1126):

    for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16.py run$s.json > run$s.txt; done; python3 agg.py > agg.txt

## Results

Weight bytes (4 sites, packed + per-row scales): t5 13,986,304; **u16 17,458,176**; int8 69,559,296 (int8 / t5 4.973x, **int8 / u16 3.984x**, u16 / t5 1.248x). LUT: void5 1,215 B, u16 52,488 B (fits L2). Expand scratch over 8 workers: t5vg 70,809,600 B, u16g 70,778,880 B, u16x 84,934,656 B (+intp index), int8g 35,389,440 B.

Plain layer, ms, median [round min–max], runs 1 / 2 / 3:

| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | t5vg_gd2 | 20.88 [19.30–21.32] | 20.87 [19.69–24.58] | 21.98 [21.02–24.32] |
| 1 | u16g_gd2 | 18.95 [17.48–20.64] | 18.04 [17.16–22.36] | 19.66 [18.12–21.54] |
| 1 | u16x_gd2 | 17.18 [15.78–20.87] | 16.38 [15.51–20.06] | 17.79 [15.79–19.54] |
| 1 | i8g_dy | 14.01 [13.47–15.89] | 14.28 [12.87–15.15] | 14.81 [12.91–16.13] |
| 1 | i8g_gd2 | 17.64 [15.53–19.48] | 18.08 [15.79–23.29] | 19.61 [18.23–20.75] |
| 1 | tmL_dy | 11.49 [10.81–12.43] | 10.79 [9.94–12.14] | 11.82 [11.42–13.02] |
| 1 | i8d_gd2 | 6.29 [5.91–7.56] | 6.00 [5.61–7.00] | 6.73 [6.62–7.82] |
| 4 | t5vg_gd2 | 19.63 [17.45–21.19] | 15.21 [14.58–16.43] | 17.11 [16.72–18.04] |
| 4 | u16g_gd2 | 16.43 [14.37–18.02] | 12.44 [11.74–13.41] | 13.72 [12.95–15.61] |
| 4 | u16x_gd2 | 14.23 [12.69–16.54] | 11.09 [10.33–12.08] | 12.03 [11.60–15.33] |
| 4 | i8g_dy | 8.58 [7.74–9.05] | 6.29 [6.01–7.53] | 7.57 [7.13–9.08] |
| 4 | i8g_gd2 | 10.13 [9.13–11.93] | 8.29 [8.03–8.89] | 9.05 [8.80–11.04] |
| 8 | t5vg_gd2 | 17.45 [15.99–19.21] | 16.59 [16.30–17.66] | 16.59 [15.80–17.45] |
| 8 | u16g_gd2 | 13.93 [13.54–16.29] | 13.80 [13.35–15.15] | 13.18 [12.44–14.85] |
| 8 | u16x_gd2 | 12.60 [12.03–15.29] | 12.82 [11.98–13.99] | 11.50 [10.91–14.11] |
| 8 | i8g_dy | 7.48 [7.23–7.91] | 7.55 [7.09–7.79] | 6.69 [6.37–7.76] |
| 8 | i8g_gd2 | 7.71 [7.47–10.74] | 7.66 [7.24–8.53] | 7.03 [6.62–7.81] |

Ratios (median of run medians; [per-round min–max]), runs 1 / 2 / 3:

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | t5vg_dy/u16g_dy | 1.126x [1.053–1.316] (t5vg_dy/u16g_dy) | 1.070x [0.879–1.121] (t5vg_dy/u16g_dy) | 1.100x [1.026–1.256] (t5vg_dy/u16g_dy) |
| 1 | t5vg_gd2/u16g_gd2 | 1.102x [1.007–1.167] (t5vg_gd2/u16g_gd2) | 1.157x [0.883–1.204] (t5vg_gd2/u16g_gd2) | 1.118x [1.069–1.270] (t5vg_gd2/u16g_gd2) |
| 1 | u16x_gd2/u16g_gd2 | 0.906x [0.832–1.089] (u16x_gd2/u16g_gd2) | 0.908x [0.828–0.931] (u16x_gd2/u16g_gd2) | 0.905x [0.844–0.993] (u16x_gd2/u16g_gd2) |
| 1 | best_void5/best_u16 | 1.216x [0.925–1.308] (t5vg_gd2/u16x_gd2) | 1.274x [0.984–1.333] (t5vg_gd2/u16x_gd2) | 1.236x [1.111–1.367] (t5vg_gd2/u16x_gd2) |
| 1 | best_u16/best_i8 | 2.731x [2.545–2.868] (u16x_gd2/i8d_gd2) | 2.730x [2.665–3.344] (u16x_gd2/i8d_gd2) | 2.642x [2.299–2.952] (u16x_gd2/i8d_gd2) |
| 1 | best_void5/best_i8 | 3.321x [2.552–3.493] (t5vg_gd2/i8d_gd2) | 3.479x [3.290–3.721] (t5vg_gd2/i8d_gd2) | 3.264x [2.897–3.586] (t5vg_gd2/i8d_gd2) |
| 1 | best_t/best_i8 | 1.826x [1.644–2.066] (tmL_dy/i8d_gd2) | 1.798x [1.734–1.851] (tmL_dy/i8d_gd2) | 1.756x [1.495–1.865] (tmL_dy/i8d_gd2) |
| 4 | t5vg_dy/u16g_dy | 1.190x [1.089–1.326] (t5vg_dy/u16g_dy) | 1.323x [1.271–1.408] (t5vg_dy/u16g_dy) | 1.303x [1.202–1.437] (t5vg_dy/u16g_dy) |
| 4 | t5vg_gd2/u16g_gd2 | 1.194x [1.096–1.361] (t5vg_gd2/u16g_gd2) | 1.223x [1.214–1.277] (t5vg_gd2/u16g_gd2) | 1.247x [1.121–1.339] (t5vg_gd2/u16g_gd2) |
| 4 | u16x_gd2/u16g_gd2 | 0.866x [0.858–0.930] (u16x_gd2/u16g_gd2) | 0.892x [0.854–0.926] (u16x_gd2/u16g_gd2) | 0.877x [0.845–0.982] (u16x_gd2/u16g_gd2) |
| 4 | best_void5/best_u16 | 1.379x [1.264–1.498] (t5vg_gd2/u16x_gd2) | 1.372x [1.355–1.474] (t5vg_gd2/u16x_gd2) | 1.421x [1.145–1.476] (t5vg_gd2/u16x_gd2) |
| 4 | best_u16/best_i8 | 1.659x [1.510–2.052] (u16x_gd2/i8g_dy) | 1.763x [1.530–1.799] (u16x_gd2/i8g_dy) | 1.590x [1.580–1.688] (u16x_gd2/i8g_dy) |
| 4 | best_void5/best_i8 | 2.288x [1.929–2.629] (t5vg_gd2/i8g_dy) | 2.418x [2.111–2.614] (t5vg_gd2/i8g_dy) | 2.260x [1.933–2.364] (t5vg_gd2/i8g_dy) |
| 4 | best_t/best_i8 | 1.659x [1.510–2.052] (u16x_gd2/i8g_dy) | 1.763x [1.530–1.799] (u16x_gd2/i8g_dy) | 1.590x [1.580–1.688] (u16x_gd2/i8g_dy) |
| 8 | t5vg_dy/u16g_dy | 1.268x [1.115–1.441] (t5vg_dy/u16g_dy) | 1.245x [1.160–1.296] (t5vg_dy/u16g_dy) | 1.310x [1.258–1.353] (t5vg_dy/u16g_dy) |
| 8 | t5vg_gd2/u16g_gd2 | 1.253x [1.162–1.279] (t5vg_gd2/u16g_gd2) | 1.203x [1.079–1.297] (t5vg_gd2/u16g_gd2) | 1.259x [1.117–1.332] (t5vg_gd2/u16g_gd2) |
| 8 | u16x_gd2/u16g_gd2 | 0.905x [0.888–0.943] (u16x_gd2/u16g_gd2) | 0.929x [0.868–0.969] (u16x_gd2/u16g_gd2) | 0.872x [0.845–0.950] (u16x_gd2/u16g_gd2) |
| 8 | best_void5/best_u16 | 1.384x [1.256–1.428] (t5vg_gd2/u16x_gd2) | 1.295x [1.169–1.445] (t5vg_gd2/u16x_gd2) | 1.442x [1.176–1.517] (t5vg_gd2/u16x_gd2) |
| 8 | best_u16/best_i8 | 1.686x [1.577–2.046] (u16x_gd2/i8g_dy) | 1.698x [1.569–1.800] (u16x_gd2/i8g_dy) | 1.720x [1.545–1.944] (u16x_gd2/i8g_dy) |
| 8 | best_void5/best_i8 | 2.334x [2.122–2.570] (t5vg_gd2/i8g_dy) | 2.199x [2.100–2.492] (t5vg_gd2/i8g_dy) | 2.480x [2.205–2.617] (t5vg_gd2/i8g_dy) |
| 8 | best_t/best_i8 | 1.686x [1.577–2.046] (u16x_gd2/i8g_dy) | 1.698x [1.569–1.800] (u16x_gd2/i8g_dy) | 1.720x [1.545–1.944] (u16x_gd2/i8g_dy) |

Per-stage worker time, ms wall / CPU summed over 8 workers and 4 sites (runs 1, 2, 3); span = sum over sites of the slowest worker:

| B | config | idx | take | cast | mm | task | span ms |
|---|---|---|---|---|---|---|---|
| 1 | t5vg_gd2 | – | 111.4 / 63.4, 105.1 / 60.3, 97.9 / 56.4 | 44.7 / 28.2, 43.4 / 27.3, 29.7 / 18.4 | 14.2 / 12.9, 13.8 / 12.7, 11.1 / 10.1 | 172.7 / 106.2, 165.1 / 102.2, 140.6 / 86.0 | 25.58, 24.54, 21.06 |
| 1 | u16g_gd2 | – | 80.5 / 32.7, 80.5 / 32.2, 70.1 / 25.5 | 42.0 / 24.3, 40.4 / 23.5, 32.3 / 16.0 | 13.0 / 11.6, 13.3 / 11.8, 11.1 / 9.9 | 138.6 / 71.8, 135.5 / 69.9, 114.4 / 53.5 | 20.67, 20.48, 17.19 |
| 1 | u16x_gd2 | 24.8 / 7.6, 22.9 / 5.7, 20.9 / 5.4 | 51.4 / 22.7, 49.1 / 18.4, 45.1 / 18.1 | 40.3 / 23.9, 31.4 / 16.4, 29.3 / 16.6 | 12.8 / 11.5, 11.6 / 10.0, 11.4 / 10.0 | 131.1 / 69.0, 114.1 / 52.8, 109.2 / 52.0 | 19.37, 16.76, 16.13 |
| 1 | i8g_dy | – | – | 92.3 / 17.6, 86.1 / 13.4, 84.1 / 13.8 | 8.1 / 8.1, 7.9 / 7.9, 7.5 / 7.5 | 102.1 / 27.2, 95.8 / 23.0, 93.7 / 23.3 | 15.46, 14.24, 14.54 |
| 4 | t5vg_gd2 | – | 66.3 / 51.8, 71.0 / 56.5, 69.9 / 55.0 | 22.9 / 20.7, 18.6 / 16.3, 21.0 / 18.1 | 25.6 / 23.5, 25.9 / 23.7, 25.0 / 22.7 | 116.5 / 98.3, 117.9 / 97.4, 118.3 / 97.7 | 17.05, 16.73, 17.35 |
| 4 | u16g_gd2 | – | 43.4 / 27.3, 41.8 / 23.2, 39.5 / 21.6 | 22.6 / 19.9, 16.2 / 13.7, 15.8 / 12.6 | 25.9 / 23.2, 24.4 / 20.5, 23.4 / 19.5 | 95.1 / 72.5, 84.5 / 59.2, 81.1 / 55.6 | 14.23, 12.44, 11.81 |
| 4 | u16x_gd2 | 12.9 / 6.1, 11.4 / 5.2, 10.9 / 4.5 | 28.8 / 19.1, 27.2 / 17.5, 27.8 / 16.1 | 21.9 / 19.3, 18.0 / 16.1, 16.4 / 13.2 | 25.2 / 22.5, 23.9 / 21.3, 23.8 / 19.9 | 91.5 / 69.4, 83.1 / 61.7, 80.5 / 56.3 | 13.21, 12.27, 11.96 |
| 4 | i8g_dy | – | – | 22.2 / 13.8, 19.9 / 12.8, 19.1 / 12.1 | 28.2 / 21.4, 26.6 / 21.0, 26.6 / 21.3 | 52.3 / 36.9, 49.2 / 35.6, 47.8 / 35.1 | 7.78, 7.29, 7.01 |
| 8 | t5vg_gd2 | – | 63.7 / 49.8, 65.0 / 49.9, 65.5 / 50.1 | 20.2 / 17.8, 21.1 / 19.1, 16.0 / 13.9 | 26.2 / 23.9, 26.8 / 24.2, 24.0 / 21.5 | 111.4 / 92.5, 114.9 / 94.6, 107.8 / 87.2 | 16.36, 16.91, 15.62 |
| 8 | u16g_gd2 | – | 42.5 / 25.2, 43.3 / 25.1, 41.1 / 22.1 | 19.6 / 17.0, 19.6 / 17.1, 16.1 / 13.3 | 26.0 / 22.7, 26.1 / 23.1, 24.7 / 20.9 | 90.1 / 66.7, 90.6 / 66.2, 83.2 / 58.0 | 13.49, 13.56, 12.16 |
| 8 | u16x_gd2 | 11.7 / 5.2, 11.5 / 5.2, 11.5 / 4.4 | 27.5 / 17.6, 27.7 / 17.5, 25.9 / 15.8 | 18.9 / 16.5, 18.1 / 16.3, 15.2 / 13.1 | 24.6 / 22.0, 24.9 / 22.3, 23.5 / 20.5 | 85.3 / 63.5, 83.9 / 63.0, 78.1 / 55.7 | 12.32, 12.54, 11.25 |
| 8 | i8g_dy | – | – | 20.1 / 12.7, 20.2 / 12.6, 16.9 / 11.0 | 27.6 / 21.5, 27.2 / 21.7, 24.9 / 20.5 | 49.3 / 35.9, 50.6 / 36.1, 43.4 / 33.1 | 7.36, 7.53, 6.41 |

Expand-stage ratio (t5vg idx + take + cast) / (u16 idx + take + cast), runs 1 / 2 / 3:

| B | u16g dy_T | u16g gd2 | u16x gd2 | take5 / take16 (gd2) |
|---|---|---|---|---|
| 1 | 1.110, 0.942, 1.157 | 1.274, 1.228, 1.247 | 1.340, 1.437, 1.339 | 1.384, 1.305, 1.397 |
| 4 | 1.454, 1.318, 1.274 | 1.351, 1.547, 1.643 | 1.402, 1.582, 1.648 | 1.529, 1.700, 1.768 |
| 8 | 1.431, 1.306, 1.426 | 1.352, 1.368, 1.424 | 1.444, 1.502, 1.550 | 1.500, 1.502, 1.593 |

Error (all three runs): every t5vg / u16g / u16x config is bit-identical to t5vg_dy and to its instrumented run. B 1 max abs 9.46e-07, cos 0.999999999999977 (run 1); B 4 / B 8 max abs 0.99–1.23e-06, cos 0.999999999999972–973. int8g: max abs 3.3–4.5e-06, cos 0.99999999999982–87 vs its own float64 int8 reference. Full per-config numbers (dy_T, B 1 i8d_dy) are in agg.txt / run1–3.txt beside v16.py.

### Reading

- The pad bytes now carry weights, so the take output is the int8 tile and the cast stays contiguous: u16g cast is 15.8–22.6 worker-ms at B 4 / B 8, same as void5's 16.0–22.9. All of the win is in the gather: take16 39.5–43.4 worker-ms vs take5 63.7–71.0 (1.50–1.77x), with CPU time roughly halved (21.6–27.3 vs 49.8–56.5 ms). One u64 item now carries 8 trits vs 5 per void5 item, so the take does 0.625x as many item copies, each on the fixed 8-byte path the 10:49 note found.
- u16x shows numpy's uint16 → intp index widening costs. Doing it ourselves into a reused intp scratch (idx 10.9–12.9 worker-ms) plus an intp take (25.9–28.8) is a few ms below the plain uint16 take, which allocates a fresh intp temporary per call. u16x is 0.87–0.93x of u16g on the layer at B 4 / B 8 (i.e. faster), for +14 MB scratch.
- Gap to int8: best ternary / best int8 is now 1.59–1.76x at B 4 and 1.69–1.72x at B 8 (u16x vs int8g dy_T), down from 2.26–2.42x and 2.20–2.48x for void5 in the same runs. What remains is the gather itself (idx + take ≈ 38–42 worker-ms that int8 does not have); cast and matmul are at parity with int8g.
- B 1: u16x beats void5 by 1.22–1.27x but the weight-major matmul is the wrong shape at B 1; tmL (T-MAC) stays the best ternary path (1.76–1.83x behind int8d, which runs per-row np.dot at 6.0–6.7 ms).
- Memory: weights go from 13.99 MB to 17.46 MB (+25%); int8 / u16 is 3.98x instead of 4.97x. LUT is 52 KB (L2-resident), scratch is unchanged for u16g. For an in-process bitnetd / i2sd decode this is a speed-for-memory trade inside the ternary family, not a new win over int8.

## Verdict

**not-viable** (vs int8). The bar set by the 10:49 note is met: the expand stage is 1.35–1.64x (B 4) and 1.35–1.42x (B 8) faster than void5 take5 + cast (u16x 1.40–1.65x / 1.44–1.55x), and the plain layer is 1.37–1.42x (B 4), 1.30–1.44x (B 8) faster, bit-identical. That makes the 8-trit uint16 pack the new best weight-major ternary decode path and the reference for later ternary-vs-ternary runs. But it is still 1.59–1.76x slower than int8 at B 4 / B 8, and it gives up 20% of the weight-memory win (4.97x → 3.98x vs int8). Neither gives a new >= 1.5x speed or memory win over int8 at acceptable error, so it is not pinged.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-26 14:55 wave reached this lane. JuniorLLM HEAD at clone was 886e40f (cloudcover95, 10:54 MT, this lane's 10:49 note). Nothing landed after it, so no Trit Pack, Handshake or LLC commit falls in the last 6 hours (the shallow clone shows only HEAD; the 10:49 note found none in the window before it). This note only adds one new file under junior_osai/vault/findings.

## Next question

At B 1 the weight-major matmul is the bottleneck shape, and int8 wins with per-row np.dot (6.0–6.7 ms). Does the u16x expand (intp idx + 6561 × u64 take + contiguous cast) followed by per-row np.dot (the t5vd shape) beat tmL T-MAC at B 1 and B 2 by >= 1.3x on the W 8 full layer? Same harness, same bit-identity check, weights 3.98x below int8.

## Script (v16.py, verbatim)

```python
#!/usr/bin/env python3
"""V16: 8-trit base-3 pack in uint16 (2 bit/weight) + 6561-row u64 LUT take whose output IS the contiguous int8 tile, vs the void5
int8 take + cast (t5vg), W=8 decode layer, full bitnetd layer, ctx-free, 14:4x MT 11/26.
Base: /workspace/bench1855_1125/gd.py (exec'd without main(); it execs bench0545_1125/xb5.py: weights, P5 base-3 pack + exact trit
round-trip assert, kernels, float64 refs, dy_T / gd2 schedules). Same harness as bench1055_1126/v8.py.
P16[k]: trits decoded from P5 (asserted), repacked 8 per uint16 (3^8 = 6561 codes), exact round-trip asserted.
t5 weight-major kernels (tile (n, in) @ x.T (in, B) -> yT), T = 256:
  t5vg  : np.take(LUT5V 243 x V5, P5 tile, out=V5 scratch) -> np.copyto int8 view -> f32 tile -> np.matmul        (current baseline)
  u16g  : np.take(LUT16U 6561 x u64 (8 int8 trits), P16 tile uint16, out=u64 scratch) -> contiguous int8 view (n, q8*8) -> f32 tile -> np.matmul
  u16x  : same, but the uint16 index is first widened into an intp scratch (np.copyto) so np.take gets intp (isolates numpy's index conversion)
int8 baselines: 8g_T128 (cast + weight-major matmul), and at B1 8d_T128 (cast + per-row np.dot) and t5 tmL_T384 (T-MAC, B1 best).
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16.py out.json"""
import json, os, sys, time, itertools
import numpy as np
_S = open("/workspace/bench1855_1125/gd.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "gd.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
L16 = np.array([[((c // 3 ** j) % 3) for j in range(8)] for c in range(6561)], np.int8); L16[L16 == 2] = -1
LUT16U = np.ascontiguousarray(L16).view(np.uint64).ravel()        # 6561 x 8-byte rows (52,488 B)
assert LUT16U.shape == (6561,) and np.array_equal(LUT16U.view(np.int8).reshape(6561, 8), L16)
POW8 = (3 ** np.arange(8)).astype(np.uint16)
def p16(k):
    P = P5[k]; o, q5 = P["PR"].shape; i = P["inp"]; assert i % 8 == 0, (k, i)
    t = L5[P["PR"]].reshape(o, q5 * 5)[:, :i]                      # trits decoded from the t5 pack
    d = np.where(t < 0, 2, t).astype(np.uint16).reshape(o, i // 8, 8)
    p = np.ascontiguousarray((d * POW8).sum(-1, dtype=np.uint16))
    assert p.max() < 6561 and np.array_equal(L16[p].reshape(o, i), t), k   # exact round-trip of every trit
    return dict(PR=p, g=P["g"], split=P["split"], inp=i)
P16 = {k: p16(k) for k in SITES}
PRF = [False]; ACC = [dict() for _ in range(W)]; CUR = ["?"]
def _add(w, st, dw, dc):
    a = ACC[w].get((CUR[0], st))
    if a is None: ACC[w][(CUR[0], st)] = [dw, dc, 1]
    else: a[0] += dw; a[1] += dc; a[2] += 1
Q8 = F // 8
def r_t5vg(PR, xT, yT, r0, r1, T, w, i):
    q = PR.shape[1]; sv = scr(("v5", w, T), (T * F5,), "V5")[:T * q].reshape(T, q); s = scr(("f5", w, T), (T * F5 * 5,), np.float32)[:T * q * 5].reshape(T, q * 5); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT5V, PR[o0:o0+n], out=sv[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.copyto(s[:n], sv[:n].view(np.int8).reshape(n, q * 5), casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n, :i], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
def r_u16g(PR, xT, yT, r0, r1, T, w, i):
    q = PR.shape[1]; su = scr(("u16", w, T), (T * Q8,), np.uint64)[:T * q].reshape(T, q)
    s = scr(("f16", w, T), (T * F,), np.float32)[:T * q * 8].reshape(T, q * 8); s8 = su.view(np.int8).reshape(T, q * 8); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16U, PR[o0:o0+n], out=su[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.copyto(s[:n], s8[:n], casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
def r_u16x(PR, xT, yT, r0, r1, T, w, i):
    q = PR.shape[1]; su = scr(("u16", w, T), (T * Q8,), np.uint64)[:T * q].reshape(T, q); ix = scr(("ix16", w, T), (T * Q8,), np.intp)[:T * q].reshape(T, q)
    s = scr(("f16", w, T), (T * F,), np.float32)[:T * q * 8].reshape(T, q * 8); s8 = su.view(np.int8).reshape(T, q * 8); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.copyto(ix[:n], PR[o0:o0+n])
        if P: _add(w, "idx", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.take(LUT16U, ix[:n], out=su[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.copyto(s[:n], s8[:n], casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
def r_8g(Q, xT, yT, r0, r1, T, w, i):
    _, s = par.scratch(w, T, i // 4); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.copyto(s[:n], Q[o0:o0+n], casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
KF = {"t5vg": r_t5vg, "u16g": r_u16g, "u16x": r_u16x, "8g": r_8g}
_mm_gd = mm
def mm(k, a, kind, T, mode, rec):
    CUR[0] = k
    if kind in ("tmL", "8d"): return _mm_gd(k, a, kind, T, mode, rec)
    B = a.shape[0]
    if kind == "8g": P = par.F8[k]; R = P["Q"]; i = R.shape[1]
    elif kind == "t5vg": P = P5[k]; R = P["PR"]; i = P["inp"]
    else: P = P16[k]; R = P["PR"]; i = P["inp"]
    out = R.shape[0]; xT = np.ascontiguousarray(np.asarray(a, np.float32).T); yT = np.empty((out, B), np.float32)
    run_site(KF[kind], R, xT, yT, out, T, i, mode, rec, k, "t5vg" if kind != "8g" else "8g")
    y = np.ascontiguousarray(yT.T); y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
G = {"t5vg_dy": ("t5vg", 256, "dy"), "t5vg_gd2": ("t5vg", 256, "gd2"), "u16g_dy": ("u16g", 256, "dy"), "u16g_gd2": ("u16g", 256, "gd2"),
     "u16x_gd2": ("u16x", 256, "gd2"), "i8g_dy": ("8g", 128, "dy"), "i8g_gd2": ("8g", 128, "gd2")}
CF = {1: dict(G, tmL_dy=("tmL", 384, "dy"), i8d_dy=("8d", 128, "dy"), i8d_gd2=("8d", 128, "gd2")), 4: G, 8: G}
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, time={}, prof={}, err={})
    W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
    W16 = sum(P16[k]["PR"].nbytes + P16[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(t5=W5, u16=W16, int8=W8, int8_over_t5=W8 / W5, int8_over_u16=W8 / W16, u16_over_t5=W16 / W5, lut_v5=LUT5V.nbytes, lut_u16=LUT16U.nbytes,
                        scr_t5vg=W * 256 * F5 * (5 + 20), scr_u16g=W * 256 * Q8 * (8 + 32), scr_u16x=W * 256 * Q8 * (8 + 8 + 32), scr_i8g=par.scratch_bytes(W, 128))
    print("bytes", res["bytes"], flush=True)
    for B in (1, 4, 8):
        C = CF[B]; names = list(C); xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0] in ("8g", "8d") else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in names:
            if C[c][0] in ("u16g", "u16x", "t5vg"): e[c]["identical_t5vg_dy"] = bool(np.array_equal(outs[c], outs["t5vg_dy"])); e[c]["vs_t5vg_dy_mab"] = float(np.abs(outs[c] - outs["t5vg_dy"]).max())
        res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.15f}", v["instr_identical"], v.get("identical_t5vg_dy")) for c, v in e.items()}, flush=True)
        xb = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32); rm = {c: [] for c in names}
        for c in names: layer(xb, *C[c], [])
        for rd in range(ROUNDS):
            for c in names[rd % len(names):] + names[:rd % len(names)]:
                s = []
                for _ in range(REPS):
                    t0 = pc(); layer(xb, *C[c], []); s.append((pc() - t0) * 1e3)
                rm[c].append(float(np.median(s)))
        o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), rm=v) for c, v in rm.items()}
        rat = {}
        def pr_(a, b, key=None):
            v = [p / q for p, q in zip(rm[a], rm[b])]; rat[key or f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v), a=a, b=b)
        for m in ("dy", "gd2"): pr_(f"t5vg_{m}", f"u16g_{m}"); pr_(f"u16g_{m}", f"i8g_{m}"); pr_(f"t5vg_{m}", f"i8g_{m}")
        pr_("t5vg_gd2", "u16x_gd2"); pr_("u16x_gd2", "u16g_gd2")
        i8n = [c for c in names if C[c][0] in ("8g", "8d")]; bi = min(i8n, key=lambda c: o[c]["med"])
        bo = min(("t5vg_dy", "t5vg_gd2"), key=lambda c: o[c]["med"]); bv = min(("u16g_dy", "u16g_gd2", "u16x_gd2"), key=lambda c: o[c]["med"])
        bt = min((c for c in names if c not in i8n), key=lambda c: o[c]["med"])
        pr_(bo, bv, "best_void5/best_u16"); pr_(bv, bi, "best_u16/best_i8"); pr_(bo, bi, "best_void5/best_i8"); pr_(bt, bi, "best_t/best_i8")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:9s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
        pr = {}; PRF[0] = True
        for c in [n for n in names if C[n][0] in KF]:
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
            print(f"B{B} prof {c:9s} " + "  ".join(f"{st} {v['wall']:.1f}/{v['cpu']:.1f}ms" for st, v in pr[c].items()), flush=True)
        PRF[0] = False; res["prof"][f"B{B}"] = pr
    json.dump(res, open(sys.argv[1], "w"), indent=1)
main()
```

close: c5190933898221ea
