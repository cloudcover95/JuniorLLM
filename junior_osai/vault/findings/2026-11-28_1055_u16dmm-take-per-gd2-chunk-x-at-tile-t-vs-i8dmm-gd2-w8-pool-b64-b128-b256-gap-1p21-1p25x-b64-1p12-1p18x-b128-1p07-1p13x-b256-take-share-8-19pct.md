---
local: true
date: 2026-11-28
wave: "2026-11-28 10:55 slot (fired 10:49 MT, ran ~10:50–10:58 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (prefill chunks B 64 / 128 / 256, W 8 pool, full bitnetd layer, ctx-free: u16dmm_gd2 = one 8-trit uint16 LUT take per gd2 chunk, cast int8 -> f32, x @ tile.T, on the exact i8dmm_gd2 chunk list. Moving u16 into the i8dmm orientation is the big step: it is 1.15–1.22x faster than the 05:55 best u16 (u16gfk_st) at B 64, 1.37–1.41x at B 128, 1.58–1.70x at B 256, and the gap to i8dmm_gd2 falls from 1.43–1.64x (05:55) to 1.21–1.25x at B 64, 1.12–1.18x at B 128, 1.07–1.13x at B 256 (best u16dmm variant incl. down K-blocked u16dmk: 1.20–1.25x / 1.11–1.15x / 1.07–1.11x). The <= 1.15x bar is met at B 256 (all runs), at B 128 in 2 of 3 runs for u16dmm and 3 of 3 for the best variant, and missed at B 64. The dot stage is now the same as int8's; the remaining gap is the take (8–9% of u16 worker cpu-ms at B 256, 12–13% at B 128, 18–19% at B 64, a flat ~17–23 cpu-ms per layer). 128-row chunks (u16dmm128) are no faster than the 256-row i8dmm list (slower at B 64 / 128, tied at B 256). Ternary is still slower than int8 at every B, so no speed win; error <= 1.45e-6 vs the float64 ternary reference (cos 1.0), trits exact; int8 <= 5.32e-6. Memory unchanged: 17.46 vs 69.56 MB (3.98x, already known). No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, prefill, chunked-prefill, thread-pool, guided-schedule, gd2, lut-expand, uint16, eight-trit-pack, base-3, 2-bit, np-take, x-at-tile-t, k-blocking, np-matmul, int8-control, int8, numpy, not-viable]
---

# u16dmm (take per gd2 chunk + x @ tile.T) closes most of the prefill gap to i8dmm: 1.21–1.25x at B 64, 1.12–1.18x at B 128, 1.07–1.13x at B 256; what is left is the take

The 05:55 note ([[2026-11-28_0555_prefill-chunks-b64-b128-weight-major-u16gft-u16gfk-vs-i8g-i8dmm-w8-pool-u16-vs-i8g-1p04-1p30x-but-i8dmm-x-at-tile-t-wins-gap-1p43-1p58x-b64-1p58-1p64x-b128]]) found that at prefill sizes the int8 winner is i8dmm_gd2 (x @ tile.T under the guided chunk queue), not the weight-major kernels, and the best u16 (weight-major u16gft / u16gfk) trailed it by 1.43–1.58x at B 64 and 1.58–1.64x at B 128. The take was only about 20% of u16 work at B 128, so the gap looked like orientation and schedule. It asked for u16dmm_gd2: a per-chunk np.take(LUT16U) into the int8 tile, cast to f32, then x @ tile.T, run against i8dmm_gd2 at B 64 / 128 / 256.

## Question

On the W 8 pool layer (full bitnetd layer, ctx-free), at B 64, 128 and 256: does u16dmm_gd2 (8-trit uint16 LUT take per gd2 chunk → int8 tile → f32 → x @ tile.T) come within 1.15x of i8dmm_gd2 (int8 → f32 → x @ tile.T, same chunk list)? That would show the 05:55 gap was orientation and schedule. Or does the per-chunk take (no longer one fused take per worker range) keep it above 1.3x?

## Method

- Box: 8 CPUs (Xeon, L2 2 MiB per core), numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents (load avg at run start in the bytes line below), so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1055_1128/v16q.py is bench0555_1128/v16p.py (kernels and harness unchanged; it execs /workspace/bench1855_1127/v16x.py without main() → v16m → v16k → v16t → v16d → v16.py: same weights, P16 8-trit uint16 pack with exact trit round-trip asserts, st / gd2 schedules, i8dmm kernel mk_8, profiling hooks) with B (64, 128, 256), the matmul tmp scratch raised to 256 × 256 floats, and the new row-major kernels below.
- gd2 chunk list: i8dmm_gd2 runs with T 128 under run_site kind "8d", so its guided list is 256-row chunks while >= W × T rows remain, then 64-row tail chunks, and the kernel gets one 256-row (or 64-row) tile per chunk.
- New: **u16dmm_gd2** uses that same chunk list (run_site kind "8d"). Per chunk: np.take(LUT16U, PR[r0:r1]) into a u64 scratch (n × in/8) viewed as the int8 tile, np.copyto cast int8 → f32 (n, in), np.matmul(x, tile.T) → tmp (B, n) → y[:, r0:r1]. That is mk_8 / i8dmm with the take in front of the cast. **u16dmm128_gd2** is the chunk-size variant: 128-row chunks, then 64-row tail, at every site (kind "x128"), with the matched control **i8dmm128_gd2** (mk_8 on the same list). **u16dmk_gd2** is u16dmm_gd2 with the down site (in 6912) K-blocked at Kc 1728, take once per chunk, with the matched control **i8dmk_gd2** (the 18:55 int8 one-matmul down-K-blocked kernel). **u16gfk_st** (05:55 best u16 at B 128) is kept as the link to the previous wave.
- Reference: each run draws a 256-row XREFP (rng XSEED + 2000) and builds a float64 reference from the trits decoded from P16 (int8: F8 Q) times g in float64. That path is first checked against the original wf64 reference on XREF[:32]: ternary differs by 0.0, int8 by 9.9e-8 (F8 g is float32), asserted ternary <= 1e-12 and int8 <= 1e-6. The P16 decode is asserted to equal the t5-pack trits at every site.
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round. Reported as median of round medians, round min–max, per-round ratios, and token-layers/s (B / layer time). Then 9 instrumented layers per config give stage worker-ms (wall/cpu summed over workers) and the take share. Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench1055_1128 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16q.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Below: layer ms, ratios (a/b > 1 means b is faster), stage worker-ms (wall/cpu, runs 1, 2, 3; int8 kernels have no take stage), error and take share. Ratio rows named best_* pick the fastest config of that group within each run (best_u16dmm = fastest of u16dmm / u16dmm128 / u16dmk; best_i8 = fastest of i8dmm / i8dmm128 / i8dmk; best_u16 also includes u16gfk_st). u16dmk_gd2 output is bit-identical to u16gfk_st; u16dmm / u16dmm128 differ from it by <= 7.2e-7 (summation order of the unsplit down site).

Bytes: weights u16 17,458,176 B vs int8 69,559,296 B (3.98x); LUT u64 52,488 B; down f32 sub-tile 1,769,472 B. Load avg at run start: 0.69, 3.56, 3.60.

| B | config | layer ms, median of round medians [round min–max], runs 1 / 2 / 3 | token-layers/s (run 1) |
|---|---|---|---|
| 64 | u16dmm_gd2 | 20.56 [19.70–23.27], 22.18 [20.80–22.66], 18.84 [18.03–19.34] | 3113 |
| 64 | u16dmm128_gd2 | 22.62 [21.14–26.48], 22.79 [22.22–23.90], 19.79 [18.82–20.18] | 2829 |
| 64 | u16dmk_gd2 | 20.78 [19.29–21.75], 21.40 [20.73–22.51], 18.58 [18.50–18.76] | 3080 |
| 64 | u16gfk_st | 24.61 [23.38–29.35], 27.12 [26.09–28.26], 21.59 [20.38–23.28] | 2600 |
| 64 | i8dmm_gd2 | 16.45 [15.59–18.60], 17.88 [16.45–19.24], 15.53 [15.04–15.86] | 3890 |
| 64 | i8dmm128_gd2 | 16.91 [15.69–18.04], 17.71 [14.78–18.40], 14.97 [14.60–15.46] | 3786 |
| 64 | i8dmk_gd2 | 17.07 [15.65–18.94], 17.84 [17.02–18.64], 15.20 [15.00–15.37] | 3749 |
| 128 | u16dmm_gd2 | 29.78 [28.42–34.04], 27.56 [25.41–30.24], 26.95 [26.20–28.24] | 4298 |
| 128 | u16dmm128_gd2 | 30.71 [28.36–32.10], 27.49 [26.52–31.39], 27.88 [27.08–28.47] | 4169 |
| 128 | u16dmk_gd2 | 29.51 [27.54–32.72], 26.84 [26.26–37.77], 26.66 [26.28–28.35] | 4338 |
| 128 | u16gfk_st | 41.92 [39.97–49.42], 38.95 [37.56–51.57], 36.96 [36.25–42.05] | 3053 |
| 128 | i8dmm_gd2 | 26.65 [24.85–28.83], 23.41 [22.11–27.36], 23.61 [22.79–25.39] | 4802 |
| 128 | i8dmm128_gd2 | 25.70 [24.18–28.67], 23.47 [22.66–25.94], 23.53 [22.76–25.51] | 4980 |
| 128 | i8dmk_gd2 | 26.29 [24.37–28.22], 23.25 [22.74–26.34], 23.64 [22.83–24.30] | 4869 |
| 256 | u16dmm_gd2 | 48.57 [46.78–49.81], 43.92 [42.83–51.86], 42.75 [41.16–48.26] | 5271 |
| 256 | u16dmm128_gd2 | 49.58 [46.72–50.97], 43.79 [42.74–46.91], 44.35 [42.81–45.39] | 5163 |
| 256 | u16dmk_gd2 | 47.61 [44.96–50.87], 43.47 [41.60–44.39], 42.95 [42.58–46.18] | 5377 |
| 256 | u16gfk_st | 77.18 [71.82–81.34], 69.15 [66.45–97.88], 72.71 [66.15–90.39] | 3317 |
| 256 | i8dmm_gd2 | 42.96 [41.09–44.58], 39.90 [39.28–40.85], 40.00 [38.85–44.98] | 5959 |
| 256 | i8dmm128_gd2 | 44.47 [43.36–55.30], 39.97 [39.11–44.90], 40.28 [38.92–43.48] | 5757 |
| 256 | i8dmk_gd2 | 45.03 [42.01–46.37], 42.15 [38.79–46.06], 40.27 [38.81–42.62] | 5685 |

| B | ratio (a/b, >1 = b faster) | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 64 | u16dmm/i8dmm_gd2 | 1.250x [1.105–1.427] (u16dmm_gd2/i8dmm_gd2) | 1.241x [1.160–1.318] (u16dmm_gd2/i8dmm_gd2) | 1.213x [1.158–1.258] (u16dmm_gd2/i8dmm_gd2) |
| 64 | u16dmm128/i8dmm128_gd2 | 1.338x [1.177–1.518] (u16dmm128_gd2/i8dmm128_gd2) | 1.287x [1.239–1.567] (u16dmm128_gd2/i8dmm128_gd2) | 1.322x [1.249–1.380] (u16dmm128_gd2/i8dmm128_gd2) |
| 64 | u16dmk/i8dmk_gd2 | 1.217x [1.124–1.294] (u16dmk_gd2/i8dmk_gd2) | 1.199x [1.162–1.304] (u16dmk_gd2/i8dmk_gd2) | 1.222x [1.204–1.240] (u16dmk_gd2/i8dmk_gd2) |
| 64 | u16dmm128/i8dmm_gd2 | 1.375x [1.142–1.623] (u16dmm128_gd2/i8dmm_gd2) | 1.275x [1.242–1.409] (u16dmm128_gd2/i8dmm_gd2) | 1.274x [1.203–1.322] (u16dmm128_gd2/i8dmm_gd2) |
| 64 | u16dmk/i8dmm_gd2 | 1.263x [1.117–1.334] (u16dmk_gd2/i8dmm_gd2) | 1.197x [1.170–1.349] (u16dmk_gd2/i8dmm_gd2) | 1.196x [1.167–1.233] (u16dmk_gd2/i8dmm_gd2) |
| 64 | best_u16dmm/i8dmm_gd2 | 1.250x [1.105–1.427] (u16dmm_gd2/i8dmm_gd2) | 1.197x [1.170–1.349] (u16dmk_gd2/i8dmm_gd2) | 1.196x [1.167–1.233] (u16dmk_gd2/i8dmm_gd2) |
| 64 | best_u16dmm/best_i8 | 1.250x [1.105–1.427] (u16dmm_gd2/i8dmm_gd2) | 1.208x [1.180–1.501] (u16dmk_gd2/i8dmm128_gd2) | 1.241x [1.203–1.267] (u16dmk_gd2/i8dmm128_gd2) |
| 64 | best_u16/best_i8 | 1.250x [1.105–1.427] (u16dmm_gd2/i8dmm_gd2) | 1.208x [1.180–1.501] (u16dmk_gd2/i8dmm128_gd2) | 1.241x [1.203–1.267] (u16dmk_gd2/i8dmm128_gd2) |
| 64 | u16gfk_st/i8dmm_gd2 | 1.496x [1.348–1.784] (u16gfk_st/i8dmm_gd2) | 1.517x [1.390–1.649] (u16gfk_st/i8dmm_gd2) | 1.390x [1.341–1.467] (u16gfk_st/i8dmm_gd2) |
| 64 | u16gfk_st/u16dmm_gd2 | 1.197x [1.040–1.344] (u16gfk_st/u16dmm_gd2) | 1.223x [1.174–1.320] (u16gfk_st/u16dmm_gd2) | 1.146x [1.076–1.235] (u16gfk_st/u16dmm_gd2) |
| 128 | u16dmm/i8dmm_gd2 | 1.117x [1.048–1.232] (u16dmm_gd2/i8dmm_gd2) | 1.178x [1.076–1.229] (u16dmm_gd2/i8dmm_gd2) | 1.141x [1.036–1.196] (u16dmm_gd2/i8dmm_gd2) |
| 128 | u16dmm128/i8dmm128_gd2 | 1.195x [1.077–1.252] (u16dmm128_gd2/i8dmm128_gd2) | 1.171x [1.147–1.230] (u16dmm128_gd2/i8dmm128_gd2) | 1.185x [1.093–1.233] (u16dmm128_gd2/i8dmm128_gd2) |
| 128 | u16dmk/i8dmk_gd2 | 1.123x [1.054–1.192] (u16dmk_gd2/i8dmk_gd2) | 1.154x [1.130–1.625] (u16dmk_gd2/i8dmk_gd2) | 1.128x [1.081–1.199] (u16dmk_gd2/i8dmk_gd2) |
| 128 | u16dmm128/i8dmm_gd2 | 1.152x [0.984–1.236] (u16dmm128_gd2/i8dmm_gd2) | 1.175x [1.085–1.217] (u16dmm128_gd2/i8dmm_gd2) | 1.181x [1.098–1.231] (u16dmm128_gd2/i8dmm_gd2) |
| 128 | u16dmk/i8dmm_gd2 | 1.107x [0.955–1.216] (u16dmk_gd2/i8dmm_gd2) | 1.147x [1.145–1.546] (u16dmk_gd2/i8dmm_gd2) | 1.129x [1.043–1.212] (u16dmk_gd2/i8dmm_gd2) |
| 128 | best_u16dmm/i8dmm_gd2 | 1.107x [0.955–1.216] (u16dmk_gd2/i8dmm_gd2) | 1.147x [1.145–1.546] (u16dmk_gd2/i8dmm_gd2) | 1.129x [1.043–1.212] (u16dmk_gd2/i8dmm_gd2) |
| 128 | best_u16dmm/best_i8 | 1.148x [1.037–1.177] (u16dmk_gd2/i8dmm128_gd2) | 1.154x [1.130–1.625] (u16dmk_gd2/i8dmk_gd2) | 1.133x [1.038–1.188] (u16dmk_gd2/i8dmm128_gd2) |
| 128 | best_u16/best_i8 | 1.148x [1.037–1.177] (u16dmk_gd2/i8dmm128_gd2) | 1.154x [1.130–1.625] (u16dmk_gd2/i8dmk_gd2) | 1.133x [1.038–1.188] (u16dmk_gd2/i8dmm128_gd2) |
| 128 | u16gfk_st/i8dmm_gd2 | 1.573x [1.471–1.714] (u16gfk_st/i8dmm_gd2) | 1.664x [1.631–1.891] (u16gfk_st/i8dmm_gd2) | 1.566x [1.506–1.656] (u16gfk_st/i8dmm_gd2) |
| 128 | u16gfk_st/u16dmm_gd2 | 1.408x [1.329–1.524] (u16gfk_st/u16dmm_gd2) | 1.413x [1.346–1.758] (u16gfk_st/u16dmm_gd2) | 1.372x [1.284–1.599] (u16gfk_st/u16dmm_gd2) |
| 256 | u16dmm/i8dmm_gd2 | 1.131x [1.086–1.182] (u16dmm_gd2/i8dmm_gd2) | 1.101x [1.065–1.320] (u16dmm_gd2/i8dmm_gd2) | 1.069x [1.021–1.119] (u16dmm_gd2/i8dmm_gd2) |
| 256 | u16dmm128/i8dmm128_gd2 | 1.115x [0.910–1.157] (u16dmm128_gd2/i8dmm128_gd2) | 1.095x [0.968–1.173] (u16dmm128_gd2/i8dmm128_gd2) | 1.101x [1.041–1.128] (u16dmm128_gd2/i8dmm128_gd2) |
| 256 | u16dmk/i8dmk_gd2 | 1.057x [0.981–1.130] (u16dmk_gd2/i8dmk_gd2) | 1.031x [0.903–1.121] (u16dmk_gd2/i8dmk_gd2) | 1.067x [1.016–1.150] (u16dmk_gd2/i8dmk_gd2) |
| 256 | u16dmm128/i8dmm_gd2 | 1.154x [1.094–1.224] (u16dmm128_gd2/i8dmm_gd2) | 1.097x [1.072–1.193] (u16dmm128_gd2/i8dmm_gd2) | 1.109x [1.006–1.168] (u16dmm128_gd2/i8dmm_gd2) |
| 256 | u16dmk/i8dmm_gd2 | 1.108x [1.064–1.200] (u16dmk_gd2/i8dmm_gd2) | 1.089x [1.052–1.117] (u16dmk_gd2/i8dmm_gd2) | 1.074x [1.005–1.155] (u16dmk_gd2/i8dmm_gd2) |
| 256 | best_u16dmm/i8dmm_gd2 | 1.108x [1.064–1.200] (u16dmk_gd2/i8dmm_gd2) | 1.089x [1.052–1.117] (u16dmk_gd2/i8dmm_gd2) | 1.069x [1.021–1.119] (u16dmm_gd2/i8dmm_gd2) |
| 256 | best_u16dmm/best_i8 | 1.108x [1.064–1.200] (u16dmk_gd2/i8dmm_gd2) | 1.089x [1.052–1.117] (u16dmk_gd2/i8dmm_gd2) | 1.069x [1.021–1.119] (u16dmm_gd2/i8dmm_gd2) |
| 256 | best_u16/best_i8 | 1.108x [1.064–1.200] (u16dmk_gd2/i8dmm_gd2) | 1.089x [1.052–1.117] (u16dmk_gd2/i8dmm_gd2) | 1.069x [1.021–1.119] (u16dmm_gd2/i8dmm_gd2) |
| 256 | u16gfk_st/i8dmm_gd2 | 1.797x [1.736–1.848] (u16gfk_st/i8dmm_gd2) | 1.733x [1.652–2.396] (u16gfk_st/i8dmm_gd2) | 1.818x [1.703–2.010] (u16gfk_st/i8dmm_gd2) |
| 256 | u16gfk_st/u16dmm_gd2 | 1.589x [1.479–1.633] (u16gfk_st/u16dmm_gd2) | 1.575x [1.311–2.124] (u16gfk_st/u16dmm_gd2) | 1.701x [1.522–1.873] (u16gfk_st/u16dmm_gd2) |

| B | config | take | cast | mm / dot |
|---|---|---|---|---|
| 64 | u16dmm_gd2 | 41.1/23.1, 42.5/18.6, 31.2/18.1 | 19.2/16.0, 20.0/12.7, 14.4/12.0 | 91.0/86.3, 74.5/64.5, 68.1/64.3 |
| 64 | u16dmm128_gd2 | 46.2/21.5, 51.9/19.5, 39.6/18.2 | 18.8/12.8, 21.5/12.1, 16.0/11.3 | 90.5/81.8, 81.7/67.8, 74.0/67.0 |
| 64 | u16dmk_gd2 | 35.0/20.3, 41.8/17.9, 30.8/17.2 | 17.7/13.2, 21.0/12.3, 16.2/12.1 | 87.4/79.2, 79.5/65.4, 71.8/65.4 |
| 64 | u16gfk_st | 31.8/25.6, 27.5/17.5, 24.3/18.3 | 18.2/15.3, 18.4/11.5, 15.1/11.8 | 71.0/68.8, 62.1/54.9, 58.3/56.2 |
| 64 | i8dmm_gd2 | –, –, – | 22.8/19.1, 22.5/16.0, 20.7/15.9 | 87.6/84.7, 72.6/63.5, 68.4/64.7 |
| 64 | i8dmm128_gd2 | –, –, – | 21.0/15.7, 21.1/13.2, 16.3/11.6 | 82.6/77.4, 75.8/66.3, 71.5/66.3 |
| 64 | i8dmk_gd2 | –, –, – | 21.4/15.9, 20.3/12.4, 17.6/12.3 | 85.6/77.0, 77.0/64.6, 72.9/65.4 |
| 128 | u16dmm_gd2 | 40.8/20.8, 31.8/17.6, 31.2/17.8 | 19.0/14.2, 15.2/11.9, 16.1/12.3 | 129.1/119.7, 112.7/108.0, 111.6/108.1 |
| 128 | u16dmm128_gd2 | 51.7/22.0, 37.6/17.7, 36.1/17.6 | 20.9/13.3, 15.2/11.3, 14.4/11.2 | 147.5/136.6, 118.9/113.1, 118.8/113.7 |
| 128 | u16dmk_gd2 | 39.8/20.3, 31.4/17.0, 32.2/17.3 | 19.6/13.4, 15.1/11.9, 15.3/12.2 | 137.6/128.0, 115.0/109.5, 116.6/110.3 |
| 128 | u16gfk_st | 30.4/22.5, 24.5/17.7, 24.6/18.6 | 18.5/13.7, 15.1/11.7, 15.2/12.0 | 120.5/113.9, 99.3/97.4, 101.3/98.5 |
| 128 | i8dmm_gd2 | –, –, – | 24.3/18.3, 20.6/16.1, 19.2/15.7 | 135.9/124.0, 114.4/111.2, 111.7/107.2 |
| 128 | i8dmm128_gd2 | –, –, – | 24.5/15.5, 17.8/12.5, 18.1/12.4 | 137.0/127.0, 118.1/112.8, 118.3/112.2 |
| 128 | i8dmk_gd2 | –, –, – | 25.1/16.1, 18.7/12.2, 18.5/12.4 | 139.3/129.6, 112.3/107.7, 115.3/110.8 |
| 256 | u16dmm_gd2 | 41.4/20.2, 34.4/18.5, 33.4/18.4 | 18.7/14.1, 16.5/12.8, 15.0/12.9 | 203.6/195.3, 194.8/190.4, 194.0/189.7 |
| 256 | u16dmm128_gd2 | 44.6/20.5, 40.5/18.9, 36.5/18.2 | 19.6/12.8, 15.7/11.8, 15.1/11.4 | 218.6/205.2, 206.1/200.3, 205.0/200.3 |
| 256 | u16dmk_gd2 | 40.6/20.5, 34.4/18.0, 34.7/19.4 | 19.8/13.1, 15.3/12.3, 16.2/13.2 | 209.1/199.2, 196.7/192.7, 224.0/219.9 |
| 256 | u16gfk_st | 34.0/26.6, 32.1/22.0, 34.0/26.5 | 24.0/14.7, 17.0/13.1, 18.8/14.3 | 206.7/198.2, 187.7/184.5, 223.5/214.6 |
| 256 | i8dmm_gd2 | –, –, – | 22.8/17.4, 21.3/15.9, 22.5/17.5 | 205.0/195.6, 194.3/191.0, 219.0/213.5 |
| 256 | i8dmm128_gd2 | –, –, – | 26.0/16.2, 21.3/14.3, 21.7/15.4 | 223.6/208.7, 205.0/198.3, 225.6/217.0 |
| 256 | i8dmk_gd2 | –, –, – | 26.0/17.5, 20.8/13.6, 25.1/16.8 | 221.5/208.3, 196.7/191.1, 226.2/217.1 |

| B | config | max abs err vs ref, runs 1 / 2 / 3 | cos (min) | instr identical |
|---|---|---|---|---|
| 64 | u16dmm_gd2 | 1.24e-06, 1.44e-06, 1.24e-06 | 1.000000000000 | True |
| 64 | u16dmm128_gd2 | 1.24e-06, 1.44e-06, 1.24e-06 | 1.000000000000 | True |
| 64 | u16dmk_gd2 | 1.23e-06, 1.44e-06, 1.23e-06 | 1.000000000000 | True |
| 64 | u16gfk_st | 1.23e-06, 1.44e-06, 1.23e-06 | 1.000000000000 | True |
| 64 | i8dmm_gd2 | 4.96e-06, 5.03e-06, 4.88e-06 | 1.000000000000 | True |
| 64 | i8dmm128_gd2 | 4.96e-06, 5.03e-06, 4.88e-06 | 1.000000000000 | True |
| 64 | i8dmk_gd2 | 5.32e-06, 4.79e-06, 5.18e-06 | 1.000000000000 | True |
| 128 | u16dmm_gd2 | 1.28e-06, 1.44e-06, 1.29e-06 | 1.000000000000 | True |
| 128 | u16dmm128_gd2 | 1.28e-06, 1.44e-06, 1.29e-06 | 1.000000000000 | True |
| 128 | u16dmk_gd2 | 1.23e-06, 1.44e-06, 1.25e-06 | 1.000000000000 | True |
| 128 | u16gfk_st | 1.23e-06, 1.44e-06, 1.25e-06 | 1.000000000000 | True |
| 128 | i8dmm_gd2 | 4.96e-06, 5.19e-06, 4.88e-06 | 1.000000000000 | True |
| 128 | i8dmm128_gd2 | 4.96e-06, 5.19e-06, 4.88e-06 | 1.000000000000 | True |
| 128 | i8dmk_gd2 | 5.32e-06, 5.17e-06, 5.18e-06 | 1.000000000000 | True |
| 256 | u16dmm_gd2 | 1.36e-06, 1.45e-06, 1.29e-06 | 1.000000000000 | True |
| 256 | u16dmm128_gd2 | 1.36e-06, 1.45e-06, 1.29e-06 | 1.000000000000 | True |
| 256 | u16dmk_gd2 | 1.36e-06, 1.44e-06, 1.37e-06 | 1.000000000000 | True |
| 256 | u16gfk_st | 1.36e-06, 1.44e-06, 1.37e-06 | 1.000000000000 | True |
| 256 | i8dmm_gd2 | 4.96e-06, 5.19e-06, 4.97e-06 | 1.000000000000 | True |
| 256 | i8dmm128_gd2 | 4.96e-06, 5.19e-06, 4.97e-06 | 1.000000000000 | True |
| 256 | i8dmk_gd2 | 5.32e-06, 5.17e-06, 5.18e-06 | 1.000000000000 | True |

| B | config | take share of worker cpu-ms (take / (take + cast + dot)), runs 1 / 2 / 3 |
|---|---|---|
| 64 | u16dmm_gd2 | 18%, 19%, 19% |
| 64 | u16dmm128_gd2 | 19%, 20%, 19% |
| 64 | u16dmk_gd2 | 18%, 19%, 18% |
| 64 | u16gfk_st | 23%, 21%, 21% |
| 128 | u16dmm_gd2 | 13%, 13%, 13% |
| 128 | u16dmm128_gd2 | 13%, 12%, 12% |
| 128 | u16dmk_gd2 | 13%, 12%, 12% |
| 128 | u16gfk_st | 15%, 14%, 14% |
| 256 | u16dmm_gd2 | 9%, 8%, 8% |
| 256 | u16dmm128_gd2 | 9%, 8%, 8% |
| 256 | u16dmk_gd2 | 9%, 8%, 8% |
| 256 | u16gfk_st | 11%, 10%, 10% |

## Verdict

**not-viable.** The orientation was most of the 05:55 gap. Putting the u16 take in front of the i8dmm kernel (same gd2 chunk list, same x @ tile.T matmul) makes u16dmm_gd2 1.15–1.22x faster than the 05:55 best u16 (u16gfk_st) at B 64, 1.37–1.41x at B 128 and 1.58–1.70x at B 256. Weight-major u16gfk_st falls further behind i8dmm as B grows (1.39–1.52x at B 64, 1.57–1.66x at B 128, 1.73–1.82x at B 256), which reproduces the 05:55 gap. Against i8dmm_gd2, u16dmm_gd2 is 1.21–1.25x at B 64, 1.12–1.18x at B 128 and 1.07–1.13x at B 256 (per-round 1.02–1.43x). The best u16dmm variant (u16dmk_gd2 in most runs, down K-blocked) is 1.20–1.25x, 1.11–1.15x and 1.07–1.11x. So the <= 1.15x bar is met at B 256 in every run, and at B 128 in 2 of 3 runs for u16dmm (3 of 3 for the best variant), but missed at B 64. Against the fastest int8 config per run (i8dmm / i8dmm128 / i8dmk), best u16 is 1.21–1.25x, 1.13–1.15x and 1.07–1.11x. The per-chunk take did not push the ratio back above 1.3x. The profile says why: the u16dmm dot stage (64–120 cpu-ms at B 64/128, 190–220 at B 256) matches i8dmm's dot and the cast stages are about the same (11–16 vs 12–19 cpu-ms). The remaining gap is the take, a flat ~17–23 cpu-ms per layer that does not grow with B, so its share of u16 worker cpu falls from 18–19% at B 64 to 12–13% at B 128 and 8–9% at B 256, and the ratio falls with it. 128-row chunks (u16dmm128 vs i8dmm128) do not help: 1.29–1.34x at B 64, 1.17–1.20x at B 128, 1.10–1.12x at B 256. Ternary is still slower than int8 at every B tested, so there is no speed win, and the memory win is the known 17.46 vs 69.56 MB of weights (3.98x). Runs 2 and 3 started at load avg 3.6 (other agents on the box), but the ratios are within-run and agree across runs. Error: all u16 variants <= 1.45e-6 vs the float64 ternary reference (cos 1.0), trits exact (pack round-trip asserted); int8 variants <= 5.32e-6 vs the int8 reference; instrumented and plain outputs identical everywhere.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-28 10:55 wave reached this lane, and no Chief-named item was found (pairs idle), so this fire took the chained question from the 05:55 note. This note only adds one new file under junior_osai/vault/findings/; HEAD at clone and the last-6-hours check are in the push commit message.

## Next question

With the orientation fixed, the u16 / int8 prefill gap is a flat per-layer take cost (~17–23 cpu-ms) that does not grow with B. (The f32-LUT take is already ruled out: 11-27 05:50 found it write-bound and slower.) At B 512 and B 1024 on the same layer, with a whole prompt in one weight pass, does u16dmm_gd2 / i8dmm_gd2 fall to <= 1.05x (take share ~2–4%)? And does the 3.98x smaller weight stream then show up as a speed edge, i.e. a ratio < 1.0? Or does the dot stage stay compute-bound, so that u16 converges to int8 from above and never passes it?

## Script (v16q.py)

```python
#!/usr/bin/env python3
"""V16Q: u16dmm (8-trit uint16 LUT take per gd2 chunk -> int8 tile -> f32 -> x @ tile.T, the i8dmm orientation) vs int8 i8dmm_gd2,
prefill chunks B 64 / 128 / 256, W 8 pool, full bitnetd layer, ctx-free, 10:5x MT 11/28. Base: bench0555_1128/v16p.py (copied below, kernels unchanged).
New row-major kernels (x (B, in), y (B, out)), run under the gd2 guided chunk queue, one np.take(LUT16U) per chunk:
  u16dmm_gd2    : chunk list identical to i8dmm_gd2 (kind "8d", T 128 -> 256-row chunks, then 64-row tail chunks); take chunk, cast int8 -> f32 (n, in),
                  np.matmul(x, tile.T) -> tmp (B, n) -> y[:, rows]  (exactly mk_8 / i8dmm with the take in front of the cast)
  u16dmm128_gd2 : same, chunk-size variant: 128-row chunks then 64-row tail at every site (kind "x128"); control i8dmm128_gd2 (i8dmm on the same list)
  u16dmk_gd2    : u16dmm_gd2 with the down site K-blocked at Kc 1728 (control i8dmk_gd2 = int8 one-matmul with the same K-blocking)
Link to 05:55: u16gfk_st (05:55 best u16 at B 128, weight-major fused take). XREFP has max(BS) = 256 rows.
Earlier V16G doc:
Base: /workspace/bench1855_1127/v16x.py exec'd without main() (-> v16m -> v16k -> v16t -> v16d -> v16.py: same weights, P16 8-trit uint16 pack
with exact trit round-trip asserts, float64 refs, st / dy / gd2 schedules, u16ad / i8dmm kernels, profiling hooks).
New weight-major kernels (xT (in, B), yT (out, B)), static split, one np.take(LUT16U) per worker range:
  u16gft{R}_st : cast R-row int8 tile -> f32 (R, in) -> np.matmul(tile, xT, out=yT[rows]); R 256 / 128
  u16gfk_st    : same, R 256, down site K-blocked at Kc 1728 (f32 sub-tile 1,769,472 B), tmp (m, B) accumulate
Controls: u16ad_st (row-major B-adaptive, 18:55 best u16), i8g_gd2 (weight-major int8 cast + tile @ xT, T 128), i8dmm_gd2 (int8 x @ tile.T, T 128),
          i8g_st (weight-major int8, static split, T 256).
Usage (V16G): OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16g.py out.json
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16q.py out.json"""
import json, os, sys
import numpy as np
_S = open("/workspace/bench1855_1127/v16x.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "v16x.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 9
BS = (64, 128, 256); TMPN = 256 * 256
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
def mk_ud(KC):
    def r(PR, x, y, r0, r1, T, w, i):
        q = PR.shape[1]; n = r1 - r0; P = PRF[0]; B = x.shape[0]; kc = KC if (KC and i > D) else i; assert n <= 256 and q * 8 == i
        su = scr(("udm", w), (256 * Q8,), np.uint64)[:n * q].reshape(n, q); s8 = su.view(np.int8).reshape(n, q * 8)
        sb = scr(("udf", w), (256 * F,), np.float32); tmp = scr(("udt", w), (TMPN,), np.float32); yv = y[:, r0:r1]
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16U, PR[r0:r1], out=su)
        if P: _add(w, "take", pc() - t0, tt() - c0)
        for j, k0 in enumerate(range(0, i, kc)):
            k1 = min(i, k0 + kc); s = sb[:n * (k1 - k0)].reshape(n, k1 - k0)
            if P: t0 = pc(); c0 = tt()
            np.copyto(s, s8[:, k0:k1], casting="unsafe")
            if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
            t = tmp[:B * n].reshape(B, n); np.matmul(x[:, k0:k1], s.T, out=t)
            if j == 0: yv[...] = t
            else: np.add(yv, t, out=yv)
            if P: _add(w, "dot", pc() - t0, tt() - c0)
    return r
KU = {"u16dmm": (mk_ud(0), "8d"), "u16dmm128": (mk_ud(0), "x128"), "u16dmk": (mk_ud(1728), "8d")}
K8X = {"8dmm128": (K8["8dmm"], "x128")}
_mm_q = mm
def mm(k, a, kind, T, mode, rec):
    if kind not in KU and kind not in K8X: return _mm_q(k, a, kind, T, mode, rec)
    CUR[0] = k; B = a.shape[0]
    if kind in KU: P = P16[k]; R = P["PR"]; i = P["inp"]; fn, ck = KU[kind]
    else: P = par.F8[k]; R = P["Q"]; i = R.shape[1]; fn, ck = K8X[kind]
    out = R.shape[0]; x = np.ascontiguousarray(a, np.float32); y = np.empty((B, out), np.float32)
    run_site(fn, R, x, y, out, T, i, mode, rec, k, ck)
    y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
C = {"u16dmm_gd2": ("u16dmm", 128, "gd2"), "u16dmm128_gd2": ("u16dmm128", 128, "gd2"), "u16dmk_gd2": ("u16dmk", 128, "gd2"), "u16gfk_st": ("u16gfk", 8, "st"),
     "i8dmm_gd2": ("8dmm", 128, "gd2"), "i8dmm128_gd2": ("8dmm128", 128, "gd2"), "i8dmk_gd2": ("8dmk", 128, "gd2")}
UD = ["u16dmm_gd2", "u16dmm128_gd2", "u16dmk_gd2"]; U = UD + ["u16gfk_st"]; UG = U; I8 = ["i8dmm_gd2", "i8dmm128_gd2", "i8dmk_gd2"]
def wq64(k, bits):
    if bits == 2: P = P16[k]; o, q = P["PR"].shape; return L16[P["PR"]].reshape(o, q * 8)[:, :P["inp"]].astype(np.float64) * P["g"].astype(np.float64)[:, None]
    P = par.F8[k]; return P["Q"].astype(np.float64) * P["g"].astype(np.float64)[:, None]
def layer_refq(x, bits):
    x = x.astype(np.float64); r = lambda a: a / np.sqrt(np.mean(a ** 2, axis=-1, keepdims=True) + 1e-6); sg = lambda a: a / (1 + np.exp(-a))
    mm_ = lambda a, k: a @ wq64(k, bits).T
    qkv = mm_(r(x), "qkv"); v = qkv[:, D + KVH * HD:]
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    x2 = x + mm_(att, "o"); gu = mm_(r(x2), "gu"); z = sg(gu[:, :F]) * gu[:, F:]
    return x2 + mm_(z, "down")
for k in SITES:   # trit pack round-trip re-asserted here (P16 decode == trits of the t5 pack)
    P = P16[k]; o, q = P["PR"].shape; assert np.array_equal(L16[P["PR"]].reshape(o, q * 8)[:, :P["inp"]], L5[P5[k]["PR"]].reshape(o, -1)[:, :P["inp"]]), k
REFCHK = {b: float(np.abs(layer_refq(XREF, b) - REF[b]).max()) for b in (2, 8)}; assert REFCHK[2] <= 1e-12 and REFCHK[8] <= 1e-6, REFCHK   # int8: F8 g is float32 vs base m8 (~1e-7)
XREF = np.random.default_rng(int(os.environ.get("XSEED", "1")) + 2000).standard_normal((max(BS), D)).astype(np.float32)
REF = {b: layer_refq(XREF, b) for b in (2, 8)}
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, RMAX=RMAX, load=os.getloadavg(), time={}, prof={}, err={}, refchk=REFCHK)
    W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES); W16 = sum(P16[k]["PR"].nbytes + P16[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(u16=W16, int8=W8, int8_over_u16=W8 / W16, lut_u64=LUT16U.nbytes, subtile_f32=256 * 1728 * 4, refchk=REFCHK)
    print("bytes", res["bytes"], "load", res["load"], flush=True)
    names = list(C)
    for B in BS:
        xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0].startswith("8") else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in U: e[c]["vs_u16gfk_mab"] = float(np.abs(outs[c] - outs["u16gfk_st"]).max())
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
        bd = min(UD, key=lambda c: o[c]["med"]); bu = min(U, key=lambda c: o[c]["med"]); bi = min(I8, key=lambda c: o[c]["med"])
        pr_("u16dmm_gd2", "i8dmm_gd2", "u16dmm/i8dmm_gd2"); pr_("u16dmm128_gd2", "i8dmm128_gd2", "u16dmm128/i8dmm128_gd2"); pr_("u16dmk_gd2", "i8dmk_gd2", "u16dmk/i8dmk_gd2")
        pr_("u16dmm128_gd2", "i8dmm_gd2", "u16dmm128/i8dmm_gd2"); pr_("u16dmk_gd2", "i8dmm_gd2", "u16dmk/i8dmm_gd2")
        pr_(bd, "i8dmm_gd2", "best_u16dmm/i8dmm_gd2"); pr_(bd, bi, "best_u16dmm/best_i8"); pr_(bu, bi, "best_u16/best_i8")
        pr_("u16gfk_st", "i8dmm_gd2", "u16gfk_st/i8dmm_gd2"); pr_("u16gfk_st", "u16dmm_gd2", "u16gfk_st/u16dmm_gd2")
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

close: e1105c26b0ca5c4d
