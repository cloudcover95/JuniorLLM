---
local: true
date: 2026-11-28
wave: "2026-11-28 05:55 slot (fired 05:47 MT, ran ~05:48–05:58 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (prefill-sized chunks B 64 / B 128, W 8 pool, full bitnetd layer, ctx-free, weight-major u16gft256 / u16gft128 / u16gfk vs int8 i8g_gd2 / i8g_st / i8dmm_gd2. Against the weight-major int8 kernels the u16 gap does shrink: best u16 / i8g_gd2 1.19–1.30x at B 64 and 1.15–1.17x at B 128; best u16 / i8g_st 1.07–1.18x at B 64 and 1.04–1.06x at B 128 (the <= 1.1x bar is met only against i8g_st, and only at B 128). But at prefill sizes the prefill-orientation int8 kernel i8dmm_gd2 (x @ tile.T) is the fastest config by a wide margin, and the gap to it grows with B: 1.43–1.58x at B 64, 1.58–1.64x at B 128. So take + cast is not the real tax at prefill; u16 lacks a fast x @ tile.T orientation. Error <= 1.44e-6 vs the float64 ternary reference (cos 1.0), trits exact; int8 <= 5.19e-6 vs the int8 reference. Memory unchanged: 17.46 vs 69.56 MB (3.98x, already known). No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, prefill, chunked-prefill, thread-pool, lut-expand, uint16, eight-trit-pack, base-3, 2-bit, np-take, fused-take, static-split, weight-major, k-blocking, np-matmul, int8-control, int8, numpy, not-viable]
---

# Prefill chunks B 64 / 128: u16gft gets within 1.04–1.17x of weight-major int8, but int8's x @ tile.T kernel (i8dmm) pulls away to 1.43–1.64x

The 02:55 note ([[2026-11-28_0255_weight-major-u16gft-fused-take-tile-at-xt-yt-vs-i8g-i8dmm-w8-pool-b8-b16-b32-gap-int8-1p34-1p36x-b8-1p29-1p33x-b16-1p21-1p28x-b32-u16gft-1p07-1p17x-vs-u16ad]]) found that weight-major u16gft (one fused LUT take per worker range, cast per tile, tile @ xT → yT) is the fastest u16 kernel at B >= 8, and that the gap to the best int8 kernel shrank from 1.35x at B 8 to 1.21–1.28x at B 32. The take stage was a flat ~21 worker-ms per layer while mm grew with B. It asked: at prefill-sized chunks, B 64 and B 128 (still ctx-free, same layer), where mm should dominate, does the u16 / int8 ratio fall to <= 1.1x, or does take + cast stay a >= 1.15x tax?

## Question

On the W 8 pool layer (full bitnetd layer, ctx-free), at B 64 and B 128: does u16gft128 / u16gfk come within 1.1x of the int8 kernels (i8g_gd2, i8g_st, and the prefill-orientation i8dmm_gd2), making ternary near speed-parity for chunked prefill with its 3.98x weight-memory win?

## Method

- Box: 8 CPUs (Xeon, L2 2 MiB per core), numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents (load avg at run start in the bytes line below), so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench0555_1128/v16p.py is bench0255_1128/v16g.py with B (64, 128), the u16gft tmp scratch raised to 128 × 256 floats, and u16ad_st dropped (row-major decode kernel, already 1.07–1.17x behind u16gft). It execs /workspace/bench1855_1127/v16x.py without main() (→ v16m → v16k → v16t → v16d → v16.py: same weights, P16 8-trit uint16 pack with exact trit round-trip asserts, st / dy / gd2 schedules, i8dmm kernel, profiling hooks).
- Kernels as at 02:55: **u16gft256_st / u16gft128_st** (fused take per static worker range, 256- or 128-row tiles, cast int8 → f32, tile @ xT → yT), **u16gfk_st** (256-row tiles, down site K-blocked at Kc 1728). Controls: **i8g_gd2** (weight-major int8, T 128), **i8g_st** (same, static split, T 256), **i8dmm_gd2** (int8 cast + x @ tile.T, T 128, the prefill orientation).
- Reference: the original XREF has only 32 rows, so each run draws a 128-row XREFP (rng XSEED + 2000) and builds a new float64 reference from the trits decoded from P16 (int8: F8 Q) times g in float64. Before use, that path is checked against the original wf64 reference on XREF[:32]: ternary differs by 0.0 in all runs, int8 by 0.99–1.13e-7 (F8 g is stored as float32), asserted ternary <= 1e-12 and int8 <= 1e-6. The P16 decode is also asserted to equal the t5-pack trits at every site.
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round. Reported as median of round medians, round min–max, per-round ratios, and token-layers/s (B / layer time). Then 9 instrumented layers per config give stage worker-ms (wall/cpu summed over workers). Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench0555_1128 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16p.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Below: layer ms, ratios (a/b > 1 means b is faster), stage worker-ms (wall/cpu, runs 1, 2, 3; int8 kernels have no take stage) and error. Ratio rows named best_* pick the fastest config of that group within each run (best_u16g = fastest of the three u16gft variants; best_i8 = fastest of i8g_gd2 / i8g_st / i8dmm_gd2, which was i8dmm_gd2 in every run at both B).

Bytes: weights u16 17,458,176 B vs int8 69,559,296 B (3.98x); LUT u64 52,488 B; down f32 sub-tile 1,769,472 B. Load avg at run start: 1.10, 1.73, 2.60.

| B | config | layer ms, median of round medians [round min–max], runs 1 / 2 / 3 | token-layers/s (run 1) |
|---|---|---|---|
| 64 | u16gft256_st | 31.95 [27.95–33.36], 31.20 [29.50–36.95], 27.98 [25.88–31.00] | 2003 |
| 64 | u16gft128_st | 28.85 [27.02–34.18], 28.53 [26.74–30.78], 27.98 [25.10–30.18] | 2219 |
| 64 | u16gfk_st | 31.15 [28.37–35.61], 29.47 [27.10–29.81], 26.86 [23.94–29.29] | 2055 |
| 64 | i8g_gd2 | 23.92 [22.97–26.64], 23.90 [22.97–25.34], 20.66 [19.20–23.06] | 2675 |
| 64 | i8g_st | 25.71 [23.84–29.67], 26.75 [24.80–29.86], 22.87 [20.65–24.94] | 2489 |
| 64 | i8dmm_gd2 | 20.19 [18.94–21.25], 19.86 [18.70–21.64], 16.97 [15.77–19.25] | 3169 |
| 128 | u16gft256_st | 50.51 [47.87–54.78], 47.41 [45.58–53.58], 60.32 [52.23–63.16] | 2534 |
| 128 | u16gft128_st | 50.46 [47.16–58.32], 47.48 [45.26–67.67], 57.78 [54.08–67.32] | 2537 |
| 128 | u16gfk_st | 47.79 [45.08–54.59], 45.57 [43.33–50.76], 54.38 [50.26–57.84] | 2678 |
| 128 | i8g_gd2 | 41.70 [39.61–43.99], 39.59 [36.09–44.69], 46.51 [39.79–50.59] | 3070 |
| 128 | i8g_st | 45.94 [41.25–47.73], 43.80 [41.77–66.23], 51.10 [48.59–54.00] | 2786 |
| 128 | i8dmm_gd2 | 30.21 [27.99–31.92], 27.85 [27.07–30.13], 33.96 [31.39–35.16] | 4237 |

| B | ratio (a/b, >1 = b faster) | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 64 | u16gft128_st/i8g_gd2 | 1.206x [1.083–1.324] (u16gft128_st/i8g_gd2) | 1.194x [1.120–1.271] (u16gft128_st/i8g_gd2) | 1.354x [1.223–1.434] (u16gft128_st/i8g_gd2) |
| 64 | u16gft128_st/i8g_st | 1.122x [0.931–1.276] (u16gft128_st/i8g_st) | 1.066x [1.031–1.150] (u16gft128_st/i8g_st) | 1.224x [1.135–1.241] (u16gft128_st/i8g_st) |
| 64 | u16gfk_st/i8g_gd2 | 1.302x [1.223–1.337] (u16gfk_st/i8g_gd2) | 1.233x [1.139–1.297] (u16gfk_st/i8g_gd2) | 1.300x [1.158–1.387] (u16gfk_st/i8g_gd2) |
| 64 | u16gfk_st/i8g_st | 1.212x [1.073–1.234] (u16gfk_st/i8g_st) | 1.101x [0.998–1.201] (u16gfk_st/i8g_st) | 1.175x [1.077–1.280] (u16gfk_st/i8g_st) |
| 64 | best_u16g/i8g_gd2 | 1.206x [1.083–1.324] (u16gft128_st/i8g_gd2) | 1.194x [1.120–1.271] (u16gft128_st/i8g_gd2) | 1.300x [1.158–1.387] (u16gfk_st/i8g_gd2) |
| 64 | best_u16g/i8g_st | 1.122x [0.931–1.276] (u16gft128_st/i8g_st) | 1.066x [1.031–1.150] (u16gft128_st/i8g_st) | 1.175x [1.077–1.280] (u16gfk_st/i8g_st) |
| 64 | best_u16g/i8dmm_gd2 | 1.429x [1.358–1.692] (u16gft128_st/i8dmm_gd2) | 1.436x [1.385–1.518] (u16gft128_st/i8dmm_gd2) | 1.583x [1.446–1.660] (u16gfk_st/i8dmm_gd2) |
| 64 | best_u16g/best_i8 | 1.429x [1.358–1.692] (u16gft128_st/i8dmm_gd2) | 1.436x [1.385–1.518] (u16gft128_st/i8dmm_gd2) | 1.583x [1.446–1.660] (u16gfk_st/i8dmm_gd2) |
| 128 | u16gft128_st/i8g_gd2 | 1.210x [1.134–1.413] (u16gft128_st/i8g_gd2) | 1.199x [1.116–1.875] (u16gft128_st/i8g_gd2) | 1.242x [1.150–1.452] (u16gft128_st/i8g_gd2) |
| 128 | u16gft128_st/i8g_st | 1.098x [0.989–1.303] (u16gft128_st/i8g_st) | 1.084x [0.734–1.363] (u16gft128_st/i8g_st) | 1.131x [1.019–1.247] (u16gft128_st/i8g_st) |
| 128 | u16gfk_st/i8g_gd2 | 1.146x [1.086–1.309] (u16gfk_st/i8g_gd2) | 1.151x [1.020–1.265] (u16gfk_st/i8g_gd2) | 1.169x [1.075–1.305] (u16gfk_st/i8g_gd2) |
| 128 | u16gfk_st/i8g_st | 1.040x [0.983–1.170] (u16gfk_st/i8g_st) | 1.040x [0.766–1.075] (u16gfk_st/i8g_st) | 1.064x [0.985–1.143] (u16gfk_st/i8g_st) |
| 128 | best_u16g/i8g_gd2 | 1.146x [1.086–1.309] (u16gfk_st/i8g_gd2) | 1.151x [1.020–1.265] (u16gfk_st/i8g_gd2) | 1.169x [1.075–1.305] (u16gfk_st/i8g_gd2) |
| 128 | best_u16g/i8g_st | 1.040x [0.983–1.170] (u16gfk_st/i8g_st) | 1.040x [0.766–1.075] (u16gfk_st/i8g_st) | 1.064x [0.985–1.143] (u16gfk_st/i8g_st) |
| 128 | best_u16g/i8dmm_gd2 | 1.582x [1.492–1.844] (u16gfk_st/i8dmm_gd2) | 1.636x [1.515–1.857] (u16gfk_st/i8dmm_gd2) | 1.601x [1.480–1.824] (u16gfk_st/i8dmm_gd2) |
| 128 | best_u16g/best_i8 | 1.582x [1.492–1.844] (u16gfk_st/i8dmm_gd2) | 1.636x [1.515–1.857] (u16gfk_st/i8dmm_gd2) | 1.601x [1.480–1.824] (u16gfk_st/i8dmm_gd2) |

| B | config | take | cast | mm / dot |
|---|---|---|---|---|
| 64 | u16gft256_st | 40.5/34.8, 34.1/28.8, 34.0/26.5 | 24.3/22.5, 23.1/22.1, 20.3/17.1 | 79.3/78.3, 64.9/63.5, 62.3/58.9 |
| 64 | u16gft128_st | 41.0/34.7, 34.9/29.6, 34.7/27.3 | 24.6/20.1, 21.2/17.6, 22.6/14.6 | 82.4/76.7, 68.0/65.7, 72.5/64.0 |
| 64 | u16gfk_st | 37.9/33.7, 35.4/29.8, 36.4/26.2 | 21.6/18.0, 22.5/18.4, 22.8/16.0 | 70.0/66.3, 66.0/62.3, 70.2/60.4 |
| 64 | i8g_gd2 | –, –, – | 29.4/27.3, 27.7/24.9, 34.2/23.9 | 75.0/73.1, 70.9/67.4, 83.3/69.7 |
| 64 | i8g_st | –, –, – | 24.6/22.6, 23.3/20.9, 30.4/22.5 | 66.7/65.6, 60.3/59.9, 73.0/67.6 |
| 64 | i8dmm_gd2 | –, –, – | 27.5/23.2, 25.3/23.2, 36.3/26.8 | 85.2/77.2, 77.4/72.9, 99.8/81.9 |
| 128 | u16gft256_st | 33.3/27.5, 38.8/32.4, 54.4/39.0 | 20.3/19.0, 22.9/21.1, 30.1/23.8 | 110.4/109.2, 114.0/108.8, 130.1/116.7 |
| 128 | u16gft128_st | 34.7/28.2, 39.2/33.2, 46.2/35.6 | 20.2/16.5, 21.4/17.3, 27.8/18.7 | 123.0/119.9, 119.1/112.8, 132.1/119.4 |
| 128 | u16gfk_st | 34.4/27.6, 41.2/35.7, 44.8/33.7 | 17.3/15.6, 22.7/18.9, 26.7/18.5 | 114.1/111.9, 110.1/106.1, 122.7/110.8 |
| 128 | i8g_gd2 | –, –, – | 28.8/25.8, 29.4/27.2, 43.3/30.1 | 125.4/121.6, 118.3/114.2, 132.7/120.7 |
| 128 | i8g_st | –, –, – | 25.0/22.6, 25.7/22.1, 26.7/21.2 | 106.1/105.4, 105.3/102.7, 110.1/104.5 |
| 128 | i8dmm_gd2 | –, –, – | 23.9/20.8, 26.8/22.8, 34.0/22.1 | 121.8/117.7, 126.5/117.6, 130.4/118.9 |

| B | config | max abs err vs ref, runs 1 / 2 / 3 | cos (min) | instr identical |
|---|---|---|---|---|
| 64 | u16gft256_st | 1.24e-06, 1.44e-06, 1.24e-06 | 1.000000000000 | True |
| 64 | u16gft128_st | 1.24e-06, 1.44e-06, 1.24e-06 | 1.000000000000 | True |
| 64 | u16gfk_st | 1.23e-06, 1.44e-06, 1.23e-06 | 1.000000000000 | True |
| 64 | i8g_gd2 | 4.96e-06, 5.03e-06, 4.88e-06 | 1.000000000000 | True |
| 64 | i8g_st | 4.96e-06, 5.03e-06, 4.88e-06 | 1.000000000000 | True |
| 64 | i8dmm_gd2 | 4.96e-06, 5.03e-06, 4.88e-06 | 1.000000000000 | True |
| 128 | u16gft256_st | 1.28e-06, 1.44e-06, 1.29e-06 | 1.000000000000 | True |
| 128 | u16gft128_st | 1.28e-06, 1.44e-06, 1.29e-06 | 1.000000000000 | True |
| 128 | u16gfk_st | 1.23e-06, 1.44e-06, 1.25e-06 | 1.000000000000 | True |
| 128 | i8g_gd2 | 4.96e-06, 5.19e-06, 4.88e-06 | 1.000000000000 | True |
| 128 | i8g_st | 4.96e-06, 5.19e-06, 4.88e-06 | 1.000000000000 | True |
| 128 | i8dmm_gd2 | 4.96e-06, 5.19e-06, 4.88e-06 | 1.000000000000 | True |

## Verdict

**not-viable.** The 02:55 trend continues against the weight-major int8 kernels. Best u16 / i8g_gd2 is 1.19–1.30x at B 64 and 1.15–1.17x at B 128; best u16 / i8g_st is 1.07–1.18x at B 64 and 1.04–1.06x at B 128 (some rounds go below 1.0). So the <= 1.1x bar is met only against i8g_st, and only at B 128 (and in one of three runs at B 64). u16gfk (down K-blocked) becomes the best u16 variant at B 128. The profile matches that: the u16 take is 33–54 worker-ms (higher than at B 32, but noisy and not tied to B), cast 17–30, mm 110–132, which is the same as i8g's mm (105–133). At B 128 the take is about 20% of u16 worker time. The problem is the int8 control. At prefill sizes the prefill-orientation kernel i8dmm_gd2 (x @ tile.T, T 128, guided schedule) is the fastest config, at 17.0–20.2 ms for B 64 and 27.9–34.0 ms for B 128 vs 26.9–28.9 / 45.6–54.4 ms for the best u16. The gap to it grows with B: 1.43–1.58x at B 64 and 1.58–1.64x at B 128 (per-round 1.36–1.86x). That is worse than the 1.21–1.28x at B 32. Its summed dot worker-ms (77–130) are no lower than i8g's mm, so the win comes from wall time: probably better pool balance under gd2, plus BLAS doing better on the (B, in) @ (in, T) shape. Take + cast is not the binding tax at prefill. u16 has no x @ tile.T variant with a fused take and gd2 scheduling, and that is what it now has to beat. Ternary in numpy stays a memory-only win for chunked prefill: 17.46 vs 69.56 MB of weights (3.98x, already known). Error: all u16 variants <= 1.44e-6 vs the float64 ternary reference (cos 1.0), trits exact (pack round-trip asserted); int8 variants <= 5.19e-6 vs the int8 reference; instrumented and plain outputs identical everywhere.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-28 05:55 wave reached this lane, and no Chief-named item was found, so this fire took the chained question from the 02:55 note. This note only adds one new file under junior_osai/vault/findings/; HEAD at clone and the last-6-hours check are in the push commit message.

## Next question

At prefill sizes the int8 winner is i8dmm_gd2 (x @ tile.T under the guided schedule), not weight-major. Build u16dmm_gd2: per gd2 chunk of 128 rows, one np.take(LUT16U) into the int8 tile, cast to f32, then np.matmul(x, tile.T, out=y[:, rows]) (plus a down K-blocked variant). Run it against i8dmm_gd2 at B 64, 128 and 256 on the same layer. Does the u16 / i8dmm ratio fall from 1.43–1.64x to <= 1.15x, which would show the gap is orientation and schedule, not the take? Or does the per-chunk take (no longer one fused take per worker range) put it back above 1.3x?

## Script (v16p.py)

```python
#!/usr/bin/env python3
"""V16P: prefill-sized chunks B 64 / 128 for the 02:55 weight-major u16gft kernels vs int8, W 8 pool, full bitnetd layer, ctx-free, 05:5x MT 11/28.
Kernels and controls exactly as bench0255_1128/v16g.py (copied below); u16ad_st dropped (row-major decode kernel, 1.07-1.17x behind u16gft).
XREF has only 32 rows, so a 128-row XREFP (rng XSEED+2000) gets its own float64 reference, built from the trits decoded from P16 (int8: F8 Q)
times g in float64; that path is validated first against the original wf64 REF on XREF[:32] (asserted: ternary <= 1e-12, int8 <= 1e-6 since F8 g is float32).
Earlier V16G doc:
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
BS = (64, 128); TMPN = 128 * 256
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
C = {"u16gft256_st": ("u16gft256", 8, "st"), "u16gft128_st": ("u16gft128", 8, "st"), "u16gfk_st": ("u16gfk", 8, "st"),
     "i8g_gd2": ("8g", 128, "gd2"), "i8g_st": ("8g", 256, "st"), "i8dmm_gd2": ("8dmm", 128, "gd2")}
UG = ["u16gft256_st", "u16gft128_st", "u16gfk_st"]; U = UG; I8 = ["i8g_gd2", "i8g_st", "i8dmm_gd2"]
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
        for c in U: e[c]["vs_u16gft128_mab"] = float(np.abs(outs[c] - outs["u16gft128_st"]).max())
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
        for a_ in ("u16gft128_st", "u16gfk_st"):
            for b_ in ("i8g_gd2", "i8g_st"): pr_(a_, b_, f"{a_}/{b_}")
        for c in I8: pr_(bg, c, f"best_u16g/{c}")
        pr_(bg, bi, "best_u16g/best_i8")
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

close: 06a6b1edc5ef9e60
