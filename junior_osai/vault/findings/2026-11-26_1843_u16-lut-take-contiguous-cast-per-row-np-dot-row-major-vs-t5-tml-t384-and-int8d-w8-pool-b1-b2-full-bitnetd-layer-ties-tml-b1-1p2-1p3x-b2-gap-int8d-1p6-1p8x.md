---
local: true
date: 2026-11-26
wave: "2026-11-26 18:55 slot (fired 18:43 MT, ran ~18:43–18:55 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (8-trit uint16 pack + 6561-row u64 LUT take into the contiguous int8 tile, cast to f32, then per-row np.dot (row-major, u16d) on the W 8 decode layer, full bitnetd layer, ctx-free: at B 1 it only ties t5 tmL_T384 (tmL / best u16d 0.98–1.01x, bar >= 1.3x: missed); at B 2 it beats tmL by 1.20–1.28x (bar missed) and beats the weight-major u16g matmul by 1.42–1.43x (1.50–1.63x at B 1). It is still 1.67–1.82x (B 1) and 1.55–1.68x (B 2) slower than int8d (cast + per-row np.dot). Widening the uint16 index to intp first (u16xd) is a wash (0.96–1.07x); T 128 is 1.11–1.16x slower than T 256. Outputs match the float64 ternary reference to <= 1.04e-6 (cos 1.0), identical to u16g at B 1. Memory unchanged: 17.46 MB vs 69.56 MB int8 (3.98x, the known weight-memory win), LUT 52,488 B. No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, u64, uint16, eight-trit-pack, base-3, 2-bit, np-take, per-row-dot, row-major, tmac, tmL, int8, numpy, not-viable]
---

# u16 LUT take + per-row np.dot: ties tmL at B 1, 1.2–1.3x faster than tmL at B 2, still 1.6–1.8x behind int8d

The 14:44 note ([[2026-11-26_1444_8-trit-u16-pack-6561-row-u64-lut-take-contiguous-cast-vs-void5-int8-take-cast-weight-major-w8-pool-b1-b4-b8-vs-int8-full-bitnetd-layer-expand-1p3-1p6x-faster-layer-1p2-1p4x-gap-b4-b8-1p6-1p7x]]) made the 8-trit uint16 pack the best weight-major ternary path at B 4 / B 8, but at B 1 the weight-major matmul lost to tmL. It asked: feed the same u16 expand into per-row np.dot (the int8d pattern) and see whether that beats tmL at B 1 / B 2 by >= 1.3x.

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 1 and B 2, does np.take(LUT16U 6561 × u64, P16 uint16 tile) → contiguous int8 view → f32 tile → per-row np.dot(tile, x[b]) (u16d / u16xd) beat t5 tmL_T384 (T-MAC, the current B 1 ternary best) by >= 1.3x, and how far is it from int8d?

## Method

- Box: 8 CPUs, numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents, so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1455_1126/v16.py exec'd without main() (same weights, P5 and P16 packs with exact trit round-trip asserts, float64 references, dy / gd2 schedules, profiling hooks).
- Kernels (row-major x (B, in) → y (B, out)):
  - **u16d**: np.take(LUT16U, P16 tile, out=u64 scratch) → int8 view (n, in) → np.copyto f32 tile → for b: np.dot(tile, x[b], out=y[b, tile]). T 256 (dy, gd2) and T 128 (gd2).
  - **u16xd**: same, uint16 tile first widened into an intp scratch (stage idx). T 256 (dy, gd2).
  - Baselines: t5 tmL_T384 dy (B 1 ternary best), u16g_T256 gd2 (14:44 weight-major path), int8d_T128 dy / gd2 (cast + per-row np.dot).
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round; median of round medians, round min–max and per-round ratios. Then 15 instrumented layers per u16d config for stage worker-ms (wall/cpu summed over workers) and per-site span. Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench1855_1126 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16d.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Layer ms, median of round medians [round min–max], runs 1 / 2 / 3; then ratios; then stage worker ms (wall/cpu, runs 1, 2, 3) and error.

bytes {'t5': 13986304, 'u16': 17458176, 'int8': 69559296, 'int8_over_u16': 3.9843392574344536, 'int8_over_t5': 4.973386535856792, 'lut_u16': 52488}
### B1
| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | tmL_dy | 10.97 [8.64–11.74] | 9.40 [8.59–10.38] | 9.46 [8.23–10.93] |
| 1 | u16d_dy | 11.85 [10.81–13.69] | 10.00 [9.38–10.51] | 9.49 [9.24–10.61] |
| 1 | u16d_gd2 | 11.37 [9.77–12.34] | 9.34 [9.08–9.82] | 9.87 [9.21–10.86] |
| 1 | u16d128_gd2 | 13.19 [11.53–14.61] | 10.73 [10.07–11.25] | 10.98 [10.15–12.88] |
| 1 | u16xd_dy | 11.21 [9.72–11.79] | 9.81 [9.62–10.13] | 10.07 [9.44–10.95] |
| 1 | u16xd_gd2 | 11.29 [9.73–12.09] | 9.76 [9.46–10.52] | 10.12 [9.09–10.91] |
| 1 | u16g_gd2 | 16.76 [14.96–18.33] | 14.17 [13.78–15.53] | 15.48 [13.76–16.15] |
| 1 | i8d_dy | 6.69 [6.04–6.97] | 6.05 [5.78–6.69] | 6.31 [5.53–6.99] |
| 1 | i8d_gd2 | 6.15 [5.62–6.70] | 5.60 [5.41–6.71] | 5.68 [5.23–6.18] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | tmL/best_u16d | 0.979x [0.841–1.048] (tmL_dy/u16xd_dy) | 1.007x [0.924–1.112] (tmL_dy/u16d_gd2) | 0.997x [0.890–1.095] (tmL_dy/u16d_dy) |
| 1 | u16g/best_u16d | 1.496x [1.422–1.617] (u16g_gd2/u16xd_dy) | 1.518x [1.406–1.630] (u16g_gd2/u16d_gd2) | 1.632x [1.487–1.646] (u16g_gd2/u16d_dy) |
| 1 | best_u16d/best_i8d | 1.823x [1.665–1.868] (u16xd_dy/i8d_gd2) | 1.668x [1.390–1.769] (u16d_gd2/i8d_gd2) | 1.669x [1.497–1.770] (u16d_dy/i8d_gd2) |
| 1 | tmL/best_i8d | 1.785x [1.517–1.910] (tmL_dy/i8d_gd2) | 1.680x [1.546–1.785] (tmL_dy/i8d_gd2) | 1.664x [1.555–1.823] (tmL_dy/i8d_gd2) |
| 1 | u16d_gd2/u16xd_gd2 | 1.008x [0.884–1.065] (u16d_gd2/u16xd_gd2) | 0.957x [0.864–1.032] (u16d_gd2/u16xd_gd2) | 0.975x [0.863–1.063] (u16d_gd2/u16xd_gd2) |
| 1 | u16d_gd2/u16d128_gd2 | 0.862x [0.832–0.917] (u16d_gd2/u16d128_gd2) | 0.870x [0.821–0.955] (u16d_gd2/u16d128_gd2) | 0.899x [0.844–0.910] (u16d_gd2/u16d128_gd2) |

| B | config | idx | take | cast | dot | task | span |
|---|---|---|---|---|---|---|---|
| 1 | u16d_dy | – | 41.7/22.8, 32.8/19.8, 30.3/18.9 | 19.9/16.1, 15.9/12.9, 15.3/12.6 | 15.0/11.2, 14.1/10.0, 13.4/10.0 | 79.5/51.9, 64.7/44.4, 61.3/43.1 | 12.17, 10.16, 9.79 |
| 1 | u16d_gd2 | – | 40.9/24.4, 33.1/20.4, 37.2/22.0 | 22.3/19.3, 17.2/14.2, 18.4/15.3 | 14.4/11.0, 13.0/9.8, 14.0/10.3 | 80.7/57.0, 65.3/45.8, 71.8/49.8 | 12.36, 9.72, 10.71 |
| 1 | u16d128_gd2 | – | 57.7/24.0, 47.1/20.6, 46.5/20.4 | 24.4/16.9, 19.8/13.2, 18.8/12.7 | 18.6/10.4, 17.5/9.3, 16.2/9.1 | 104.2/54.2, 87.2/45.6, 83.7/44.5 | 14.59, 12.18, 11.73 |
| 1 | u16xd_dy | 11.8/5.8, 9.9/4.3, 9.9/4.1 | 28.2/17.9, 24.3/15.7, 22.1/14.5 | 20.4/17.3, 16.4/13.2, 14.7/12.3 | 14.7/11.2, 13.8/10.2, 13.5/9.8 | 76.3/54.5, 67.5/45.6, 61.6/42.3 | 11.85, 10.41, 9.69 |
| 1 | u16xd_gd2 | 12.3/6.5, 10.2/4.8, 9.9/4.9 | 30.5/19.9, 24.2/15.8, 24.6/16.0 | 22.9/19.8, 17.2/14.5, 17.3/14.6 | 14.7/11.3, 13.0/9.6, 13.3/9.7 | 83.1/59.5, 66.5/46.6, 66.9/47.3 | 12.30, 9.80, 9.84 |
| 1 | tmL_dy | – | – | – | – | 50.8/44.1, 48.7/40.3, 43.4/36.7 | 10.01, 8.77, 7.92 |

| B | config | max abs err vs ref | cos | instr identical | vs u16g mab |
|---|---|---|---|---|---|
| 1 | tmL_dy | 5.57e-07, 6.18e-07, 6.00e-07 | 1.000000000000 | True | 0.0e+00 |
| 1 | u16d_dy | 9.46e-07, 8.68e-07, 1.04e-06 | 1.000000000000 | True | 0.0e+00 |
| 1 | u16d_gd2 | 9.46e-07, 8.68e-07, 1.04e-06 | 1.000000000000 | True | 0.0e+00 |
| 1 | u16d128_gd2 | 9.46e-07, 8.68e-07, 1.04e-06 | 1.000000000000 | True | 0.0e+00 |
| 1 | u16xd_dy | 9.46e-07, 8.68e-07, 1.04e-06 | 1.000000000000 | True | 0.0e+00 |
| 1 | u16xd_gd2 | 9.46e-07, 8.68e-07, 1.04e-06 | 1.000000000000 | True | 0.0e+00 |
| 1 | u16g_gd2 | 9.46e-07, 8.68e-07, 1.04e-06 | 1.000000000000 | True | 0.0e+00 |
| 1 | i8d_dy | 3.19e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00 |
| 1 | i8d_gd2 | 3.19e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00 |

### B2
| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | tmL_dy | 16.01 [13.73–16.79] | 14.51 [12.41–17.04] | 13.50 [12.51–14.44] |
| 2 | u16d_dy | 13.41 [11.87–14.29] | 12.43 [11.86–13.53] | 12.62 [11.85–13.25] |
| 2 | u16d_gd2 | 12.76 [10.82–13.40] | 12.16 [10.91–13.62] | 11.73 [11.01–13.92] |
| 2 | u16d128_gd2 | 14.53 [12.47–16.70] | 13.47 [12.11–15.14] | 13.22 [12.82–14.66] |
| 2 | u16xd_dy | 12.72 [11.01–13.06] | 12.37 [11.57–14.56] | 11.47 [10.88–12.18] |
| 2 | u16xd_gd2 | 12.90 [10.82–13.68] | 11.37 [11.03–13.69] | 11.29 [10.77–11.94] |
| 2 | u16g_gd2 | 18.14 [15.89–19.88] | 16.17 [15.43–18.15] | 16.03 [15.71–17.26] |
| 2 | i8d_dy | 8.89 [7.35–9.44] | 8.00 [7.26–8.89] | 7.63 [7.29–7.76] |
| 2 | i8d_gd2 | 8.21 [6.77–9.56] | 7.20 [6.50–118.48] | 6.74 [6.25–7.75] |

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | tmL/best_u16d | 1.259x [1.226–1.307] (tmL_dy/u16xd_dy) | 1.276x [1.126–1.396] (tmL_dy/u16xd_gd2) | 1.195x [1.119–1.249] (tmL_dy/u16xd_gd2) |
| 2 | u16g/best_u16d | 1.426x [1.341–1.548] (u16g_gd2/u16xd_dy) | 1.422x [1.326–1.446] (u16g_gd2/u16xd_gd2) | 1.419x [1.391–1.489] (u16g_gd2/u16xd_gd2) |
| 2 | best_u16d/best_i8d | 1.550x [1.240–1.687] (u16xd_dy/i8d_gd2) | 1.578x [0.116–1.696] (u16xd_gd2/i8d_gd2) | 1.676x [1.446–1.808] (u16xd_gd2/i8d_gd2) |
| 2 | tmL/best_i8d | 1.951x [1.560–2.166] (tmL_dy/i8d_gd2) | 2.014x [0.141–2.181] (tmL_dy/i8d_gd2) | 2.003x [1.806–2.162] (tmL_dy/i8d_gd2) |
| 2 | u16d_gd2/u16xd_gd2 | 0.990x [0.913–1.070] (u16d_gd2/u16xd_gd2) | 1.069x [0.975–1.099] (u16d_gd2/u16xd_gd2) | 1.039x [0.946–1.257] (u16d_gd2/u16xd_gd2) |
| 2 | u16d_gd2/u16d128_gd2 | 0.878x [0.802–0.990] (u16d_gd2/u16d128_gd2) | 0.902x [0.828–0.992] (u16d_gd2/u16d128_gd2) | 0.887x [0.846–1.045] (u16d_gd2/u16d128_gd2) |

| B | config | idx | take | cast | dot | task | span |
|---|---|---|---|---|---|---|---|
| 2 | u16d_dy | – | 38.5/20.2, 29.1/18.8, 37.1/20.6 | 16.2/12.9, 13.8/12.4, 15.7/12.8 | 26.8/19.1, 23.8/18.4, 26.5/19.2 | 83.2/53.9, 68.4/51.2, 81.2/54.3 | 12.97, 10.94, 12.76 |
| 2 | u16d_gd2 | – | 35.2/20.1, 32.0/19.5, 35.1/20.6 | 16.0/13.4, 16.3/13.9, 16.6/14.3 | 24.3/17.9, 23.5/18.0, 24.8/18.6 | 76.9/53.1, 74.2/53.3, 79.0/55.0 | 11.18, 10.96, 11.54 |
| 2 | u16d128_gd2 | – | 53.5/22.0, 50.3/22.3, 48.1/21.0 | 20.8/13.5, 20.9/14.6, 19.0/13.0 | 33.8/17.7, 31.2/17.9, 30.1/16.9 | 111.6/56.4, 105.5/57.1, 99.2/53.4 | 15.35, 14.67, 13.66 |
| 2 | u16xd_dy | 9.8/4.0, 10.5/4.5, 10.4/4.1 | 23.6/15.3, 24.8/16.2, 23.0/15.0 | 14.7/12.5, 15.7/13.4, 14.4/12.5 | 25.1/18.6, 25.3/19.2, 24.4/18.6 | 75.5/52.3, 79.0/55.7, 73.7/51.9 | 11.78, 12.13, 11.60 |
| 2 | u16xd_gd2 | 10.9/5.1, 10.4/5.1, 11.0/4.9 | 26.1/16.6, 25.4/17.0, 24.3/16.0 | 17.0/14.7, 17.5/14.8, 16.7/14.0 | 24.8/18.6, 25.5/19.6, 23.9/18.4 | 80.8/56.7, 81.2/57.3, 78.6/55.2 | 11.80, 11.82, 11.23 |
| 2 | tmL_dy | – | – | – | – | 72.2/60.2, 67.3/56.5, 80.4/67.0 | 12.82, 11.28, 14.17 |

| B | config | max abs err vs ref | cos | instr identical | vs u16g mab |
|---|---|---|---|---|---|
| 2 | tmL_dy | 6.27e-07, 6.18e-07, 6.05e-07 | 1.000000000000 | True | 0.0e+00 |
| 2 | u16d_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 1.4e-06 |
| 2 | u16d_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 1.4e-06 |
| 2 | u16d128_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 1.4e-06 |
| 2 | u16xd_dy | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 1.4e-06 |
| 2 | u16xd_gd2 | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 1.4e-06 |
| 2 | u16g_gd2 | 9.50e-07, 8.85e-07, 8.50e-07 | 1.000000000000 | True | 0.0e+00 |
| 2 | i8d_dy | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00 |
| 2 | i8d_gd2 | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00 |


## Verdict

**not-viable.** Moving the u16 expand from a weight-major matmul to per-row np.dot is a real ternary-internal win (1.42–1.63x over u16g at B 1 / B 2), and at B 2 it is now the best ternary path (1.20–1.28x over tmL), but it misses the 1.3x bar vs tmL at both batch sizes (B 1 is a tie, 0.98–1.01x) and stays 1.55–1.82x slower than int8d. Profiles show why: take (~23–41 worker-ms) plus cast (~15–22) are 65–80% of the u16d worker time at B 1, while the dot itself is ~13–15 ms, about what int8d spends on its whole cast + dot. Ternary keeps only the known 3.98x weight-memory win (17.46 vs 69.56 MB).

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-26 18:55 wave reached this lane. JuniorLLM HEAD at clone was f0e7489 (cloudcover95, 14:49 MT, this lane's 14:44 note); nothing else landed after it, so no Trit Pack, Handshake or LLC commit falls in the last 6 hours. This note only adds one new file under junior_osai/vault/findings/.

## Next question

At B 2 / B 4 row-major, does a larger u16d tile (T 512 / 1024, fewer take/cast calls per worker) or one fused take over all rows of a worker's range before the per-row dots beat u16d_T256 by >= 1.3x, and does that bring u16d within 1.3x of int8d?

## Script (v16d.py)

```python
#!/usr/bin/env python3
"""V16D: 8-trit uint16 pack + 6561-row u64 LUT take -> contiguous int8 tile -> f32 -> per-row np.dot (row-major, y[b] = W x[b]),
vs t5 tmL_T384 (T-MAC, current B1 ternary best) and int8d (cast + per-row np.dot) at B1 / B2, W=8 decode layer, full bitnetd layer, ctx-free.
Base: /workspace/bench1455_1126/v16.py (exec'd without main(); same weights, P5 + P16 packs with exact trit round-trip asserts, refs, schedules).
  u16d  : np.take(LUT16U, P16 uint16 tile, out=u64) -> int8 view -> f32 tile -> for b: np.dot(tile, x[b])
  u16xd : same, uint16 index first widened to an intp scratch (np.copyto) before np.take
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16d.py out.json"""
import json, os, sys
import numpy as np
_S = open("/workspace/bench1455_1126/v16.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "v16.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
def r_u16d(PR, x, y, r0, r1, T, w, i):
    q = PR.shape[1]; su = scr(("u16", w, T), (T * Q8,), np.uint64)[:T * q].reshape(T, q)
    s = scr(("f16", w, T), (T * F,), np.float32)[:T * q * 8].reshape(T, q * 8); s8 = su.view(np.int8).reshape(T, q * 8); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16U, PR[o0:o0+n], out=su[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.copyto(s[:n], s8[:n], casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        for b in range(x.shape[0]): np.dot(s[:n], x[b], out=y[b, o0:o0+n])
        if P: _add(w, "dot", pc() - t0, tt() - c0)
def r_u16xd(PR, x, y, r0, r1, T, w, i):
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
        for b in range(x.shape[0]): np.dot(s[:n], x[b], out=y[b, o0:o0+n])
        if P: _add(w, "dot", pc() - t0, tt() - c0)
KD = {"u16d": r_u16d, "u16xd": r_u16xd}
_mm_v16 = mm
def mm(k, a, kind, T, mode, rec):
    if kind not in KD: return _mm_v16(k, a, kind, T, mode, rec)
    CUR[0] = k; B = a.shape[0]; P = P16[k]; R = P["PR"]; i = P["inp"]; out = R.shape[0]
    x = np.ascontiguousarray(a, np.float32); y = np.empty((B, out), np.float32)
    run_site(KD[kind], R, x, y, out, T, i, mode, rec, k, "t5vg")
    y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
C = {"tmL_dy": ("tmL", 384, "dy"), "u16d_dy": ("u16d", 256, "dy"), "u16d_gd2": ("u16d", 256, "gd2"), "u16d128_gd2": ("u16d", 128, "gd2"),
     "u16xd_dy": ("u16xd", 256, "dy"), "u16xd_gd2": ("u16xd", 256, "gd2"), "u16g_gd2": ("u16g", 256, "gd2"), "i8d_dy": ("8d", 128, "dy"), "i8d_gd2": ("8d", 128, "gd2")}
U = [c for c in C if C[c][0] in KD]; I8 = ["i8d_dy", "i8d_gd2"]
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, time={}, prof={}, err={})
    W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
    W16 = sum(P16[k]["PR"].nbytes + P16[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(t5=W5, u16=W16, int8=W8, int8_over_u16=W8 / W16, int8_over_t5=W8 / W5, lut_u16=LUT16U.nbytes)
    print("bytes", res["bytes"], flush=True)
    names = list(C)
    for B in (1, 2):
        xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0] == "8d" else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in U: e[c]["vs_u16g_mab"] = float(np.abs(outs[c] - outs["u16g_gd2"]).max()); e[c]["vs_tmL_mab"] = float(np.abs(outs[c] - outs["tmL_dy"]).max())
        res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.12f}", v["instr_identical"], f"{v.get('vs_u16g_mab', 0):.1e}") for c, v in e.items()}, flush=True)
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
        bu = min(U, key=lambda c: o[c]["med"]); bi = min(I8, key=lambda c: o[c]["med"])
        pr_("tmL_dy", bu, "tmL/best_u16d"); pr_("u16g_gd2", bu, "u16g/best_u16d"); pr_(bu, bi, "best_u16d/best_i8d"); pr_("tmL_dy", bi, "tmL/best_i8d")
        pr_("u16d_gd2", "u16xd_gd2"); pr_("u16d_gd2", "u16d128_gd2")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:12s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
        pr = {}; PRF[0] = True
        for c in U + ["tmL_dy"]:
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
            print(f"B{B} prof {c:12s} " + "  ".join(f"{st} {v['wall']:.1f}/{v['cpu']:.1f}ms" for st, v in pr[c].items()), flush=True)
        PRF[0] = False; res["prof"][f"B{B}"] = pr
    json.dump(res, open(sys.argv[1], "w"), indent=1)
main()
```

close: d65eeb4955ab866f
