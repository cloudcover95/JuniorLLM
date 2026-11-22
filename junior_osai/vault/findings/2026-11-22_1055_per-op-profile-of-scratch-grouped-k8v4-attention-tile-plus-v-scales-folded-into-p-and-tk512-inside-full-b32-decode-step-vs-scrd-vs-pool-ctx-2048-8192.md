---
local: true
date: 2026-11-22
wave: "2026-11-22 10:55 (fired 10:48 MT, ran ~10:49–11:00 MT; no pair reports reached chat this wave, quant derivation lane paused)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "inconclusive (step bar met by scrDV on both seeds, but the V-scale fold is speed-neutral vs scrD and attention stays at ~64–69 ms; TK 512 not viable)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, decode-step, end-to-end, attention, grouped, scratch, profile, per-op, scale-folding, tile-size, kv-cache, k8v4, u32-lut, b32, numpy, inconclusive]
---

# Per-op profile of the scratch grouped K8V4 tile at B = 32 / ctx 8192: ~70% of attention is decode (K dequant 24–24%, V4 nibble take 23–23%, V scale 22–23%), only ~24% is q·K + P·V; folding the V scales into P is speed-neutral (attention-only 0.97–1.03x vs scrD) and TK 512 is slower (0.80–0.90x), so attention stays at 64–69 ms (target ≤ ~50 ms missed)

The 05:55 note ([[2026-11-22_0555_per-worker-preallocated-f32-k-v-score-scratch-for-grouped-k8v4-attention-k8-codes-cast-scales-folded-into-q-k-inside-full-b32-decode-step-vs-alloc-grouped-vs-pool-ctx-2048-8192]]) asked: scratch takes allocation out, and B = 32 / ctx 8192 grouped attention is still ~59–62 ms inside the step. Where does the per-tile time go? Time each op of the scrD tile separately (K cast/dequant, q·K, max/exp/sum, V4 take + scale, P·V) at B = 32 / ctx 8192. Then test folding the four g32 V4 scales into P (P·diag(s_g)·codes_g, so the V nibbles are only cast, never scaled) and TK 512 with scratch (L2-resident tiles). Does either bring attention to ≤ ~50 ms and the step to a robust ≥ 1.5x vs pool (per-round low ≥ 1.5) at max abs ≤ 1e-5? Answers:
- **Where the time goes (scrD, TK 1024, W = 32 tasks on 8 threads, per-op thread time attributed to wall, r1; r2):** K dequant 17.8; 15.5 ms, q·K 12.0; 10.4, max/exp/sum 5.7; 4.1, V4 nibble take 17.3; 14.9, V scale 16.2; 14.4, P·V 5.7; 4.8 ms. Decode (K dequant + V take + V scale) is **69–70%** of the tile and the two matmuls are only 24–24%. The matmuls have just G = 4 query rows per KV head, so they behave like GEMVs over the f32 tile. The single-thread profile (one 5-head task, no contention) has the same shape. The kernel is bound by writing and re-reading the f32 K/V tiles, not by FLOPs.
- **V scales folded into P (scrDV): no speedup.** It roughly halves the serial V-scale cost (3.06–3.26 → 1.62–1.83 ms per task), because only P (4 × 4 × L) is scaled instead of L × 128. But the codes still have to be cast into a group-major f32 scratch (a transposed copy that writes the same bytes), and the batched (bh, 4, G, L)·(bh, 4, L, 32) matmul is a little slower than one P·V. Attention-only vs scrD at B = 32 / ctx 8192: **0.97–1.03x**; across all points 0.97–1.03x. In-step attention is 68.6 / 63.6 ms vs scrD 67.7 / 64.1 ms. Step scrDV vs scrD: 0.99x / 0.99x.
- **TK 512 (L2-sized tiles): slower.** Attention-only is 0.84–0.84x of scrD at B = 32 / ctx 8192 (scrDV512 0.77–0.78x), and in-step attention is 87.1 / 75.2 ms. The per-tile fixed costs double (max/exp/sum goes from 4.1–5.7 to 8.5–13.3 ms attributed), and the smaller tiles don't make the decode cheaper.
- **Step vs pool at ctx 8192 / B = 32:** scrDV **1.73x / 1.60x** (per-round range 1.53–1.81). scrD 1.75x / 1.62x (1.50–1.92). scrF 1.76x / 1.57x (1.54–1.81). scrD512 1.46x / 1.43x. scrDV512 1.39x / 1.40x. scrDV's per-round low is ≥ 1.5 on both seeds, and scrD misses that by 0.002 on run2. scrDV and scrD are within 1% of each other, so this clearance is round-to-round noise, not an effect of the fold. The pool step was also slower this wave on run1 (166 ms vs 136 ms at 05:55; the box was busier), which helps every ratio. scrDV runs at 333 / 369 tok/s vs pool 193 / 230.
- **ctx 2048 / B = 32 and ctx 8192 / B = 16 (context):** at 2048, scrD 1.74x / 1.64x and scrDV 1.67x / 1.71x vs the pool step. At 8192 / B = 16, scrD 1.56x / 1.41x and scrDV 1.48x / 1.39x; B = 16 still straddles 1.5.
- **Error:** scrDV max abs vs pool ≤ **1.1e-06** inside the step (vs scrD ≤ 2.4e-07). TK 512 variants are ≤ 2.0e-06 (different online-softmax tiling), and min layer cos is unchanged. Large-q smoke test (gs.py-style q = 2·N(0,1)·SC, seed 1, B = 32 / ctx 8192): scrDV is within 4.8e-07 of scrD at TK 1024 (05:55: scrD vs pool 3.3e-6), so it stays under 1e-5 by construction. scrF (K-scale fold) is again 2.1e-05 off scrD on large q, so scrF remains unsafe for the 1e-5 bar.
- **Bytes:** KV + weights are unchanged (295.40 MiB u32 at ctx 8192, 92.90 at ctx 2048). The scrDV scratch is 48.83 MiB (8 workers; adds the P-scaled PS and O buffers) vs 46.25 MiB for 05:55 scrD. Nothing here changes memory by 1.5x.

## Method

- `step.py` is a copy of the 05:55 `step.py` (`/workspace/bench0555_1122`), which reuses `base.build` + `par.prep / par.fmm` (2-bit u32-LUT, T = 256, 8 workers) from `/workspace/bench1455_1118`, `step()` verbatim from `dstep.py` (05:55 11/21), `gs.py` (18:55 11/21) + `iq.py` (02:55 11/21), and `sk.py` (05:55 11/22: scrD / scrF). New this wave: `sk2.py` (inline below), which is sk.py generalized. `kmode` "D" is bit-identical to scrD. `vfold` casts V4 nibbles (4 × g32) / V8 window codes (2 × g64) into a group-major f32 scratch (bh, ng, L, gsz), scales P per group into PS (bh, ng, G, L), and does one batched matmul into O, added into acc viewed as (bh, G, ng, gsz). An optional `prof` dict accumulates per-op perf_counter time per task.
- Variants in the step (u32 weights, W = clamp(B, 8, 32)): scrD (TK 1024, baseline from 05:55), scrDV (TK 1024), scrD512, scrDV512, scrF (TK 1024, reference), pool (one task per (stream, KV head), TK 2048; the baseline for the bar). Grid: ctx 2048 × B 32 and ctx 8192 × B {16, 32}. Same K8V4 format (A8 g64 K, A4 g32 old V, A8 1024-token window), fixed-n cache and f32 reference step as 02:55/05:55. Timing: 5 rounds × (2 warm + 6 timed) full steps per variant, order rotated each round (30 timed steps per variant). Attention-only: 5 rounds × 3 reps per variant on the same q, rotated. Speedups use per-round medians, and "per-round low" is the minimum over the 5 rounds.
- Profile (`prof.py`): B = 32 / ctx 8192, W = 32, 5 rounds × 3 reps. Each rep runs an uninstrumented call (wall) and an instrumented call (per-op time summed over tasks; share × wall = attributed ms; the instrumentation overhead is visible as instr vs wall). It also runs a serial single 5-head task on the main thread for clean per-op ms.
- OPENBLAS_NUM_THREADS = 1, 8-thread attention pool + 8 weight workers, 8-core box. Two seeds: run1 11221055, run2 11220555 (= 05:55 run1 seed).
- Command: `cd /workspace/bench1055_1122 && OPENBLAS_NUM_THREADS=1 SEED=11221055 python3 step.py results_run1.json > run1.txt && OPENBLAS_NUM_THREADS=1 SEED=11220555 python3 step.py results_run2.json > run2.txt && OPENBLAS_NUM_THREADS=1 SEED=11221055 python3 prof.py prof_run1.json && OPENBLAS_NUM_THREADS=1 SEED=11220555 python3 prof.py prof_run2.json && OPENBLAS_NUM_THREADS=1 SEED=1 python3 smoke.py > smoke.txt` (step ~2.5 min per seed, profile ~35 s per seed).

## Results

### Summary (K8V4, u32-LUT weights, medians, run1; run2)

| ctx | B | step ms scrD / scrDV / scrD512 / scrDV512 / scrF / pool (r1; r2) | attn ms in step, same order (r1; r2) | vs pool step: scrD / scrDV / scrD512 / scrDV512 / scrF (r1, r2; per-round low) | scrDV vs scrD step | attn-only ms scrD / scrDV / scrD512 / scrDV512 (r1; r2) | total MiB u32 (KV) | attn max abs vs pool scrD / scrDV / scrD512 | min layer cos |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 32 | 45.5 / 47.6 / 48.4 / 49.2 / 45.3 / 79.3; 43.3 / 41.4 / 44.7 / 44.1 / 41.6 / 71.0 | 19.8 / 20.2 / 22.0 / 23.5 / 19.7 / 52.2; 19.2 / 18.8 / 21.1 / 21.5 / 18.6 / 48.1 | 1.74, 1.64 (1.51) / 1.67, 1.71 (1.46) / 1.64, 1.59 (1.53) / 1.61, 1.61 (0.99) / 1.75, 1.71 (1.56) | 0.96x / 1.05x | 19.0 / 18.7 / 21.1 / 22.5; 16.7 / 17.2 / 19.9 / 20.7 | 92.90 (76.25) | 0.0e+00 / 2.4e-07 / 1.3e-06 | 0.999325 |
| 8192 | 16 | 61.3 / 64.8 / 72.2 / 74.8 / 62.8 / 95.7; 60.5 / 61.4 / 67.3 / 67.8 / 60.3 / 85.5 | 36.8 / 40.5 / 48.1 / 51.3 / 38.9 / 72.7; 38.5 / 39.0 / 45.3 / 44.4 / 38.0 / 60.7 | 1.56, 1.41 (1.26) / 1.48, 1.39 (1.10) / 1.33, 1.27 (1.14) / 1.28, 1.26 (1.13) / 1.52, 1.42 (1.28) | 0.95x / 0.98x | 37.4 / 38.0 / 47.0 / 49.1; 33.4 / 34.1 / 40.3 / 43.0 | 156.02 (139.38) | 1.1e-06 / 1.1e-06 / 1.1e-06 | 0.999415 |
| 8192 | 32 | 95.0 / 96.0 / 113.7 / 119.4 / 94.2 / 166.1; 86.1 / 86.8 / 97.4 / 99.1 / 88.9 / 139.2 | 67.7 / 68.6 / 87.1 / 93.6 / 67.0 / 140.1; 64.1 / 63.6 / 75.2 / 77.7 / 64.3 / 116.5 | 1.75, 1.62 (1.50) / 1.73, 1.60 (1.53) / 1.46, 1.43 (1.33) / 1.39, 1.40 (1.22) / 1.76, 1.57 (1.54) | 0.99x / 0.99x | 60.7 / 59.1 / 72.5 / 78.1; 55.3 / 56.9 / 66.1 / 71.9 | 295.40 (278.75) | 1.1e-06 / 1.1e-06 / 2.0e-06 | 0.999415 |

### Per-op profile (B = 32, ctx 8192, attention only)

| variant (B 32, ctx 8192) | attn-only wall ms (min–max) r1; r2 | k_deq ms (share) r1; r2 | qk ms (share) r1; r2 | softmax ms (share) r1; r2 | v_take ms (share) r1; r2 | v_scale ms (share) r1; r2 | pv ms (share) r1; r2 | serial 1-task ms r1; r2 | serial per-op ms r1 (k_deq / qk / softmax / v_take / v_scale / pv) |
|---|---|---|---|---|---|---|---|---|---|
| D_TK1024 | 74.8 (69.9–82.3); 64.2 (59.6–77.8) | 17.8 (24%); 15.5 (24%) | 12.0 (16%); 10.4 (16%) | 5.7 (8%); 4.1 (6%) | 17.3 (23%); 14.9 (23%) | 16.2 (22%); 14.4 (23%) | 5.7 (8%); 4.8 (7%) | 12.36; 11.62 | 3.16 / 2.18 / 0.36 / 2.74 / 3.26 / 0.89 |
| DV_TK1024 | 78.4 (68.0–90.8); 58.4 (52.7–65.4) | 17.7 (23%); 14.6 (25%) | 12.9 (17%); 9.7 (17%) | 6.9 (9%); 4.8 (8%) | 20.2 (26%); 14.3 (24%) | 13.4 (17%); 9.8 (17%) | 7.3 (9%); 5.2 (9%) | 11.11; 10.40 | 3.22 / 2.18 / 0.38 / 2.91 / 1.83 / 1.06 |
| D_TK512 | 91.6 (84.9–101.2); 67.9 (64.6–74.2) | 18.0 (20%); 15.3 (23%) | 12.6 (14%); 9.6 (14%) | 13.3 (15%); 8.5 (13%) | 23.4 (26%); 16.6 (24%) | 17.5 (19%); 13.4 (20%) | 6.9 (7%); 4.4 (6%) | 12.05; 10.65 | 2.87 / 1.96 / 0.53 / 2.82 / 3.20 / 0.59 |
| DV_TK512 | 129.0 (121.4–138.5); 78.2 (73.4–88.0) | 25.5 (20%); 15.9 (20%) | 16.0 (12%); 10.9 (14%) | 17.6 (14%); 11.4 (15%) | 34.8 (27%); 20.1 (26%) | 23.1 (18%); 13.1 (17%) | 12.0 (9%); 6.8 (9%) | 12.41; 9.67 | 3.77 / 2.18 / 0.57 / 2.80 / 2.07 / 1.02 |
| F_TK1024 | 83.0 (74.7–96.6); 57.7 (53.8–65.9) | 9.3 (11%); 6.2 (11%) | 21.7 (26%); 14.9 (26%) | 6.9 (8%); 4.8 (8%) | 21.5 (26%); 14.0 (24%) | 16.9 (20%); 13.2 (23%) | 6.7 (8%); 4.5 (8%) | 10.19; 9.43 | 0.97 / 2.52 / 0.31 / 2.59 / 3.21 / 0.86 |

### Run 1 (SEED 11221055)

| cfg | step ms med (min–max) | IQR | qkv ms | attn ms | rest ms | attn share | tok/s | vs u32+pool (per-round range) | vs u32+scrD (range) | W / TK | total MiB (KV) | layer cos vs f32-attn step | attn max abs vs pool | attn max abs vs scrD | attn min cos vs f32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u32+scrD_ctx2048_B32 | 45.53 (39.64–53.08) | 4.78 | 3.86 | 19.77 | 21.75 | 0.431 | 702.8 | 1.741x (1.51–1.87) | 1.000x (1.00–1.00) | 32 / 1024 | 92.90 (76.25) | 0.999325 | 0.0e+00 | 0.0e+00 | 0.992919 |
| u32+scrDV_ctx2048_B32 | 47.59 (40.51–55.06) | 7.83 | 4.05 | 20.20 | 22.65 | 0.439 | 672.4 | 1.665x (1.46–1.94) | 0.957x (0.85–1.07) | 32 / 1024 | 92.90 (76.25) | 0.999325 | 1.5e-07 | 1.5e-07 | 0.992919 |
| u32+scrD512_ctx2048_B32 | 48.40 (41.37–55.59) | 4.69 | 4.03 | 22.04 | 22.08 | 0.462 | 661.2 | 1.638x (1.54–1.90) | 0.941x (0.89–1.05) | 32 / 512 | 92.90 (76.25) | 0.999325 | 1.3e-06 | 1.3e-06 | 0.992919 |
| u32+scrDV512_ctx2048_B32 | 49.24 (42.02–116.91) | 5.32 | 4.04 | 23.46 | 22.56 | 0.474 | 649.9 | 1.610x (1.48–1.71) | 0.925x (0.88–0.99) | 32 / 512 | 92.90 (76.25) | 0.999325 | 1.2e-06 | 1.2e-06 | 0.992919 |
| u32+scrF_ctx2048_B32 | 45.31 (40.07–56.19) | 5.69 | 4.08 | 19.75 | 21.45 | 0.433 | 706.2 | 1.749x (1.70–1.80) | 1.005x (0.92–1.13) | 32 / 1024 | 92.90 (76.25) | 0.999325 | 4.4e-06 | 4.4e-06 | 0.992919 |
| u32+pool_ctx2048_B32 | 79.26 (70.80–89.05) | 8.95 | 3.95 | 52.16 | 23.07 | 0.662 | 403.8 | 1.000x (1.00–1.00) | 0.574x (0.53–0.66) | 160 / 2048 | 92.90 (76.25) | 0.999325 | 0.0e+00 | 0.0e+00 | 0.992919 |
| u32+scrD_ctx8192_B16 | 61.26 (54.74–72.20) | 3.76 | 3.63 | 36.78 | 20.03 | 0.613 | 261.2 | 1.563x (1.29–1.69) | 1.000x (1.00–1.00) | 16 / 1024 | 156.02 (139.38) | 0.999775 | 1.1e-06 | 0.0e+00 | 0.989939 |
| u32+scrDV_ctx8192_B16 | 64.76 (56.89–82.76) | 8.83 | 3.63 | 40.49 | 20.40 | 0.620 | 247.0 | 1.478x (1.10–1.58) | 0.946x (0.86–1.05) | 16 / 1024 | 156.02 (139.38) | 0.999775 | 1.1e-06 | 1.2e-07 | 0.989939 |
| u32+scrD512_ctx8192_B16 | 72.22 (55.90–79.89) | 9.22 | 3.58 | 48.07 | 20.09 | 0.668 | 221.6 | 1.326x (1.19–1.57) | 0.848x (0.76–1.07) | 16 / 512 | 156.02 (139.38) | 0.999775 | 8.3e-07 | 8.9e-07 | 0.989939 |
| u32+scrDV512_ctx8192_B16 | 74.80 (65.47–86.37) | 5.18 | 3.62 | 51.34 | 20.16 | 0.678 | 213.9 | 1.280x (1.17–1.33) | 0.819x (0.76–0.91) | 16 / 512 | 156.02 (139.38) | 0.999775 | 8.3e-07 | 8.9e-07 | 0.989939 |
| u32+scrF_ctx8192_B16 | 62.79 (50.17–73.54) | 6.29 | 3.44 | 38.86 | 19.73 | 0.626 | 254.8 | 1.525x (1.39–1.58) | 0.976x (0.93–1.20) | 16 / 1024 | 156.02 (139.38) | 0.999775 | 2.0e-06 | 2.0e-06 | 0.989939 |
| u32+pool_ctx8192_B16 | 95.73 (77.33–102.62) | 9.04 | 3.47 | 72.66 | 19.70 | 0.752 | 167.1 | 1.000x (1.00–1.00) | 0.640x (0.59–0.78) | 80 / 2048 | 156.02 (139.38) | 0.999775 | 0.0e+00 | 1.1e-06 | 0.989939 |
| u32+scrD_ctx8192_B32 | 95.01 (79.43–115.65) | 10.69 | 3.90 | 67.70 | 23.15 | 0.710 | 336.8 | 1.748x (1.50–1.92) | 1.000x (1.00–1.00) | 32 / 1024 | 295.40 (278.75) | 0.999775 | 1.1e-06 | 0.0e+00 | 0.989939 |
| u32+scrDV_ctx8192_B32 | 96.04 (79.42–115.35) | 8.40 | 4.02 | 68.58 | 22.75 | 0.719 | 333.2 | 1.730x (1.56–1.81) | 0.989x (0.95–1.04) | 32 / 1024 | 295.40 (278.75) | 0.999775 | 1.1e-06 | 1.2e-07 | 0.989939 |
| u32+scrD512_ctx8192_B32 | 113.74 (90.32–130.05) | 23.89 | 3.87 | 87.12 | 22.10 | 0.767 | 281.3 | 1.460x (1.41–1.60) | 0.835x (0.78–1.07) | 32 / 512 | 295.40 (278.75) | 0.999775 | 8.3e-07 | 8.9e-07 | 0.989939 |
| u32+scrDV512_ctx8192_B32 | 119.38 (96.77–142.10) | 16.91 | 3.85 | 93.60 | 22.00 | 0.781 | 268.1 | 1.391x (1.22–1.58) | 0.796x (0.71–0.91) | 32 / 512 | 295.40 (278.75) | 0.999775 | 8.3e-07 | 8.9e-07 | 0.989939 |
| u32+scrF_ctx8192_B32 | 94.20 (79.66–108.95) | 12.04 | 3.89 | 67.00 | 21.68 | 0.726 | 339.7 | 1.763x (1.66–1.81) | 1.009x (0.93–1.12) | 32 / 1024 | 295.40 (278.75) | 0.999775 | 2.3e-06 | 2.3e-06 | 0.989939 |
| u32+pool_ctx8192_B32 | 166.11 (129.74–189.53) | 19.85 | 3.82 | 140.11 | 22.28 | 0.839 | 192.6 | 1.000x (1.00–1.00) | 0.572x (0.52–0.66) | 160 / 2048 | 295.40 (278.75) | 0.999775 | 0.0e+00 | 1.1e-06 | 0.989939 |

| attention only (same q, no weight pass) | scrD ms (min–max) | scrDV ms (min–max) | scrD512 ms (min–max) | scrDV512 ms (min–max) | scrF ms (min–max) | pool ms (min–max) | scrDV/scrD | scrD512/scrD | scrDV512/scrD |
|---|---|---|---|---|---|---|---|---|---|
| ctx2048_B32 | 18.97 (16.53–22.65) | 18.67 (17.03–21.90) | 21.10 (19.11–25.21) | 22.54 (20.99–25.20) | 18.80 (16.34–20.21) | 50.58 (46.19–53.21) | 1.016x | 0.899x | 0.841x |
| ctx8192_B16 | 37.37 (31.40–45.58) | 37.98 (31.61–44.04) | 46.98 (35.72–55.46) | 49.12 (37.93–55.46) | 36.19 (30.96–45.21) | 62.63 (55.41–75.49) | 0.984x | 0.795x | 0.761x |
| ctx8192_B32 | 60.73 (52.49–67.99) | 59.08 (55.52–71.61) | 72.55 (66.27–82.55) | 78.09 (73.79–95.38) | 59.82 (57.26–67.87) | 109.47 (104.38–116.74) | 1.028x | 0.837x | 0.778x |

### Run 2 (SEED 11220555)

| cfg | step ms med (min–max) | IQR | qkv ms | attn ms | rest ms | attn share | tok/s | vs u32+pool (per-round range) | vs u32+scrD (range) | W / TK | total MiB (KV) | layer cos vs f32-attn step | attn max abs vs pool | attn max abs vs scrD | attn min cos vs f32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u32+scrD_ctx2048_B32 | 43.33 (36.69–51.13) | 4.93 | 3.47 | 19.21 | 20.16 | 0.448 | 738.6 | 1.638x (1.51–1.80) | 1.000x (1.00–1.00) | 32 / 1024 | 92.90 (76.25) | 0.999507 | 0.0e+00 | 0.0e+00 | 0.991451 |
| u32+scrDV_ctx2048_B32 | 41.45 (37.86–56.72) | 3.99 | 3.48 | 18.81 | 19.26 | 0.453 | 772.1 | 1.713x (1.66–1.75) | 1.045x (0.96–1.11) | 32 / 1024 | 92.90 (76.25) | 0.999507 | 2.4e-07 | 2.4e-07 | 0.991451 |
| u32+scrD512_ctx2048_B32 | 44.70 (39.65–49.61) | 2.64 | 3.65 | 21.05 | 19.65 | 0.468 | 715.9 | 1.588x (1.53–1.70) | 0.969x (0.91–1.05) | 32 / 512 | 92.90 (76.25) | 0.999507 | 1.0e-06 | 1.0e-06 | 0.991451 |
| u32+scrDV512_ctx2048_B32 | 44.13 (40.72–103.68) | 3.17 | 3.34 | 21.54 | 19.55 | 0.479 | 725.1 | 1.609x (0.99–1.78) | 0.982x (0.64–1.03) | 32 / 512 | 92.90 (76.25) | 0.999507 | 1.0e-06 | 1.0e-06 | 0.991451 |
| u32+scrF_ctx2048_B32 | 41.59 (37.87–51.70) | 4.08 | 3.36 | 18.59 | 19.57 | 0.451 | 769.4 | 1.707x (1.56–1.81) | 1.042x (0.94–1.17) | 32 / 1024 | 92.90 (76.25) | 0.999507 | 3.2e-06 | 3.2e-06 | 0.991451 |
| u32+pool_ctx2048_B32 | 70.99 (65.75–83.21) | 5.86 | 3.23 | 48.12 | 19.54 | 0.676 | 450.8 | 1.000x (1.00–1.00) | 0.610x (0.56–0.66) | 160 / 2048 | 92.90 (76.25) | 0.999507 | 0.0e+00 | 0.0e+00 | 0.991451 |
| u32+scrD_ctx8192_B16 | 60.48 (54.20–82.12) | 4.65 | 3.09 | 38.46 | 18.42 | 0.630 | 264.6 | 1.414x (1.26–1.67) | 1.000x (1.00–1.00) | 16 / 1024 | 156.02 (139.38) | 0.999415 | 4.2e-07 | 0.0e+00 | 0.991568 |
| u32+scrDV_ctx8192_B16 | 61.43 (52.52–80.58) | 9.76 | 3.11 | 39.03 | 18.96 | 0.642 | 260.5 | 1.393x (1.28–1.73) | 0.985x (0.95–1.07) | 16 / 1024 | 156.02 (139.38) | 0.999415 | 4.8e-07 | 2.4e-07 | 0.991568 |
| u32+scrD512_ctx8192_B16 | 67.28 (58.53–81.02) | 11.53 | 3.39 | 45.26 | 19.84 | 0.663 | 237.8 | 1.271x (1.14–1.58) | 0.899x (0.84–0.96) | 16 / 512 | 156.02 (139.38) | 0.999415 | 1.1e-06 | 1.1e-06 | 0.991568 |
| u32+scrDV512_ctx8192_B16 | 67.80 (56.62–85.82) | 10.79 | 3.32 | 44.41 | 19.02 | 0.672 | 236.0 | 1.262x (1.13–1.54) | 0.892x (0.80–0.94) | 16 / 512 | 156.02 (139.38) | 0.999415 | 1.1e-06 | 1.1e-06 | 0.991568 |
| u32+scrF_ctx8192_B16 | 60.25 (49.85–76.77) | 7.34 | 3.27 | 37.97 | 18.71 | 0.626 | 265.6 | 1.420x (1.28–1.84) | 1.004x (0.87–1.10) | 16 / 1024 | 156.02 (139.38) | 0.999415 | 2.2e-06 | 2.3e-06 | 0.991568 |
| u32+pool_ctx8192_B16 | 85.55 (76.81–101.45) | 12.04 | 3.25 | 60.67 | 19.25 | 0.732 | 187.0 | 1.000x (1.00–1.00) | 0.707x (0.60–0.79) | 80 / 2048 | 156.02 (139.38) | 0.999415 | 0.0e+00 | 4.2e-07 | 0.991568 |
| u32+scrD_ctx8192_B32 | 86.12 (71.73–91.60) | 7.62 | 3.19 | 64.08 | 18.91 | 0.743 | 371.6 | 1.616x (1.50–1.85) | 1.000x (1.00–1.00) | 32 / 1024 | 295.40 (278.75) | 0.999415 | 4.8e-07 | 0.0e+00 | 0.991568 |
| u32+scrDV_ctx8192_B32 | 86.83 (74.09–96.08) | 7.16 | 3.23 | 63.63 | 20.10 | 0.729 | 368.5 | 1.603x (1.53–1.74) | 0.992x (0.94–1.04) | 32 / 1024 | 295.40 (278.75) | 0.999415 | 4.8e-07 | 2.4e-07 | 0.991568 |
| u32+scrD512_ctx8192_B32 | 97.39 (87.27–122.38) | 13.36 | 3.48 | 75.24 | 19.13 | 0.770 | 328.6 | 1.429x (1.33–1.59) | 0.884x (0.80–1.00) | 32 / 512 | 295.40 (278.75) | 0.999415 | 2.0e-06 | 2.0e-06 | 0.991568 |
| u32+scrDV512_ctx8192_B32 | 99.13 (90.24–119.11) | 13.32 | 3.21 | 77.66 | 18.60 | 0.786 | 322.8 | 1.404x (1.25–1.46) | 0.869x (0.76–0.94) | 32 / 512 | 295.40 (278.75) | 0.999415 | 2.0e-06 | 2.0e-06 | 0.991568 |
| u32+scrF_ctx8192_B32 | 88.94 (76.16–95.74) | 11.60 | 3.34 | 64.29 | 20.23 | 0.728 | 359.8 | 1.565x (1.54–1.78) | 0.968x (0.83–1.14) | 32 / 1024 | 295.40 (278.75) | 0.999415 | 2.2e-06 | 2.3e-06 | 0.991568 |
| u32+pool_ctx8192_B32 | 139.20 (123.83–149.98) | 6.39 | 3.22 | 116.45 | 19.82 | 0.835 | 229.9 | 1.000x (1.00–1.00) | 0.619x (0.54–0.67) | 160 / 2048 | 295.40 (278.75) | 0.999415 | 0.0e+00 | 4.8e-07 | 0.991568 |

| attention only (same q, no weight pass) | scrD ms (min–max) | scrDV ms (min–max) | scrD512 ms (min–max) | scrDV512 ms (min–max) | scrF ms (min–max) | pool ms (min–max) | scrDV/scrD | scrD512/scrD | scrDV512/scrD |
|---|---|---|---|---|---|---|---|---|---|
| ctx2048_B32 | 16.69 (15.64–20.76) | 17.16 (15.38–18.96) | 19.94 (18.56–21.38) | 20.68 (19.42–22.50) | 17.15 (15.65–20.87) | 46.67 (43.78–52.69) | 0.972x | 0.837x | 0.807x |
| ctx8192_B16 | 33.40 (28.92–36.89) | 34.11 (29.79–38.47) | 40.26 (37.32–43.89) | 43.02 (41.25–46.12) | 34.26 (29.84–45.50) | 57.00 (54.52–60.94) | 0.979x | 0.830x | 0.776x |
| ctx8192_B32 | 55.34 (51.68–60.43) | 56.90 (52.36–60.25) | 66.11 (64.17–74.77) | 71.89 (69.17–75.31) | 58.48 (55.79–89.25) | 103.01 (99.83–117.06) | 0.973x | 0.837x | 0.770x |

### Large-q smoke test (SEED 1, q = 2·N(0,1)·SC, B = 32 / ctx 8192, W = 32; kmode, vfold, TK, median of 5, max abs vs 05:55 scrD TK 1024)

    D False 512 69.1 ms  maxabs vs scrD 3.6e-06
    D False 1024 54.9 ms  maxabs vs scrD 0.0e+00
    D True 512 73.4 ms  maxabs vs scrD 3.8e-06
    D True 1024 60.9 ms  maxabs vs scrD 4.8e-07
    F False 512 91.1 ms  maxabs vs scrD 2.1e-05
    F False 1024 57.9 ms  maxabs vs scrD 2.1e-05
    F True 512 86.8 ms  maxabs vs scrD 2.2e-05
    F True 1024 55.4 ms  maxabs vs scrD 2.1e-05

## Verdict

**Inconclusive.** Taken literally, the step bar is met at ctx 8192 / B = 32: scrDV's per-round low vs the pool step is ≥ 1.5 on both seeds (min 1.53) (medians 1.73x / 1.60x), with max abs ≤ 1.1e-06 and the large-q check safe. But neither tested change does what the question hoped. The V-scale fold is 0.97–1.03x of scrD on attention, TK 512 is 0.80–0.90x (**not viable**), and attention stays at 64–69 ms, not ≤ ~50 ms. scrD itself sits at a per-round low of 1.498, so whether either variant clears 1.5 depends on round noise and box load, not on the kernel. The profile is the useful result: about 70% of the tile is decoding K8/V4 into f32, and the matmuls are about a quarter. The next lever is the decode, not the tiling or where the scales are applied.

## This wave's pair files

Pair builders were idle: no quant pair reports reached chat this wave (the 5:30 quant derivation lane is paused), so there are no pair files to link. Previous note: [[2026-11-22_0555_per-worker-preallocated-f32-k-v-score-scratch-for-grouped-k8v4-attention-k8-codes-cast-scales-folded-into-q-k-inside-full-b32-decode-step-vs-alloc-grouped-vs-pool-ctx-2048-8192]].

## Script

sk2.py:

    """Scratch grouped K8V4 attention, 10:55 11/22: sk.py (05:55 11/22) generalized.
    kmode "D" = K dequant into f32 scratch with out= (same math as iq.dq8, bit-identical to 05:55 scrD);
    kmode "F" = K8 codes cast into scratch, two g64 key scales folded into q.K (05:55 scrF).
    vfold True = V scales folded into P: the V4 nibbles (4 x g32) / V8 window codes (2 x g64) are only cast into a
    group-major f32 scratch (bh, ng, L, gsz); P is scaled per group, Ps[g] = P * s_g (bh, ng, G, L), and one batched
    matmul Ps @ codes_g gives (bh, ng, G, gsz), added into acc viewed as (bh, G, ng, gsz). Scale multiplies drop from
    L*128 to L*G*ng per head (8x / 16x fewer).
    prof: optional dict; when given, per-op perf_counter time is accumulated per thread (keys below)."""
    import threading, time
    import numpy as np
    OPS = ("k_deq", "qk", "softmax", "v_take", "v_scale", "pv")
    def make(gs, kmode="D", vfold=False, prof=None):
        iq = gs.iq; KVH, G, HD = iq.KVH, iq.G, iq.HD; LUT16 = iq.LUT16; H2 = HD // 2
        tl = threading.local(); REG = []; lock = threading.Lock(); pc = time.perf_counter
        def scr(bh, TK):
            s = getattr(tl, "s", None)
            if s is None or s["bh"] < bh or s["TK"] < TK:
                s = dict(bh=bh, TK=TK, K=np.empty((bh, TK * HD), np.float32), V=np.empty((bh, TK * HD), np.float32),
                         S=np.empty((bh, G * TK), np.float32), S1=np.empty((bh, G * TK), np.float32), U=np.empty((bh, TK, H2), np.uint16),
                         PS=np.empty((bh, 4 * G * TK), np.float32), O=np.empty((bh, G * HD), np.float32))
                tl.s = s; REG.append(s)
            return s
        def hs_range(st, Q, h0, h1, TK):
            bh = h1 - h0; s = scr(bh, max(TK, iq.WIN)); T = dict.fromkeys(OPS, 0.0) if prof is not None else None
            m = np.full((bh, G, 1), -np.inf, np.float32); l = np.zeros((bh, G, 1), np.float32); acc = np.zeros((bh, G, HD), np.float32)
            Q = np.ascontiguousarray(Q); Q0 = np.ascontiguousarray(Q[:, :, :H2]); Q1 = np.ascontiguousarray(Q[:, :, H2:])
            for a, b, old in gs.tiles(st.n, TK):
                L = b - a; K = s["K"][:bh, :L * HD].reshape(bh, L, HD); Sx = s["S"][:bh, :G * L].reshape(bh, G, L)
                if T is not None: t0 = pc()
                if kmode == "F":
                    np.copyto(K, st.k8[h0:h1, a:b], casting="unsafe"); ks = st.ks8[h0:h1, a:b]
                    if T is not None: t1 = pc(); T["k_deq"] += t1 - t0; t0 = t1
                    S1 = s["S1"][:bh, :G * L].reshape(bh, G, L)
                    np.matmul(Q0, K[:, :, :H2].transpose(0, 2, 1), out=Sx); Sx *= ks[:, None, :, 0]
                    np.matmul(Q1, K[:, :, H2:].transpose(0, 2, 1), out=S1); S1 *= ks[:, None, :, 1]; Sx += S1
                else:
                    np.multiply(st.k8[h0:h1, a:b].reshape(bh, L, 2, 64), st.ks8[h0:h1, a:b][..., None], out=K.reshape(bh, L, 2, 64))
                    if T is not None: t1 = pc(); T["k_deq"] += t1 - t0; t0 = t1
                    np.matmul(Q, K.transpose(0, 2, 1), out=Sx)
                if T is not None: t1 = pc(); T["qk"] += t1 - t0; t0 = t1
                mn = np.maximum(m, Sx.max(2, keepdims=True)); al = np.exp(m - mn); Sx -= mn; np.exp(Sx, out=Sx)
                l *= al; l += Sx.sum(2, keepdims=True); acc *= al
                if T is not None: t1 = pc(); T["softmax"] += t1 - t0; t0 = t1
                if old:
                    U = s["U"][:bh, :L]; np.take(LUT16, st.v4[h0:h1, a:b], out=U); C = U.view(np.int8).reshape(bh, L, 4, 32); sc = st.v4s[h0:h1, a:b]; ng, gz = 4, 32
                else:
                    C = st.v8[h0:h1, a:b].reshape(bh, L, 2, 64); sc = st.vs8[h0:h1, a:b]; ng, gz = 2, 64
                if T is not None: t1 = pc(); T["v_take"] += t1 - t0; t0 = t1
                if vfold:
                    Vc = s["V"][:bh, :L * HD].reshape(bh, ng, L, gz); np.copyto(Vc, C.transpose(0, 2, 1, 3), casting="unsafe")
                    PS = s["PS"][:bh, :ng * G * L].reshape(bh, ng, G, L); np.multiply(Sx[:, None], sc.transpose(0, 2, 1)[:, :, None, :], out=PS)
                    if T is not None: t1 = pc(); T["v_scale"] += t1 - t0; t0 = t1
                    O = s["O"][:bh].reshape(bh, ng, G, gz); np.matmul(PS, Vc, out=O)
                    acc.reshape(bh, G, ng, gz)[...] += O.transpose(0, 2, 1, 3)
                else:
                    V = s["V"][:bh, :L * HD].reshape(bh, L, HD); np.multiply(C, sc[..., None], out=V.reshape(bh, L, ng, gz))
                    if T is not None: t1 = pc(); T["v_scale"] += t1 - t0; t0 = t1
                    acc += np.matmul(Sx, V)
                if T is not None: t1 = pc(); T["pv"] += t1 - t0
                m = mn
            if T is not None:
                with lock:
                    for k, v in T.items(): prof[k] = prof.get(k, 0.0) + v
            return acc / l
        def hsP(st, Qb, W, TK):
            BH = Qb.shape[0]; cuts = sorted(set(round(i * BH / W) for i in range(W + 1)))
            futs = [iq.POOL.submit(hs_range, st, Qb[h0:h1], h0, h1, TK) for h0, h1 in zip(cuts[:-1], cuts[1:])]
            return np.concatenate([f.result() for f in futs], 0)
        def scratch_bytes(): return sum(sum(v.nbytes for v in s.values() if isinstance(v, np.ndarray)) for s in REG)
        return hsP, scratch_bytes, hs_range

prof.py:

    """Per-op profile of the scratch grouped K8V4 tile at B = 32 / ctx 8192 (10:55 11/22).
    pooled: W = 32 tasks on the 8-thread pool, per-op thread time summed over tasks, attributed to wall by share.
    serial: one task (bh = 5 heads) on the main thread, per-op ms per task.
    Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 prof.py out.json"""
    import sys, types, time, json, numpy as np
    sys.path.insert(0, "/workspace/bench1055_1122"); import sk2
    gsrc = open("/workspace/bench1855_1121/gs.py").read().rstrip(); gs = types.ModuleType("gs"); exec(compile(gsrc[:-6], "gs.py", "exec"), gs.__dict__)
    iq = gs.iq; B, n, R, REP = 32, 8192, 5, 3
    st = gs.Stack(B, n)
    for i in range(B): k, v = iq.kvhist(n); st.add(i, k, v)
    Q = (iq.rng.standard_normal((B * iq.KVH, iq.G, iq.HD)) * iq.SC).astype(np.float32)
    VAR = [(f"{km}{'V' if vf else ''}_TK{TK}", km, vf, TK) for km in ("D",) for vf in (False, True) for TK in (1024, 512)] + [("F_TK1024", "F", False, 1024)]
    out = {"seed": iq.SEED}
    for nm, km, vf, TK in VAR:
        P = {}; fP, _, hr = sk2.make(gs, km, vf, P); f0, _, hr0 = sk2.make(gs, km, vf, None)
        f0(st, Q, 32, TK); fP(st, Q, 32, TK)
        wall, wallP, ser, serP = [], [], [], []; P.clear(); SP = {}
        for r in range(R):
            for _ in range(REP):
                t0 = time.perf_counter(); f0(st, Q, 32, TK); wall.append((time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter(); fP(st, Q, 32, TK); wallP.append((time.perf_counter() - t0) * 1e3)
        tot = sum(P.values()); calls = R * REP
        _, _, hs = sk2.make(gs, km, vf, SP); _, _, hs0 = sk2.make(gs, km, vf, None); hs(st, Q[:5], 0, 5, TK); hs0(st, Q[:5], 0, 5, TK); SP.clear()
        for r in range(R * REP):
            t0 = time.perf_counter(); hs0(st, Q[:5], 0, 5, TK); ser.append((time.perf_counter() - t0) * 1e3); hs(st, Q[:5], 0, 5, TK)
        d = dict(wall=float(np.median(wall)), wall_lo=min(wall), wall_hi=max(wall), wallP=float(np.median(wallP)),
                 share={k: v / tot for k, v in P.items()}, thread_ms={k: v * 1e3 / calls for k, v in P.items()},
                 ser=float(np.median(ser)), ser_op={k: v * 1e3 / (R * REP) for k, v in SP.items()})
        d["attrib_ms"] = {k: d["share"][k] * d["wall"] for k in sk2.OPS}
        out[nm] = d
        print(f"{nm:12s} wall {d['wall']:.1f} ({d['wall_lo']:.1f}-{d['wall_hi']:.1f}) instr {d['wallP']:.1f} | " + " ".join(f"{k} {d['attrib_ms'][k]:.1f}({100*d['share'][k]:.0f}%)" for k in sk2.OPS) +
              f" | serial task {d['ser']:.2f} ms: " + " ".join(f"{k} {d['ser_op'][k]:.2f}" for k in sk2.OPS), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)

smoke.py:

    import sys, types, time, numpy as np
    sys.path.insert(0, "/workspace/bench1055_1122"); sys.path.insert(0, "/workspace/bench0555_1122")
    import sk, sk2
    gsrc = open("/workspace/bench1855_1121/gs.py").read().rstrip(); gs = types.ModuleType("gs"); exec(compile(gsrc[:-6], "gs.py", "exec"), gs.__dict__)
    iq = gs.iq; B, n = 32, 8192
    st = gs.Stack(B, n)
    for i in range(B): k, v = iq.kvhist(n); st.add(i, k, v)
    Q = (2 * iq.rng.standard_normal((B * iq.KVH, iq.G, iq.HD)) * iq.SC).astype(np.float32)
    oD = sk.make(gs, False)[0](st, Q, 32, 1024)
    for km, vf in [("D", False), ("D", True), ("F", False), ("F", True)]:
        f = sk2.make(gs, km, vf)[0]
        for TK in (512, 1024):
            o = f(st, Q, 32, TK); t = []
            for _ in range(5): t0 = time.perf_counter(); f(st, Q, 32, TK); t.append(time.perf_counter() - t0)
            print(km, vf, TK, f"{np.median(t)*1e3:.1f} ms  maxabs vs scrD {np.abs(o - oD).max():.1e}")

step.py:

    #!/usr/bin/env python3
    """V-scale fold into P and TK 512 for scratch grouped K8V4 attention inside the full B = 32 bitnetd decode step, 10:55 11/22.
    Copy of step.py (05:55 11/22; itself a copy of 02:55 11/22 step.py) which reuses:
      - weights: base.build + par.prep/par.fmm (shared u32-LUT 2-bit pass, T=256, W=8) and par.prep8/par.fmm8 (int8 control)
        from /workspace/bench1455_1118, exactly as dstep.py (05:55 11/21, commit 0a0aab5);
      - step(): dstep.py's x -> rms -> fused qkv -> attention -> o -> rms -> gate|up -> silu -> down (qkv / attn / rest timed);
      - attention + lean K8V4 Stack + hbP from gs.py (18:55 11/21), which loads iq.py (02:55 11/21) without main().
    sk.py (05:55 11/22) = gs.hb_range with per-worker preallocated f32 K/V/score scratch. New: sk2.py (this wave) = sk.py + V-scale fold into P.
    Variants per (ctx, B), all grouped ones W = clamp(B, 8, 32):
                            u32+scrD     = 2-bit u32-LUT weights + 05:55 scratch grouped, K dequant with out=, TK 1024 (05:55 safe variant)
                            u32+scrDV    = scrD + V scales folded into P (codes only cast, group-major), TK 1024
                            u32+scrD512  = scrD at TK 512 (L2-sized tiles)
                            u32+scrDV512 = scrDV at TK 512
                            u32+scrF     = 05:55 scrF (K scales folded into q.K), TK 1024, reference
                            u32+pool     = 2-bit weights + one-task-per-(stream, KV head) pool attention, TK 2048 (baseline)
    Plus attention-only timing (same Q, no weight pass) of every attention variant per point. int8 weights are still built (REF) but not timed.
    Cache held at fixed n (no append). Usage: OPENBLAS_NUM_THREADS=1 SEED=... python3 step.py out.json"""
    import json, sys, time, types
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    sys.path.insert(0, "/workspace/bench0555_1122"); import sk
    sys.path.insert(0, "/workspace/bench1055_1122"); import sk2
    from base import rms, silu, D
    gsrc = open("/workspace/bench1855_1121/gs.py").read().rstrip()
    assert gsrc.endswith("main()"); gs = types.ModuleType("gs"); exec(compile(gsrc[:-6], "gs.py", "exec"), gs.__dict__)
    hsF, scrb = sk.make(gs, True); hsD, _ = sk.make(gs, False); hsDV, scrb2, _ = sk2.make(gs, "D", True)
    iq = gs.iq; KVH, G, HD, SC = iq.KVH, iq.G, iq.HD, iq.SC
    TU, W = 256, 8
    GRID = [(2048, 32), (8192, 16), (8192, 32)]
    ROUNDS, STEPS, WARM = 5, 6, 2
    base.build(); WB = {"u32": par.prep(), "int8": par.prep8()}
    for nm in base.PW:
        for f in ("Wd", "Wt", "wm", "ws", "wm8", "ws8", "PT", "q8"): base.PW[nm].pop(f, None)
    MM = {"u32": par.fmm, "int8": par.fmm8}
    def step(mm, x, af):   # verbatim from dstep.py (05:55 11/21)
        t0 = time.perf_counter(); q = mm("qkv", rms(x), TU, W)[0]
        t1 = time.perf_counter(); att = af(q.reshape(x.shape[0], KVH, G, HD) * SC).reshape(x.shape[0], D)
        t2 = time.perf_counter(); x2 = x + mm("o", att, TU, W)[0]
        g, u = mm("gu", rms(x2), TU, W); y = x2 + mm("down", (silu(g) * u).astype(np.float32), TU, W)[0]
        t3 = time.perf_counter(); return y, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)
    cos = gs.cos
    def main():
        rng = iq.rng; out = {"seed": iq.SEED, "weight_bytes": WB, "cfg": {}, "attn_only": {}}
        print(f"seed {iq.SEED} | weights u32 {WB['u32']/2**20:.2f} MiB int8 {WB['int8']/2**20:.2f} MiB | rounds {ROUNDS} x steps {STEPS} (+{WARM} warm)", flush=True)
        Bmax = max(b for _, b in GRID)
        XS = rng.standard_normal((STEPS, Bmax, D)).astype(np.float32); x0 = XS[0]
        Q0 = {wk: MM[wk]("qkv", rms(x0), TU, W)[0].reshape(Bmax, KVH, G, HD) * SC for wk in MM}
        for n in sorted({g[0] for g in GRID}):
            Bm = max(B for m, B in GRID if m == n); st = gs.Stack(Bm, n)
            REF = {wk: np.empty((Bm * KVH, G, HD), np.float32) for wk in MM}
            for i in range(Bm):
                k, v = iq.kvhist(n); st.add(i, k, v)
                for wk in MM:
                    for h in range(KVH): REF[wk][i * KVH + h] = iq.sm(Q0[wk][i, h], k[h], v[h])
                del k, v
            for B in [B for m, B in GRID if m == n]:
                kvs = st.kvs[:B]; Wg = min(32, max(8, B)); xb = x0[:B]
                AF = {"scrD": lambda qq: hsD(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), Wg, 1024),
                      "scrDV": lambda qq: hsDV(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), Wg, 1024),
                      "scrD512": lambda qq: hsD(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), Wg, 512),
                      "scrDV512": lambda qq: hsDV(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), Wg, 512),
                      "scrF": lambda qq: hsF(st, np.ascontiguousarray(qq.reshape(-1, G, HD)), Wg, 1024),
                      "pool": lambda qq: iq.attn(kvs, qq, "k8v4", 2048)}
                RUNS = [("u32", "scrD"), ("u32", "scrDV"), ("u32", "scrD512"), ("u32", "scrDV512"), ("u32", "scrF"), ("u32", "pool")]
                kvmib = B * kvs[0].bytes("k8v4") / 2**20; cfgs = {}
                for wk, a in RUNS:
                    ref = REF[wk][:B * KVH]
                    yref = step(MM[wk], xb, lambda qq: ref)[0]
                    ao = np.asarray(AF[a](Q0[wk][:B])).reshape(B * KVH, G, HD)
                    pb = np.asarray(AF["pool"](Q0[wk][:B])).reshape(B * KVH, G, HD)
                    y = step(MM[wk], xb, AF[a])[0]
                    cfgs[(wk, a)] = dict(t=[], lc=min(cos(y[s] - xb[s], yref[s] - xb[s]) for s in range(B)),
                                         amab=float(np.abs(ao - pb).max()), amabg=float(np.abs(ao - np.asarray(AF["scrD"](Q0[wk][:B])).reshape(B * KVH, G, HD)).max()), acos=min(cos(ao[i], ref[i]) for i in range(B * KVH)))
                order = list(cfgs)
                for rd in range(ROUNDS):
                    for kk in order[rd % len(order):] + order[:rd % len(order)]:
                        c = cfgs[kk]
                        for i in range(WARM + STEPS):
                            _, t = step(MM[kk[0]], XS[i % STEPS, :B], AF[kk[1]])
                            if i >= WARM: c["t"].append([v * 1e3 for v in t])
                ao_t = {a: [] for a in ("scrD", "scrDV", "scrD512", "scrDV512", "scrF", "pool")}; qa = Q0["u32"][:B]; ao_o = list(ao_t)
                for a in ao_o: AF[a](qa)
                for rd in range(ROUNDS):
                    for a in ao_o[rd % len(ao_o):] + ao_o[:rd % len(ao_o)]:
                        for _ in range(3):
                            t0 = time.perf_counter(); AF[a](qa); ao_t[a].append((time.perf_counter() - t0) * 1e3)
                AO = {a: dict(ms=float(np.median(v)), lo=float(min(v)), hi=float(max(v)), rounds=[float(np.median(v[r * 3:(r + 1) * 3])) for r in range(ROUNDS)]) for a, v in ao_t.items()}
                out["attn_only"][f"ctx{n}_B{B}"] = AO
                print(f"attn-only ctx{n} B{B}: " + " | ".join(f"{a} {d['ms']:.2f} ({d['lo']:.2f}-{d['hi']:.2f})" for a, d in AO.items()) + " | vs scrD " + " ".join(f"{a} {AO['scrD']['ms']/AO[a]['ms']:.3f}x" for a in AO), flush=True)
                rmed = lambda kk: [float(np.median(np.array(cfgs[kk]["t"])[r * STEPS:(r + 1) * STEPS, 3])) for r in range(ROUNDS)]
                REFS = {"pool": ("u32", "pool"), "scrD": ("u32", "scrD")}
                ref_rounds = {nm: rmed(kk) for nm, kk in REFS.items()}
                for (wk, a), c in cfgs.items():
                    T = np.array(c["t"]); med = [float(np.median(T[:, j])) for j in range(4)]; rr = rmed((wk, a))
                    key = f"{wk}+{a}_ctx{n}_B{B}"
                    d = dict(B=B, ctx=n, W=B * KVH if a == "pool" else Wg, TK=2048 if a == "pool" else (512 if a.endswith("512") else 1024),
                             qkv=med[0], attn=med[1], rest=med[2], step=med[3], lo=float(T[:, 3].min()), hi=float(T[:, 3].max()),
                             iqr=float(np.percentile(T[:, 3], 75) - np.percentile(T[:, 3], 25)), share=float(np.median(T[:, 1] / T[:, 3])),
                             tok_s=B * 1e3 / med[3], kv_mib=kvmib, w_mib=WB[wk] / 2**20, tot_mib=WB[wk] / 2**20 + kvmib,
                             layer_cos=c["lc"], attn_mab_pool=c["amab"], attn_mab_scrD=c["amabg"], scratch_mib=(scrb() + scrb2()) / 2**20, attn_cos_f32=c["acos"], nsteps=len(T))
                    for nm, R in ref_rounds.items():
                        sp = [x / y for x, y in zip(R, rr)]
                        d[f"vs_{nm}"] = float(np.median(np.array(cfgs[REFS[nm]]["t"])[:, 3])) / med[3]
                        d[f"vs_{nm}_lo"], d[f"vs_{nm}_hi"] = min(sp), max(sp)
                    out["cfg"][key] = d
                    print(f"{key:24s} {d['step']:7.2f} ({d['lo']:.2f}-{d['hi']:.2f}) iqr {d['iqr']:5.2f} | qkv {d['qkv']:5.2f} attn {d['attn']:6.2f} rest {d['rest']:5.2f} share {d['share']:.3f} | {d['tok_s']:6.1f} tok/s | vs pool {d['vs_pool']:.3f}x ({d['vs_pool_lo']:.2f}-{d['vs_pool_hi']:.2f}) vs scrD {d['vs_scrD']:.3f}x ({d['vs_scrD_lo']:.2f}-{d['vs_scrD_hi']:.2f}) | tot {d['tot_mib']:.2f} MiB (KV {kvmib:.2f}) | lcos {d['layer_cos']:.6f} amab {d['attn_mab_pool']:.1e} amabD {d['attn_mab_scrD']:.1e} acos {d['attn_cos_f32']:.6f}", flush=True)
            del st, REF
        out["scratch_bytes"] = scrb(); out["scratch2_bytes"] = scrb2(); json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Next question

About 70% of the scratch tile is decode: K dequant ~24%, V4 nibble take ~23%, V scale ~22%. The q·K / P·V matmuls are only ~24%. Can the decode be cut in half without touching error? Replace the u16-LUT take + int8 view + cast with one np.take on a 256 × 2 f32 pair LUT straight into the group-major f32 V scratch (scales already folded into P, so the V path is a single gather). For K, gather from a 256-entry f32 LUT of int8 values into scratch and scale with out= (exact, unlike scrF). Does decode drop below 50% of the tile, attention to ≤ ~50 ms at B = 32 / ctx 8192, and the step to a per-round low ≥ 1.5x vs pool at max abs ≤ 1e-5 (including the large-q check)?

close: 0eeec44a17e5c1c1
