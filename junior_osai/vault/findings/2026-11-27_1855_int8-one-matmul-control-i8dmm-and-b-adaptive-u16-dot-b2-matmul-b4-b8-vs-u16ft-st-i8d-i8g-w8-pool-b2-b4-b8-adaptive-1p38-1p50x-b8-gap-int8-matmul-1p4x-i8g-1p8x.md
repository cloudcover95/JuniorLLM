---
local: true
date: 2026-11-27
wave: "2026-11-27 18:55 slot (fired 18:48 MT, ran ~18:49–18:56 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (fair int8 one-matmul control + B-adaptive u16 on the W 8 decode layer, full bitnetd layer, ctx-free, B 2 / 4 / 8. The B-adaptive u16 (per-b dot at B <= 2, down-only 256x1728 sub-tile one-matmul at B >= 4) clears the >= 1.3x bar vs u16ft_st only at B 8 (1.38–1.50x; B 4 1.08–1.12x; B 2 1.01–1.04x). But int8 gains more from one matmul per tile: i8dmm_gd2 is 1.67–1.69x faster than i8d_gd2 at B 8, so the adaptive u16 is 1.41–1.47x behind the best int8 one-matmul control at B 8 (1.38–1.43x at B 4, 1.28–1.38x at B 2) and 1.76–1.86x behind the existing weight-major int8 matmul i8g_gd2 at B 8. The within-1.3x-of-i8dmm bar is missed at B 4 and B 8; the int8 gap reopens. Error <= 1.25e-6 vs the float64 ternary reference (cos 1.0); trits exact. Memory unchanged: 17.46 vs 69.56 MB int8 (3.98x, the known weight-memory win). No new 1.5x speed or memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, uint16, eight-trit-pack, base-3, 2-bit, np-take, fused-take, static-split, k-blocking, np-matmul, b-adaptive, int8-control, weight-major, int8, numpy, not-viable]
---

# Fair int8 one-matmul control + B-adaptive u16: u16 gains 1.38–1.50x at B 8, but int8 gains 1.67x and the gap reopens to 1.4x (1.8x vs i8g)

The 14:55 note ([[2026-11-27_1455_down-only-k-blocked-u16ft-fixed-1p77mb-subtile-256x1728-512x864-1024x432-per-b-dot-vs-one-matmul-vs-u16ft-st-and-int8d-w8-pool-b2-b4-best-1p09-1p15x-b4-1p0x-b2-gap-int8d-1p2x-b4]]) found that one np.matmul per down-site 1.77 MB sub-tile beats u16ft_st by 1.13–1.15x at B 4 and ties at B 2, bringing u16 within 1.20x of int8d at B 4. int8d had not been given the same one-matmul treatment. It asked for two tests. First, a fair int8 control with one matmul per tile (i8dmm), unsplit and down-only K-blocked. Second, a B-adaptive u16 kernel (per-b dot at B <= 2, sub-tile matmul at B >= 4), measured at B 4 and B 8. The bar: >= 1.3x vs u16ft_st and within 1.3x of i8dmm.

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 2, B 4 and B 8: does a B-adaptive u16 kernel reach >= 1.3x over u16ft_st and stay within 1.3x of a fair int8 one-matmul-per-tile control? Or does int8 gain as much from matmul and reopen the gap?

## Method

- Box: 8 CPUs (Xeon, L2 2 MiB per core), numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network, python3 + numpy only. The box is shared with other agents (load avg 0.86 / 2.20 / 3.26 at the start of runs 1 / 2 / 3), so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1455_1127/v16m.py exec'd without main() (→ v16k.py → v16t.py → v16d.py → v16.py: same weights, P16 8-trit uint16 pack with exact trit round-trip asserts, float64 references, st / dy / gd2 schedules, r_u16ft kernel, profiling hooks). The new kernels size their (B, m) matmul scratch for B 8.
- u16 kernels. Each uses one fused u64 LUT take per static worker range, as u16ft does.
  - **u16x256x1728mm_st / u16x512x864mm_st**: the in-2560 sites (qkv / o / gu) use 256-row tiles with one np.matmul per tile. The down site is K-blocked into an (R, Kc) f32 sub-tile (1,769,472 B) with one np.matmul per sub-tile: tmp(B, m) = X[:, k0:k1] @ sub.T, then y[:, rows] = / += tmp.
  - **u16ad_st** (B-adaptive): r_u16ft (per-b np.dot, unsplit) when B <= 2, the u16x256x1728mm kernel when B >= 4. At B 4 and B 8 it runs the same code as u16x256x1728mm_st, so the gap between those two rows shows the run noise.
- int8 controls:
  - **i8dmm_gd2 / i8dmm_st**: cast each int8 tile to f32, then one np.matmul per tile (x @ tile.T → y[:, rows]). All sites unsplit, T 128 gd2 or T 256 static.
  - **i8dmk_st / i8dmk_gd2**: the same, but the down site is K-blocked at Kc 1728, with the same structure as u16x256x1728mm.
  - Baselines: **u16ft_st**, **i8d_dy / i8d_gd2** (per-row np.dot, T 128), and **i8g_gd2** (the existing weight-major cast + tile @ x.T matmul, T 128).
- Error: max abs error and cos vs the float64 ternary (u16) or int8 reference. Each run also checks that instrumented and plain output are identical, and records max abs diff vs u16ft_st / i8d_gd2.
- Timing: per B, 7 rounds × 5 reps of the plain layer, config order rotated per round. Reported as median of round medians, round min–max and per-round ratios. Then 15 instrumented layers per config give stage worker-ms (wall/cpu summed over workers) and per-site span. Three runs, XSEED 1/2/3.
- Command: `cd /workspace/bench1855_1127 && for s in 1 2 3; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 v16x.py run$s.json > run$s.txt; done; python3 agg.py`

## Results

Below: layer ms, ratios (a/b > 1 means b is faster), stage worker-ms (wall/cpu, runs 1, 2, 3; int8 kernels have no take stage) and error. Ratio rows named best_* pick the fastest config of that group within each run.

Bytes: weights u16 17,458,176 B vs int8 69,559,296 B (3.98x); LUT u64 52,488 B; down f32 sub-tile 1,769,472 B. Load avg at run start: 0.86, 2.20, 3.26.

| B | config | layer ms, median of round medians [round min–max], runs 1 / 2 / 3 |
|---|---|---|
| 2 | u16ft_st | 10.16 [8.85–13.20], 9.81 [9.42–13.70], 11.67 [11.19–12.73] |
| 2 | u16x256x1728mm_st | 13.63 [12.41–15.19], 13.09 [12.15–14.68], 14.22 [13.88–15.29] |
| 2 | u16x512x864mm_st | 12.87 [11.93–15.44], 12.59 [11.93–13.47], 14.66 [13.13–15.38] |
| 2 | u16ad_st | 10.05 [8.95–13.03], 9.67 [9.13–10.34], 11.23 [10.25–12.21] |
| 2 | i8d_dy | 7.63 [6.97–8.49], 7.45 [7.18–7.61], 8.23 [8.11–8.95] |
| 2 | i8d_gd2 | 6.62 [6.39–8.22], 6.85 [6.66–7.39], 7.46 [7.05–8.11] |
| 2 | i8g_gd2 | 6.69 [6.58–7.74], 6.97 [6.62–7.41], 7.61 [7.17–9.25] |
| 2 | i8dmm_gd2 | 7.37 [7.14–8.21], 7.82 [7.57–8.18], 8.84 [7.93–9.21] |
| 2 | i8dmm_st | 8.07 [7.81–8.81], 8.11 [7.79–9.28], 9.70 [8.50–11.83] |
| 2 | i8dmk_st | 7.28 [6.91–9.10], 7.58 [7.05–7.72], 8.81 [7.99–10.99] |
| 2 | i8dmk_gd2 | 8.04 [7.49–10.05], 7.99 [7.59–8.28], 8.79 [8.22–10.46] |
| 4 | u16ft_st | 12.79 [12.18–17.90], 14.82 [12.92–15.81], 14.36 [14.16–14.80] |
| 4 | u16x256x1728mm_st | 11.73 [11.23–14.90], 14.26 [12.36–15.33], 12.90 [11.89–14.44] |
| 4 | u16x512x864mm_st | 11.87 [11.37–15.16], 14.11 [11.84–15.13], 13.36 [12.46–13.90] |
| 4 | u16ad_st | 11.88 [11.21–14.87], 13.36 [11.09–15.93], 12.89 [12.25–13.92] |
| 4 | i8d_dy | 10.44 [10.08–10.75], 12.78 [10.85–13.90], 11.55 [11.36–11.96] |
| 4 | i8d_gd2 | 9.49 [9.33–9.73], 10.71 [10.13–11.52], 10.28 [10.02–11.58] |
| 4 | i8g_gd2 | 8.98 [8.46–9.50], 9.66 [9.34–10.31], 9.07 [8.63–9.58] |
| 4 | i8dmm_gd2 | 9.73 [9.33–10.51], 11.42 [10.89–12.61], 10.41 [9.61–11.36] |
| 4 | i8dmm_st | 8.66 [7.98–9.40], 9.72 [8.80–10.43], 9.04 [8.53–10.10] |
| 4 | i8dmk_st | 8.86 [8.02–9.74], 10.20 [9.41–11.25], 9.23 [8.75–10.94] |
| 4 | i8dmk_gd2 | 8.38 [7.96–9.23], 10.16 [9.44–11.06], 9.23 [9.00–9.73] |
| 8 | u16ft_st | 18.55 [16.30–19.88], 20.34 [19.26–21.72], 19.91 [19.07–21.81] |
| 8 | u16x256x1728mm_st | 13.46 [11.63–14.20], 14.00 [13.40–15.63], 13.38 [12.77–14.93] |
| 8 | u16x512x864mm_st | 13.90 [12.07–14.93], 15.14 [14.04–15.50], 13.81 [13.03–14.67] |
| 8 | u16ad_st | 12.39 [11.83–14.17], 14.70 [13.84–15.96], 13.50 [13.28–14.14] |
| 8 | i8d_dy | 16.58 [16.01–17.06], 19.56 [18.83–21.19], 18.32 [17.98–19.07] |
| 8 | i8d_gd2 | 14.88 [14.22–15.57], 16.78 [16.56–18.41], 15.80 [15.27–16.09] |
| 8 | i8g_gd2 | 7.03 [6.56–8.56], 7.89 [7.54–8.20], 7.33 [7.05–8.42] |
| 8 | i8dmm_gd2 | 8.82 [8.52–9.41], 10.03 [9.79–10.28], 9.38 [9.07–9.97] |
| 8 | i8dmm_st | 9.79 [8.97–11.05], 11.02 [10.44–11.61], 10.19 [9.67–10.63] |
| 8 | i8dmk_st | 10.02 [9.75–10.80], 10.88 [9.98–12.30], 10.08 [9.26–10.51] |
| 8 | i8dmk_gd2 | 9.26 [8.69–9.52], 10.06 [9.69–10.67], 9.49 [9.16–10.11] |

| B | ratio (a/b, >1 = b faster) | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 2 | u16ft/u16x256x1728mm_st | 0.745x [0.649–0.936] (u16ft_st/u16x256x1728mm_st) | 0.749x [0.692–0.933] (u16ft_st/u16x256x1728mm_st) | 0.821x [0.750–0.905] (u16ft_st/u16x256x1728mm_st) |
| 2 | u16ft/u16x512x864mm_st | 0.790x [0.591–1.006] (u16ft_st/u16x512x864mm_st) | 0.779x [0.728–1.051] (u16ft_st/u16x512x864mm_st) | 0.796x [0.763–0.889] (u16ft_st/u16x512x864mm_st) |
| 2 | u16ft/u16ad_st | 1.011x [0.755–1.409] (u16ft_st/u16ad_st) | 1.015x [0.929–1.417] (u16ft_st/u16ad_st) | 1.040x [0.994–1.139] (u16ft_st/u16ad_st) |
| 2 | u16ad/i8d_gd2 | 1.519x [1.352–1.841] (u16ad_st/i8d_gd2) | 1.411x [1.309–1.538] (u16ad_st/i8d_gd2) | 1.505x [1.427–1.732] (u16ad_st/i8d_gd2) |
| 2 | u16ad/best_i8old | 1.519x [1.352–1.841] (u16ad_st/i8d_gd2) | 1.411x [1.309–1.538] (u16ad_st/i8d_gd2) | 1.505x [1.427–1.732] (u16ad_st/i8d_gd2) |
| 2 | u16ad/best_i8mm | 1.380x [1.231–1.653] (u16ad_st/i8dmk_st) | 1.275x [1.239–1.374] (u16ad_st/i8dmk_st) | 1.277x [0.980–1.358] (u16ad_st/i8dmk_gd2) |
| 2 | best_u16/best_i8mm | 1.380x [1.231–1.653] (u16ad_st/i8dmk_st) | 1.275x [1.239–1.374] (u16ad_st/i8dmk_st) | 1.277x [0.980–1.358] (u16ad_st/i8dmk_gd2) |
| 2 | best_i8old/best_i8mm | 0.908x [0.845–0.946] (i8d_gd2/i8dmk_st) | 0.904x [0.887–0.969] (i8d_gd2/i8dmk_st) | 0.848x [0.684–0.890] (i8d_gd2/i8dmk_gd2) |
| 2 | u16ft/best_i8old | 1.536x [1.295–2.054] (u16ft_st/i8d_gd2) | 1.432x [1.374–1.854] (u16ft_st/i8d_gd2) | 1.565x [1.505–1.725] (u16ft_st/i8d_gd2) |
| 2 | i8d_gd2/i8dmm_gd2 | 0.898x [0.833–1.044] (i8d_gd2/i8dmm_gd2) | 0.876x [0.818–0.962] (i8d_gd2/i8dmm_gd2) | 0.844x [0.771–0.940] (i8d_gd2/i8dmm_gd2) |
| 2 | i8d_gd2/i8dmm_st | 0.820x [0.776–1.022] (i8d_gd2/i8dmm_st) | 0.844x [0.739–0.877] (i8d_gd2/i8dmm_st) | 0.769x [0.604–0.899] (i8d_gd2/i8dmm_st) |
| 2 | i8d_gd2/i8dmk_st | 0.908x [0.845–0.946] (i8d_gd2/i8dmk_st) | 0.904x [0.887–0.969] (i8d_gd2/i8dmk_st) | 0.847x [0.651–0.953] (i8d_gd2/i8dmk_st) |
| 2 | i8d_gd2/i8dmk_gd2 | 0.823x [0.816–0.876] (i8d_gd2/i8dmk_gd2) | 0.857x [0.828–0.942] (i8d_gd2/i8dmk_gd2) | 0.848x [0.684–0.890] (i8d_gd2/i8dmk_gd2) |
| 4 | u16ft/u16x256x1728mm_st | 1.090x [1.031–1.202] (u16ft_st/u16x256x1728mm_st) | 1.039x [0.927–1.114] (u16ft_st/u16x256x1728mm_st) | 1.113x [1.011–1.243] (u16ft_st/u16x256x1728mm_st) |
| 4 | u16ft/u16x512x864mm_st | 1.078x [0.962–1.181] (u16ft_st/u16x512x864mm_st) | 1.050x [0.937–1.133] (u16ft_st/u16x512x864mm_st) | 1.075x [1.018–1.146] (u16ft_st/u16x512x864mm_st) |
| 4 | u16ft/u16ad_st | 1.077x [1.026–1.204] (u16ft_st/u16ad_st) | 1.109x [0.930–1.188] (u16ft_st/u16ad_st) | 1.115x [1.031–1.188] (u16ft_st/u16ad_st) |
| 4 | u16ad/i8d_gd2 | 1.251x [1.200–1.557] (u16ad_st/i8d_gd2) | 1.248x [1.095–1.383] (u16ad_st/i8d_gd2) | 1.254x [1.177–1.318] (u16ad_st/i8d_gd2) |
| 4 | u16ad/best_i8old | 1.322x [1.250–1.566] (u16ad_st/i8g_gd2) | 1.382x [1.188–1.648] (u16ad_st/i8g_gd2) | 1.421x [1.351–1.482] (u16ad_st/i8g_gd2) |
| 4 | u16ad/best_i8mm | 1.418x [1.278–1.611] (u16ad_st/i8dmk_gd2) | 1.375x [1.064–1.639] (u16ad_st/i8dmm_st) | 1.426x [1.256–1.548] (u16ad_st/i8dmm_st) |
| 4 | best_u16/best_i8mm | 1.400x [1.229–1.614] (u16x256x1728mm_st/i8dmk_gd2) | 1.375x [1.064–1.639] (u16ad_st/i8dmm_st) | 1.426x [1.256–1.548] (u16ad_st/i8dmm_st) |
| 4 | best_i8old/best_i8mm | 1.072x [0.994–1.097] (i8g_gd2/i8dmk_gd2) | 0.994x [0.896–1.062] (i8g_gd2/i8dmm_st) | 1.003x [0.867–1.123] (i8g_gd2/i8dmm_st) |
| 4 | u16ft/best_i8old | 1.424x [1.387–1.885] (u16ft_st/i8g_gd2) | 1.533x [1.352–1.690] (u16ft_st/i8g_gd2) | 1.584x [1.502–1.715] (u16ft_st/i8g_gd2) |
| 4 | i8d_gd2/i8dmm_gd2 | 0.975x [0.926–1.019] (i8d_gd2/i8dmm_gd2) | 0.937x [0.820–1.008] (i8d_gd2/i8dmm_gd2) | 0.988x [0.894–1.120] (i8d_gd2/i8dmm_gd2) |
| 4 | i8d_gd2/i8dmm_st | 1.096x [1.029–1.197] (i8d_gd2/i8dmm_st) | 1.102x [0.971–1.216] (i8d_gd2/i8dmm_st) | 1.137x [1.038–1.251] (i8d_gd2/i8dmm_st) |
| 4 | i8d_gd2/i8dmk_st | 1.072x [0.993–1.176] (i8d_gd2/i8dmk_st) | 1.050x [0.950–1.171] (i8d_gd2/i8dmk_st) | 1.113x [0.928–1.323] (i8d_gd2/i8dmk_st) |
| 4 | i8d_gd2/i8dmk_gd2 | 1.133x [1.035–1.192] (i8d_gd2/i8dmk_gd2) | 1.053x [0.966–1.133] (i8d_gd2/i8dmk_gd2) | 1.113x [1.030–1.254] (i8d_gd2/i8dmk_gd2) |
| 8 | u16ft/u16x256x1728mm_st | 1.378x [1.160–1.617] (u16ft_st/u16x256x1728mm_st) | 1.453x [1.302–1.600] (u16ft_st/u16x256x1728mm_st) | 1.488x [1.416–1.643] (u16ft_st/u16x256x1728mm_st) |
| 8 | u16ft/u16x512x864mm_st | 1.335x [1.172–1.602] (u16ft_st/u16x512x864mm_st) | 1.343x [1.278–1.479] (u16ft_st/u16x512x864mm_st) | 1.441x [1.329–1.673] (u16ft_st/u16x512x864mm_st) |
| 8 | u16ft/u16ad_st | 1.497x [1.312–1.645] (u16ft_st/u16ad_st) | 1.384x [1.235–1.500] (u16ft_st/u16ad_st) | 1.475x [1.426–1.606] (u16ft_st/u16ad_st) |
| 8 | u16ad/i8d_gd2 | 0.833x [0.805–0.910] (u16ad_st/i8d_gd2) | 0.876x [0.798–0.952] (u16ad_st/i8d_gd2) | 0.854x [0.829–0.884] (u16ad_st/i8d_gd2) |
| 8 | u16ad/best_i8old | 1.761x [1.450–1.931] (u16ad_st/i8g_gd2) | 1.864x [1.767–2.082] (u16ad_st/i8g_gd2) | 1.842x [1.577–1.924] (u16ad_st/i8g_gd2) |
| 8 | u16ad/best_i8mm | 1.405x [1.319–1.604] (u16ad_st/i8dmm_gd2) | 1.466x [1.380–1.630] (u16ad_st/i8dmm_gd2) | 1.439x [1.342–1.539] (u16ad_st/i8dmm_gd2) |
| 8 | best_u16/best_i8mm | 1.405x [1.319–1.604] (u16ad_st/i8dmm_gd2) | 1.396x [1.321–1.596] (u16x256x1728mm_st/i8dmm_gd2) | 1.426x [1.342–1.625] (u16x256x1728mm_st/i8dmm_gd2) |
| 8 | best_i8old/best_i8mm | 0.798x [0.759–0.910] (i8g_gd2/i8dmm_gd2) | 0.787x [0.733–0.837] (i8g_gd2/i8dmm_gd2) | 0.781x [0.724–0.898] (i8g_gd2/i8dmm_gd2) |
| 8 | u16ft/best_i8old | 2.636x [2.166–2.866] (u16ft_st/i8g_gd2) | 2.579x [2.361–2.651] (u16ft_st/i8g_gd2) | 2.717x [2.385–3.089] (u16ft_st/i8g_gd2) |
| 8 | i8d_gd2/i8dmm_gd2 | 1.686x [1.580–1.805] (i8d_gd2/i8dmm_gd2) | 1.673x [1.632–1.815] (i8d_gd2/i8dmm_gd2) | 1.685x [1.565–1.746] (i8d_gd2/i8dmm_gd2) |
| 8 | i8d_gd2/i8dmm_st | 1.519x [1.287–1.610] (i8d_gd2/i8dmm_st) | 1.524x [1.446–1.757] (i8d_gd2/i8dmm_st) | 1.551x [1.468–1.659] (i8d_gd2/i8dmm_st) |
| 8 | i8d_gd2/i8dmk_st | 1.484x [1.395–1.554] (i8d_gd2/i8dmk_st) | 1.543x [1.364–1.846] (i8d_gd2/i8dmk_st) | 1.568x [1.531–1.733] (i8d_gd2/i8dmk_st) |
| 8 | i8d_gd2/i8dmk_gd2 | 1.607x [1.518–1.770] (i8d_gd2/i8dmk_gd2) | 1.668x [1.580–1.873] (i8d_gd2/i8dmk_gd2) | 1.664x [1.592–1.752] (i8d_gd2/i8dmk_gd2) |

| B | config | take | cast | dot | task | span |
|---|---|---|---|---|---|---|
| 2 | u16ft_st | 21.9/18.4, 22.7/18.6, 24.0/19.3 | 14.7/12.5, 14.8/12.5, 16.0/13.0 | 22.4/17.3, 22.5/17.4, 23.3/17.6 | 59.9/49.2, 61.2/49.8, 64.9/50.9 | 9.79, 10.00, 10.87 |
| 2 | u16x256x1728mm_st | 22.0/18.0, 23.0/18.5, 24.3/18.8 | 26.8/12.3, 26.2/12.4, 27.1/13.2 | 27.4/20.5, 27.9/20.4, 29.9/21.1 | 78.8/52.7, 79.1/53.1, 83.6/54.7 | 12.87, 13.06, 14.39 |
| 2 | u16x512x864mm_st | 21.7/17.7, 21.5/17.9, 24.8/18.3 | 24.7/11.9, 24.9/12.0, 26.4/13.4 | 28.5/20.0, 29.1/19.7, 30.2/21.1 | 76.4/51.4, 76.9/51.3, 83.7/54.7 | 12.33, 12.55, 14.23 |
| 2 | u16ad_st | 22.2/18.9, 21.7/18.3, 23.7/19.2 | 14.8/12.5, 15.4/12.7, 16.8/13.7 | 22.3/17.6, 23.1/17.6, 25.1/18.5 | 60.4/50.1, 61.5/49.4, 67.7/52.5 | 10.21, 9.83, 11.18 |
| 2 | i8dmm_gd2 | –, –, – | 23.1/15.2, 24.1/15.6, 26.5/15.2 | 27.3/23.2, 27.0/22.5, 29.5/23.8 | 52.3/40.4, 53.0/39.5, 57.5/40.6 | 7.92, 7.90, 8.80 |
| 2 | i8dmm_st | –, –, – | 14.8/12.8, 14.1/12.6, 15.4/12.9 | 26.3/25.2, 26.0/24.5, 27.7/25.4 | 42.3/39.0, 41.0/38.3, 44.1/39.6 | 8.28, 7.59, 8.84 |
| 2 | i8dmk_st | –, –, – | 15.6/12.6, 14.5/12.3, 16.6/12.8 | 23.9/21.3, 23.1/20.5, 25.3/22.1 | 41.6/35.2, 38.9/33.9, 43.3/36.1 | 7.83, 7.26, 8.84 |
| 2 | i8dmk_gd2 | –, –, – | 27.9/12.6, 27.7/12.4, 32.1/13.4 | 25.7/19.4, 25.6/19.4, 28.6/20.8 | 55.0/33.7, 55.3/33.6, 62.2/36.0 | 8.00, 8.04, 9.17 |
| 4 | u16ft_st | 22.5/19.0, 25.0/19.7, 23.8/19.3 | 14.9/12.6, 16.7/13.5, 16.2/12.9 | 40.9/33.1, 45.8/34.7, 44.7/33.9 | 79.3/66.1, 89.3/69.5, 86.4/67.0 | 12.01, 14.34, 14.03 |
| 4 | u16x256x1728mm_st | 22.8/18.5, 25.1/19.8, 23.9/18.0 | 19.7/12.5, 23.0/13.4, 21.5/12.7 | 30.6/26.3, 33.0/28.0, 30.7/25.5 | 74.9/59.0, 84.2/63.5, 77.5/58.1 | 12.17, 13.69, 12.83 |
| 4 | u16x512x864mm_st | 21.7/18.5, 24.5/19.4, 23.8/19.0 | 18.4/12.4, 21.0/13.6, 22.1/13.6 | 32.7/27.5, 36.2/29.2, 35.8/28.8 | 75.8/59.8, 83.5/63.9, 84.4/63.4 | 11.90, 13.87, 13.53 |
| 4 | u16ad_st | 22.0/18.6, 24.0/19.3, 24.0/18.7 | 20.4/12.5, 22.5/13.0, 22.4/12.9 | 30.4/26.5, 32.2/27.1, 31.4/26.1 | 74.7/59.0, 81.5/61.0, 80.4/59.5 | 11.40, 13.16, 13.38 |
| 4 | i8dmm_gd2 | –, –, – | 29.8/15.1, 34.1/16.2, 31.2/15.6 | 30.6/25.1, 34.0/26.2, 33.1/24.4 | 61.8/41.2, 69.0/43.0, 66.1/41.4 | 9.66, 10.69, 10.31 |
| 4 | i8dmm_st | –, –, – | 14.6/12.9, 15.8/12.8, 15.3/12.8 | 27.2/26.1, 29.1/26.0, 28.2/25.6 | 43.1/40.0, 46.1/39.8, 44.8/39.8 | 8.10, 8.93, 9.02 |
| 4 | i8dmk_st | –, –, – | 15.1/13.1, 16.9/12.9, 16.1/12.7 | 29.1/28.2, 30.0/27.6, 29.6/27.2 | 45.4/42.4, 48.9/41.8, 47.4/41.1 | 8.53, 9.82, 9.27 |
| 4 | i8dmk_gd2 | –, –, – | 30.3/13.5, 32.7/13.0, 31.1/13.1 | 32.5/25.6, 32.5/24.5, 32.5/24.9 | 64.8/41.0, 66.5/39.7, 65.6/39.9 | 9.55, 9.89, 9.60 |
| 8 | u16ft_st | 25.2/22.0, 26.2/20.2, 24.4/20.1 | 16.6/14.1, 17.4/13.5, 16.4/13.3 | 82.8/68.0, 86.2/67.1, 80.4/63.6 | 126.0/105.1, 131.5/101.5, 121.6/98.1 | 18.81, 20.78, 18.46 |
| 8 | u16x256x1728mm_st | 25.1/20.7, 24.7/19.5, 25.5/19.1 | 18.9/13.4, 19.7/13.3, 19.2/12.7 | 41.0/35.3, 41.8/34.2, 39.5/31.6 | 87.0/71.8, 89.0/69.2, 85.7/65.2 | 13.67, 14.22, 13.88 |
| 8 | u16x512x864mm_st | 22.9/18.7, 25.5/19.6, 24.8/18.7 | 17.7/12.5, 20.5/13.5, 20.9/13.0 | 38.4/33.9, 42.6/34.7, 40.0/32.7 | 80.4/67.0, 91.4/69.6, 87.3/66.5 | 12.01, 14.58, 14.58 |
| 8 | u16ad_st | 21.4/18.4, 25.4/19.5, 24.0/18.4 | 16.9/12.2, 20.1/13.3, 18.9/12.5 | 36.3/31.9, 40.9/33.9, 38.2/31.3 | 77.8/64.3, 90.7/68.8, 82.8/64.2 | 11.60, 14.58, 13.49 |
| 8 | i8dmm_gd2 | –, –, – | 20.5/16.5, 21.0/15.4, 20.6/14.2 | 37.6/32.6, 41.1/34.3, 37.9/31.4 | 59.1/50.4, 64.2/51.8, 60.1/47.2 | 8.85, 9.43, 8.88 |
| 8 | i8dmm_st | –, –, – | 16.1/14.1, 16.2/13.2, 15.2/12.5 | 33.1/31.8, 34.6/32.7, 32.6/30.5 | 50.8/47.4, 52.4/47.2, 48.5/44.2 | 9.57, 10.13, 9.54 |
| 8 | i8dmk_st | –, –, – | 15.8/13.3, 16.0/12.8, 15.8/12.5 | 34.2/32.8, 35.7/32.7, 34.6/31.3 | 51.4/47.6, 52.4/46.8, 51.8/45.2 | 9.36, 10.38, 10.28 |
| 8 | i8dmk_gd2 | –, –, – | 20.8/13.7, 23.7/13.6, 22.9/13.4 | 40.5/32.2, 43.6/33.4, 43.6/32.4 | 63.3/48.0, 69.8/49.3, 68.7/47.7 | 9.13, 10.09, 9.87 |

| B | config | max abs err vs ref | cos (min) | instr identical | max abs diff vs u16ft_st / i8d_gd2 |
|---|---|---|---|---|---|
| 2 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 2 | u16x256x1728mm_st | 9.84e-07, 9.10e-07, 9.99e-07 | 1.000000000000 | True | 1.3e-06, 1.4e-06, 1.4e-06 |
| 2 | u16x512x864mm_st | 8.95e-07, 9.40e-07, 9.99e-07 | 1.000000000000 | True | 1.3e-06, 1.4e-06, 1.4e-06 |
| 2 | u16ad_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 2 | i8d_dy | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 2 | i8d_gd2 | 3.58e-06, 3.41e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 2 | i8g_gd2 | 3.67e-06, 3.58e-06, 3.66e-06 | 1.000000000000 | True | 4.8e-06, 5.6e-06, 5.2e-06 |
| 2 | i8dmm_gd2 | 3.67e-06, 3.58e-06, 3.66e-06 | 1.000000000000 | True | 4.8e-06, 5.6e-06, 5.2e-06 |
| 2 | i8dmm_st | 4.42e-06, 4.08e-06, 3.85e-06 | 1.000000000000 | True | 5.5e-06, 5.5e-06, 5.0e-06 |
| 2 | i8dmk_st | 5.02e-06, 3.96e-06, 4.54e-06 | 1.000000000000 | True | 6.1e-06, 5.2e-06, 5.2e-06 |
| 2 | i8dmk_gd2 | 3.02e-06, 3.05e-06, 3.21e-06 | 1.000000000000 | True | 5.2e-06, 5.0e-06, 5.2e-06 |
| 4 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 4 | u16x256x1728mm_st | 1.09e-06, 9.86e-07, 1.02e-06 | 1.000000000000 | True | 1.5e-06, 1.5e-06, 1.5e-06 |
| 4 | u16x512x864mm_st | 1.11e-06, 1.05e-06, 1.18e-06 | 1.000000000000 | True | 1.5e-06, 1.7e-06, 1.5e-06 |
| 4 | u16ad_st | 1.09e-06, 9.86e-07, 1.02e-06 | 1.000000000000 | True | 1.5e-06, 1.5e-06, 1.5e-06 |
| 4 | i8d_dy | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 4 | i8d_gd2 | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 4 | i8g_gd2 | 3.67e-06, 3.86e-06, 3.34e-06 | 1.000000000000 | True | 4.8e-06, 5.6e-06, 5.2e-06 |
| 4 | i8dmm_gd2 | 3.67e-06, 3.86e-06, 3.34e-06 | 1.000000000000 | True | 4.8e-06, 5.6e-06, 5.2e-06 |
| 4 | i8dmm_st | 4.42e-06, 4.08e-06, 4.52e-06 | 1.000000000000 | True | 5.8e-06, 5.5e-06, 5.0e-06 |
| 4 | i8dmk_st | 4.78e-06, 4.56e-06, 5.24e-06 | 1.000000000000 | True | 5.8e-06, 6.9e-06, 4.9e-06 |
| 4 | i8dmk_gd2 | 3.28e-06, 3.79e-06, 3.45e-06 | 1.000000000000 | True | 5.2e-06, 5.1e-06, 4.8e-06 |
| 8 | u16ft_st | 9.98e-07, 1.01e-06, 1.04e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 8 | u16x256x1728mm_st | 1.20e-06, 1.05e-06, 1.21e-06 | 1.000000000000 | True | 1.5e-06, 1.6e-06, 1.7e-06 |
| 8 | u16x512x864mm_st | 1.25e-06, 1.12e-06, 1.21e-06 | 1.000000000000 | True | 1.5e-06, 1.7e-06, 1.8e-06 |
| 8 | u16ad_st | 1.20e-06, 1.05e-06, 1.21e-06 | 1.000000000000 | True | 1.5e-06, 1.6e-06, 1.7e-06 |
| 8 | i8d_dy | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 8 | i8d_gd2 | 3.58e-06, 3.79e-06, 3.73e-06 | 1.000000000000 | True | 0.0e+00, 0.0e+00, 0.0e+00 |
| 8 | i8g_gd2 | 4.42e-06, 4.39e-06, 4.52e-06 | 1.000000000000 | True | 5.8e-06, 5.5e-06, 5.5e-06 |
| 8 | i8dmm_gd2 | 4.42e-06, 4.39e-06, 4.52e-06 | 1.000000000000 | True | 5.8e-06, 5.5e-06, 5.5e-06 |
| 8 | i8dmm_st | 4.42e-06, 4.39e-06, 4.52e-06 | 1.000000000000 | True | 5.8e-06, 5.5e-06, 5.5e-06 |
| 8 | i8dmk_st | 4.78e-06, 4.56e-06, 5.24e-06 | 1.000000000000 | True | 5.8e-06, 6.9e-06, 5.4e-06 |
| 8 | i8dmk_gd2 | 4.78e-06, 4.56e-06, 5.24e-06 | 1.000000000000 | True | 5.8e-06, 6.9e-06, 5.2e-06 |

## Verdict

**not-viable.** The B-adaptive u16 kernel does what the 14:55 note predicted for u16 alone. It ties u16ft_st at B 2 (1.01–1.04x), gains 1.08–1.12x at B 4, and clears the >= 1.3x bar only at B 8 (1.38–1.50x; the identical u16x256x1728mm_st gives 1.38–1.49x). At B 8 the stage profile shows dot worker-ms halved (80–86 → 36–41) while take (~21–25) and cast (~17–20) stay put. The fair int8 control gains more from the same change. i8dmm_gd2 is 1.67–1.69x faster than i8d_gd2 at B 8 (1.48–1.57x for the other i8dmm/i8dmk variants), and 0.94–1.14x at B 4. At B 2, int8 matmul loses (0.77–0.91x), the same pattern as u16. Against the best int8 one-matmul control, adaptive u16 is 1.28–1.38x slower at B 2, 1.38–1.43x at B 4 and 1.41–1.47x at B 8, so the within-1.3x bar is missed at B 4 and B 8. The existing weight-major int8 matmul (i8g_gd2) is the fastest config at B 8 (7.0–7.9 ms vs adaptive u16 12.4–14.7 ms, 1.76–1.86x). The gap reopens rather than closes. Down-only K-blocking did not help int8 (i8dmk ≈ i8dmm), because int8 has no take stage and its cast is already cheap. The ternary cost floor is visible in the profile: at B 8, u16's take + cast (~38–46 worker-ms) is about twice int8's cast alone (~15–21), while the two dot stages are about equal (36–41 vs 33–41). As long as the 2-bit weights are expanded to f32 per call, matmul-friendly batching helps int8 at least as much as u16. Error: all u16 variants stay <= 1.25e-6 from the float64 ternary reference (cos 1.0), and the trits are exact (pack round-trip asserted). int8 variants are <= 5.24e-6 vs the int8 reference. Ternary keeps only the known 3.98x weight-memory win (17.46 vs 69.56 MB).

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-27 18:55 wave reached this lane, and no Chief-named item was found, so this fire took the chained question from the 14:55 note. JuniorLLM HEAD at clone was af593e6 (cloudcover95, 14:51 MT, this lane's 14:55 note). The GitHub commit list since 12:45 MT holds only that commit: no Trit Pack, Handshake or LLC commit in the last 6 hours. This note only adds one new file under junior_osai/vault/findings/.

## Next question

At B 8 the int8 winner is the weight-major i8g (cast tile, tile @ x.T → yT), not x @ tile.T. u16 has never been run weight-major with the fused take, and never past B 8. Build a u16 weight-major kernel (u16gft): fused take per worker range, cast per 256-row tile or down sub-tile, then tile @ x.T → yT. Run it at B 8, B 16 and B 32, where the per-call take + cast cost is amortized over more tokens, against i8g_gd2 and i8dmm_gd2. Does the u16 / int8 ratio fall below 1.3x as B grows? Or does the take + cast floor (~40 worker-ms per layer) keep ternary >= 1.4x behind, which would mean ternary in numpy is a memory-only win at every decode batch?

## Script (v16x.py)

```python
#!/usr/bin/env python3
"""V16X: fair int8 one-matmul control + B-adaptive u16, B 2 / B 4 / B 8, W 8 decode layer, full bitnetd layer, ctx-free.
Base: /workspace/bench1455_1127/v16m.py exec'd without main() (-> v16k.py -> v16t.py -> v16d.py -> v16.py: same weights, P16 8-trit uint16
pack with exact trit round-trip asserts, float64 refs, st / dy / gd2 schedules, r_u16ft kernel, profiling hooks).
New kernels (tmp scratch sized for B 8):
  u16x{R}x{Kc}mm_st : u16ft fused take per static worker range; in-2560 sites unsplit 256-row tiles with ONE np.matmul per tile;
                      down site K-blocked to an (R, Kc) f32 sub-tile (1,769,472 B), ONE np.matmul per sub-tile (14:55 best family)
  u16ad_st          : B-adaptive: B <= 2 -> r_u16ft (per-b np.dot, unsplit); B >= 4 -> u16x256x1728mm kernel
  i8dmm_{gd2,st}    : int8d control with ONE np.matmul per tile (x @ f32(tile).T -> y[:, rows]), all sites unsplit, T 128 gd2 / T 256 st
  i8dmk_{st,gd2}    : int8 one-matmul, down site K-blocked at Kc 1728 (same structure as u16x256x1728mm), others unsplit
Baselines: u16ft_st, i8d_dy / i8d_gd2 (per-row np.dot, T 128), i8g_gd2 (weight-major cast + tile @ x.T matmul, T 128).
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v16x.py out.json"""
import json, os, sys
import numpy as np
_S = open("/workspace/bench1455_1127/v16m.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "v16m.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
BS = (2, 4, 8); TMPN = 8 * 1024
def mk_u(R, KC):
    def r(PR, x, y, r0, r1, T, w, i):
        q = PR.shape[1]; n = r1 - r0; P = PRF[0]; B = x.shape[0]
        TT, kc = (R, KC) if i > D else (256, i)
        su = scr(("u16f", w), (RMAX * Q8,), np.uint64)[:n * q].reshape(n, q); s8 = su.view(np.int8).reshape(n, q * 8)
        sb = scr(("xm", w), (max(256 * F, R * KC),), np.float32); tmp = scr(("xt", w), (TMPN,), np.float32)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT16U, PR[r0:r1], out=su)
        if P: _add(w, "take", pc() - t0, tt() - c0)
        for a in range(0, n, TT):
            m = min(TT, n - a); yv = y[:, r0+a:r0+a+m]
            for j, k0 in enumerate(range(0, i, kc)):
                k1 = min(i, k0 + kc); s = sb[:m * (k1 - k0)].reshape(m, k1 - k0)
                if P: t0 = pc(); c0 = tt()
                np.copyto(s, s8[a:a+m, k0:k1], casting="unsafe")
                if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
                t = tmp[:B * m].reshape(B, m); np.matmul(x[:, k0:k1], s.T, out=t)
                if j == 0: yv[...] = t
                else: np.add(yv, t, out=yv)
                if P: _add(w, "dot", pc() - t0, tt() - c0)
    return r
def mk_8(KC):
    def r(Q, x, y, r0, r1, T, w, i):
        P = PRF[0]; B = x.shape[0]; kc = KC if (KC and i > D) else i
        sb = scr(("x8", w), (max(256 * F, 256 * 1728),), np.float32); tmp = scr(("x8t", w), (TMPN,), np.float32)
        for o0 in range(r0, r1, T):
            m = min(T, r1 - o0); yv = y[:, o0:o0+m]
            for j, k0 in enumerate(range(0, i, kc)):
                k1 = min(i, k0 + kc); s = sb[:m * (k1 - k0)].reshape(m, k1 - k0)
                if P: t0 = pc(); c0 = tt()
                np.copyto(s, Q[o0:o0+m, k0:k1], casting="unsafe")
                if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
                t = tmp[:B * m].reshape(B, m); np.matmul(x[:, k0:k1], s.T, out=t)
                if j == 0: yv[...] = t
                else: np.add(yv, t, out=yv)
                if P: _add(w, "dot", pc() - t0, tt() - c0)
    return r
for R_, K_ in ((256, 1728), (512, 864)): KD[f"u16x{R_}x{K_}mm"] = mk_u(R_, K_)
_ftk = KD["u16ft"]; _mmk = KD["u16x256x1728mm"]
def r_ad(PR, x, y, r0, r1, T, w, i): return (_ftk if x.shape[0] <= 2 else _mmk)(PR, x, y, r0, r1, T, w, i)
KD["u16ad"] = r_ad
K8["8dmm"] = mk_8(0); K8["8dmk"] = mk_8(1728)
_mm_prev = mm
def mm(k, a, kind, T, mode, rec):
    if kind in K8: CUR[0] = k
    return _mm_prev(k, a, kind, T, mode, rec)
C = {"u16ft_st": ("u16ft", 8, "st"), "u16x256x1728mm_st": ("u16x256x1728mm", 8, "st"), "u16x512x864mm_st": ("u16x512x864mm", 8, "st"),
     "u16ad_st": ("u16ad", 8, "st"),
     "i8d_dy": ("8d", 128, "dy"), "i8d_gd2": ("8d", 128, "gd2"), "i8g_gd2": ("8g", 128, "gd2"),
     "i8dmm_gd2": ("8dmm", 128, "gd2"), "i8dmm_st": ("8dmm", 256, "st"), "i8dmk_st": ("8dmk", 256, "st"), "i8dmk_gd2": ("8dmk", 128, "gd2")}
U = [c for c in C if C[c][0] in KD]; BASE = "u16ft_st"; UN = [c for c in U if c != BASE]
I8OLD = ["i8d_dy", "i8d_gd2", "i8g_gd2"]; I8MM = ["i8dmm_gd2", "i8dmm_st", "i8dmk_st", "i8dmk_gd2"]; I8 = I8OLD + I8MM
PROFC = U + I8MM
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
        for c in U: e[c]["vs_u16ft_mab"] = float(np.abs(outs[c] - outs[BASE]).max())
        for c in I8: e[c]["vs_i8d_mab"] = float(np.abs(outs[c] - outs["i8d_gd2"]).max())
        res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.12f}", v["instr_identical"]) for c, v in e.items()}, flush=True)
        xb = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32); rm = {c: [] for c in names}
        for c in names: layer(xb, *C[c], [])
        for rd in range(ROUNDS):
            for c in names[rd % len(names):] + names[:rd % len(names)]:
                s = []
                for _ in range(REPS):
                    t0 = pc(); layer(xb, *C[c], []); s.append((pc() - t0) * 1e3)
                rm[c].append(float(np.median(s)))
        o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), rm=v) for c, v in rm.items()}; rat = {}
        def pr_(a, b, key):
            v = [p / q for p, q in zip(rm[a], rm[b])]; rat[key] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v), a=a, b=b)
        bo = min(I8OLD, key=lambda c: o[c]["med"]); bm = min(I8MM, key=lambda c: o[c]["med"]); bu = min(UN, key=lambda c: o[c]["med"])
        for c in UN: pr_(BASE, c, f"u16ft/{c}")
        pr_("u16ad_st", "i8d_gd2", "u16ad/i8d_gd2"); pr_("u16ad_st", bo, "u16ad/best_i8old"); pr_("u16ad_st", bm, "u16ad/best_i8mm")
        pr_(bu, bm, "best_u16/best_i8mm"); pr_(bo, bm, "best_i8old/best_i8mm"); pr_(BASE, bo, "u16ft/best_i8old")
        for c in I8MM: pr_("i8d_gd2", c, f"i8d_gd2/{c}")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:20s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:28s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
        pr = {}; PRF[0] = True
        for c in PROFC:
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

close: 7a911df155e204d3
