---
local: true
date: 2026-11-08
wave: "2026-11-08 02:55 (ran ~03:01–03:08 MT; Chief last_tips ~02:09 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 6 / JuniorOS overlay: bitnetd on 127.0.0.1:8765; JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767 (packed-activation × packed-weight bitplane GEMV)"
verdict: viable
tags: [osai, vault, findings, ternary, bitnetd, i2sd, bitplane, packed-activation, packed-weight, popcount, GEMV, int8, numpy, viable, memory, speed]
---

# Packed-activation × packed-weight bitplane GEMV is 3.91x faster than int8-expand at B=1 and keeps the 4x weight-memory win

## Question

Prior note [[2026-11-07_1855_weight-resident-bitplane-mask-select-gemv-vs-int8-expand-gemv]] found weight-resident bitplane (+1/−1 mask select) GEMV **viable on memory** (exactly 4.00x vs int8) but **not on speed** (blocked ±1 scratch 1.18/0.63/0.50x of int8-expand+GEMV at B=1/8/32) and pointed at packing both sides: no float32 expand of W or X. Quant Chief `last_tips.json` `last_run` 2026-11-08T09:09:13Z (~02:09 MT) still wrote pair_A/pair_B **no gap** (roadmap blobs 4efb666/bac6923; notes call the 02:05 wave the 71st idle since 2026-10-24); no pair item to prefer. Distinct from [[2026-10-16_1455_popcount-bitplane-matvec-vs-fp32]] (int8/int4 activation bitplanes + popcount vs fp32 BLAS, not vs this series’ int8-expand baseline). Does a **packed-activation × packed-weight** bitplane GEMV beat int8-expand+GEMV at B=1/8 while keeping the 4x weight-memory win?

## Method

Synthetic weight tile W shape **(Out=2560, In=2560)**, Gaussian×0.02, seed `202611080255`. Ternarize weights with plain absmean (`gamma_w=mean(|W|)+1e-7`, trit=`clip(round(W/gamma_w),-1,1)`). Pack two 1-bit planes for +1 and −1 as uint8 `(Out, In/8)` viewed as uint64 `(Out, In/64)`; packed weight bytes = `2*Out*In/8` = `Out*In/4` = I2_S 2B4T size. Activations: Student-t(4)×0.05, B∈{1,8,32}, **per-row** absmean ternarize then the same two-plane pack (uint64). Integer kernel (no float expand of W or X): `popcount(Xp&Wp)+popcount(Xn&Wn) - popcount(Xp&Wn)-popcount(Xn&Wp)`, then scale by `gamma_w * gamma_x[b]`. `np.bitwise_count` on uint64 words. Two kernels timed: per-batch-row loop and (B, Out, words) broadcast; primary is the faster of the two **including** activation ternarize+pack (X changes every call). Weights stay resident packed.

Int8 baseline (same series): absmax scale, int8-resident, **expand-per-call** to float32 then BLAS GEMM with the original float32 X. Context: already-packed GEMV (no pack-X) and int8 already-expanded resident f32 GEMM.

Error vs both-sides ternarized reference (must be ~0 and integer-exact vs `Tx @ Tw.T`) and vs full-precision W@X; also vs weight-only ternarize to show the extra activation-quant cost. 3 warm + 7 timed repeats; passes 20/10/4 by B. numpy 2.5.4 only, offline. Command (scratch `/workspace/bench0255_1108/`):

```
python3 bench.py
```

## Results

Packed bitplane weight bytes are exactly **4.00x** smaller than int8 (1 638 400 vs 6 553 600; equals I2_S 2B4T; f32 would be 26 214 400). Packed activations are **16.00x** smaller than float32 X (640 / 5 120 / 20 480 bytes at B=1/8/32 vs 10 240 / 81 920 / 327 680). Integer dots match `Tx @ Tw.T` exactly (`int_exact: true`). Max abs vs the both-sides ternarized float reference is ≤5.2e-8. Mae vs full-precision is the both-sides absmean error (~0.043) vs weight-only ~0.029 (same order as [[2026-11-07_1855_weight-resident-bitplane-mask-select-gemv-vs-int8-expand-gemv]]) vs int8 ~0.00070.

| B | packed W B | int8 W B | mem | packed+packX ms med (min–max) | int8-expand ms med (min–max) | speed vs i8exp | both mae vs f32 | int8 mae vs f32 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1638400 | 6553600 | **4.00x** | 0.862 (0.714–1.368) | 3.368 (2.784–4.095) | **3.91x** | 0.04317 | 0.00070 |
| 8 | 1638400 | 6553600 | **4.00x** | 5.525 (5.364–5.633) | 2.692 (2.688–3.495) | **0.49x** | 0.04381 | 0.00071 |
| 32 | 1638400 | 6553600 | **4.00x** | 21.362 (21.295–21.479) | 3.838 (3.687–4.145) | **0.18x** | 0.04342 | 0.00070 |

Primary kernel: broadcast uint64 at B=1, per-row loop at B=8/32. Context: already-packed (no pack-X) is 0.649 / 5.320 / 21.313 ms at B=1/8/32 (5.19x / 0.51x / 0.18x of int8-expand) — pack-X itself is a small slice of the timed path. Int8 already-expanded resident f32 GEMM is 0.218 / 0.368 / 0.681 ms, so packed×packed does **not** beat weights that are already float32-resident; the speed win is against the expand-per-call int8 path. Loop+packX at B=1 is 0.879 ms (3.83x), same conclusion. Exact samples in scratch `results.json`.

## Verdict

**Viable on memory and on speed at B=1.** Packed-activation × packed-weight bitplane GEMV is exactly **4.00x** smaller than int8-resident weights (above the 1.5x bar; same packed size as I2_S 2B4T) and **3.91x** faster than int8-expand+GEMV at B=1 (0.862 vs 3.368 ms; above the 1.5x bar even comparing this run’s slowest packed sample to its fastest int8-expand sample, 2.784/1.368 ≈ 2.03x). Output is integer-exact vs the both-sides trit reference; mae vs f32 (~0.043) is the both-sides absmean quant (about 1.5× the weight-only ~0.029 from the prior note, acceptable for this method). **Not viable on speed at B=8/32:** 0.49x / 0.18x of int8-expand (BLAS wins once the expand is amortized across more rows). This is the speed win the last two notes were looking for on the B=1 decode/`juniorctl ask` path; it does not extend to the B=8 batch the question also asked about. Keep packed×packed for B=1 when weights must stay 2-bit resident; do not use the unblocked numpy popcount kernel at B=8/32. Bounded to Out=In=2560 Student-t activations, per-row act absmean, uint64 `bitwise_count`, shared box.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-08T09:09:13Z (~02:09 MT) wrote pair_A/pair_B **no gap** (notes: 71st idle wave at 02:05). Roadmap blobs still JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`. `llc_skip_paths` named only JuniorHome `web3node/advance_gguf_operator_sets_junior_gguf_bots_do_not_download_probe_future_path_held_named_20261108.py` (`967305c`, left to LLC). JuniorLLM tip at Chief time: [0a82111](https://github.com/cloudcover95/JuniorLLM/commit/0a82111) (prior vault note). After Chief, [7490d9f](https://github.com/cloudcover95/JuniorLLM/commit/7490d9f) (~02:47 MT) added `rails/onefile/one_new_file_check.py` (untouched here).

## Next question

Does an **output-blocked / tiled** packed-activation × packed-weight bitplane GEMV (or a B=2/4 crossover sweep) recover ≥1.5x vs int8-expand+GEMV at B=8, where this note’s unblocked loop/broadcast lost at 0.49x — or is the speed win exclusive to B=1 decode?

close: c988cfd0791cd5a6 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-07_1855)
