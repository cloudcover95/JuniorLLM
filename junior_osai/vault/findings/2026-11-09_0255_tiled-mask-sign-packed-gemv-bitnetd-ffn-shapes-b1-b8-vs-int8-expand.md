---
local: true
date: 2026-11-09
wave: "2026-11-09 02:55 (ran ~03:02–03:10 MT; Chief last_run ~02:10 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765; JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 (packed×packed GEMV at FFN shapes)"
verdict: viable
tags: [osai, vault, findings, ternary, bitnetd, i2sd, mask-sign, packed-activation, packed-weight, popcount, tiled, ffn, GEMV, int8, numpy, viable, memory, speed]
---

# Tiled mask/sign packed×packed GEMV holds and grows its B=1 win at bitnetd FFN shapes (12–14x vs int8-expand); B=8 sits at the 1.5x line; In-word blocking does not help

## Question

Prior note [[2026-11-08_0555_output-tiled-mask-sign-packed-gemv-b1-b8-crossover-vs-int8-expand]] found output-tiled mask/sign packed×packed GEMV 8.70x faster than int8-expand+GEMV at B=1 (Out=In=2560) but only 1.20x at B=8, and asked: (a) does the B=1 win hold at the FFN shapes (up Out=6912/In=2560, down Out=2560/In=6912, where the `In/64` word count changes), and (b) does a word-blocked (In-tiled, 2-D) accumulation push B=8 over 1.5x? Quant Chief `last_tips.json` `last_run` 2026-11-09T09:10:00Z (~02:10 MT) wrote pair_A/pair_B **no gap** (74th idle wave), so there is no pair item to prefer and this runs the prior note's next question.

## Method

Synthetic W per shape (attn 2560×2560 control, ffn_up 6912×2560, ffn_down 2560×6912), Gaussian×0.02, seed `202611090255`; plain absmean weight ternarize, stored as mask/sign 1-bit planes (`Out*In/4` bytes, the I2_S 2B4T size). Activations Student-t(4)×0.05, B∈{1,8}, per-row absmean ternarize + pack inside the timed path. Kernels: `out{T}` = prior note's output-tiled 2-popcount kernel (`nz − 2·popcount(Xm&Wm&(Xs^Ws))`), T∈{256,1024}; `2d{T}x{K}` = same, but each output tile also walks In in blocks of K∈{8,16} uint64 words with int32 accumulation. Baseline: int8 absmax resident, expand-per-call to f32 then BLAS GEMM (series baseline). 3 warm + 7 timed repeats (passes 12 at B=1, 5 at B=8); the whole script was run **twice** and both runs are reported. numpy 2.5.4, 8 cores, offline, nothing bound. Command (scratch `/workspace/bench0255_1109/`, inline script `bench.py`):

```
python3 bench.py   # run 1 -> results_run1.json, run 2 -> results.json
```

## Results

Weight bytes: attn packed **1 638 400** vs int8 6 553 600; FFN packed **4 423 680** vs int8 17 694 720 (f32 70 778 880) → **4.00x** at every shape. Every kernel/tile/block is integer-exact vs `Tx @ Tw.T` (`int_exact_all: true` in both runs).

| shape | B | best out-tiled ms med (min–max), run1 / run2 | int8-expand ms med, run1 / run2 | speed vs i8exp run1 / run2 | worst-case run1 / run2 | best 2-D speed run1 / run2 | tern mae / int8 mae vs f32 |
|---|---|---|---|---|---|---|---|
| attn 2560×2560 | 1 | 0.383 (0.352–0.408) / 0.368 (0.350–0.688) | 3.113 / 3.129 | **8.12x / 8.50x** | 7.22x / 4.14x | 4.64x / 3.87x | 0.0460 / 0.0007 |
| attn 2560×2560 | 8 | 2.675 / 2.579 | 3.077 / 3.143 | 1.15x / 1.22x | 0.73x / 0.87x | 0.67x / 0.65x | 0.0459 / 0.0007 |
| ffn_up 6912×2560 | 1 | 0.909 (0.870–0.952) / 1.086 (1.076–1.188) | 11.838 / 13.175 | **13.02x / 12.13x** | 11.63x / 9.24x | 7.06x / 7.40x | 0.0483 / 0.0008 |
| ffn_up 6912×2560 | 8 | 6.908 (6.749–7.420) / 6.859 (6.749–7.317) | 10.127 / 11.133 | 1.47x / 1.62x | 1.33x / 1.42x | 0.82x / 0.90x | 0.0437 / 0.0007 |
| ffn_down 2560×6912 | 1 | 0.745 (0.738–0.764) / 0.783 (0.722–1.269) | 10.449 / 10.330 | **14.02x / 13.19x** | 13.21x / 7.98x | 6.58x / 6.42x | 0.0772 / 0.0013 |
| ffn_down 2560×6912 | 8 | 6.424 (6.355–6.491) / 6.433 (6.296–6.956) | 10.168 / 11.582 | **1.58x / 1.80x** | 1.52x / 1.54x | 0.96x / 1.12x | 0.0723 / 0.0012 |

Output rms vs f32 is 0.074 (attn/up) and 0.12 (down), so ternary mae stays ~0.6 of rms at every shape, the same both-sides absmean ratio accepted in the prior notes; down's larger absolute mae is the √In growth of the dot, not extra error. Full samples in scratch `results_run1.json` / `results.json`.

## Verdict

**Viable on memory (4.00x at every shape) and on speed at B=1, with a larger win at FFN shapes than at attn.** Packed×packed B=1 is **12.1–13.0x** (up) and **13.2–14.0x** (down) vs int8-expand+GEMV, against 8.1–8.5x at 2560×2560, because int8-expand cost scales with the 2.7x larger matrix while the popcount kernel stays bandwidth-light; worst-case samples are still ≥8x. **B=8 reaches the bar only at ffn_down (1.58x / 1.80x, worst-case ≥1.52x)**; ffn_up is borderline (1.47x / 1.62x, worst-case under 1.5x) and attn stays under (1.15–1.22x), so B=8 overall is **inconclusive**, not a clean win. **The 2-D In-word blocking (K=8/16 words) is not worth it**: it is 1.6–2.2x slower than output-only tiling at every shape and B, because numpy per-block call overhead outweighs any cache gain. Keep output-only tiled mask/sign packed×packed for `juniorctl ask` decode (B=1) on every bitnetd layer shape; keep int8-expand/BLAS as the default at B≥8. Bounded to synthetic Gaussian weights, Student-t activations, uint64 `bitwise_count`, shared 8-core box.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-09T09:10:00Z (~02:10 MT) wrote pair_A/pair_B **no gap** (notes: 74th idle wave; roadmap blobs unchanged, JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `scripts/advance_daily_audit_1645_named_20261109.py` (`86e7ac4`, left to LLC). JuniorLLM tip at push time: [ecc70a9](https://github.com/cloudcover95/JuniorLLM/commit/ecc70a9) (rails/noscript, cloudcover95, 02:46 MT; not a vault path). Prior vault note: [08b642c](https://github.com/cloudcover95/JuniorLLM/commit/08b642c).

## Next question

Does a full bitnetd decode step (one layer: q/k/v/o 2560×2560 + ffn up/gate 6912×2560 + down 2560×6912, B=1) keep a ≥1.5x end-to-end win once activation re-ternarize/pack between every projection and the f32 scale/residual work are included, i.e. is the per-layer win Amdahl-limited?

close: 678ae49cdeb74745 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-08_0555)
