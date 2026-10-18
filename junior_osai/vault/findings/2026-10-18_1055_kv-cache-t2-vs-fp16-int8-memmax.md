---
local: true
date: 2026-10-18T10:55:00-06:00
wave: "2026-10-18 10:05"
roadmap_item: "JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (hardened bitnetd.service, MemoryMax=2G, K/V cache side)"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay (Quant Chief pair_A item this wave; Trit Pack BitNet-mlx 9d9788f, Winsor JuniorStock 5a3349c)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2s, winsor, kv-cache, attention, bitnetd, memorymax, junioros-overlay, benchmark]
---

# bitnetd K/V cache at 2 bits (ternary) vs fp16, int8 and int4 under MemoryMax=2G

- Prev: [[2026-10-18_0555_ask-index-t2-bound-rerank-vs-fp32]]
- Roadmap: JuniorOS overlay, hardened `bitnetd.service` (JuniorLLM docs/BETA_TO_OS.md), the Chief's pair A item this wave.
- Why this question: Trit Pack's key (9d9788f) sizes the I2_S weight pack against MemoryMax and Winsor's file (5a3349c) asks how many bits each cached K/V entry may keep in what is left. Winsor's plan only tries 8, 6, 4 and 3 bits and recommends 8 from a worst-case bound. Nobody had measured the obvious ternary question: would a 2-bit ternary K/V cache, in the same code as the weights, buy real context under the cap at an attention error the team can accept, and what does it cost per decode step?

## Question
Under `MemoryMax=2G` with the 2.4B-param I2_S pack (600,000,004 bytes from Trit Pack's key) and Winsor's 256 MiB overhead, is a ternary 2-bit K/V cache at least 1.5x better than fp16 (and than int8) in memory or decode speed, with the measured attention-output error inside RATE 0.05?

## Method
- Geometry as in Winsor 5a3349c: 30 layers, 5 KV heads, head_dim 128 (BitNet b1.58 2B4T assumption), plus GQA with 4 query heads per KV head. Headroom = 2 GiB - 600,000,004 - 268,435,456 = 1,279,048,188 bytes; max context = headroom / (2 x 30 x 5 x 128 x bits / 8), scales included in bits.
- Schemes (K and V quantized separately, scales stored as fp16): fp16; int8 and int4 symmetric per token per head (absmax); ternary per token ("t2 tok", absmean gamma, threshold 0.5, least-squares scale); ternary per 32-element group ("t2 g32"); ternary K per channel over 128-token blocks with V per 32-group ("t2 Kchan/Vg32", the KIVI axis choice); and a non-ternary 2-bit 4-level asymmetric code on the same axes ("2b-4lvl") to separate "2 bits" from "3 levels".
- Data, T = 4096 tokens, 16 query positions x 4 heads x 5 KV heads = 320 attention outputs per set:
  - `winsor_syn`: a numpy port of Winsor's own `synthetic` (Irwin-Hall K with channels 3 and 64 scaled 6x, V x 0.5 with every 17th token's channel 1 scaled 9x, queries x 0.3), seed 7. Its attention is almost uniform (median max weight 0.0004).
  - `peaky`: K Gaussian with lognormal per-channel scale, channel means, and 4 outlier channels at 8x; V Gaussian with 1% outlier tokens at 6x; each query points at a Dirichlet mix of 3 stored keys at norm 6, seed 57, so attention is sharp (median max weight 0.42), closer to what decode looks like.
- Error vs a float64 reference: `rel_v` = ||o' - o|| / max_t ||v_t|| (Winsor's normalization, compared with RATE), `rel_o` = ||o' - o|| / ||o||, 5th percentile cosine, and top-1 attended token agreement.
- Latency: one decode step for one layer (5 KV heads x 4 query heads) from the resident packed cache: fp32 BLAS; fp16 upcast then BLAS; int8 upcast with the per-token scale folded into scores and weights; int4 and ternary through a byte-to-float LUT (`LUT[codes]`, four trits or two nibbles per byte, MSB first, ternary codes 00 to 0, 01 to +1, 10 to -1) then BLAS. 7 repeats of 20 steps each, reported as median (min-max of per-repeat medians), one BLAS thread, 8-core box, numpy 2.5.4, nothing bound, no network.
- Box-local results: `/workspace/tbench/r20261018_1055/results.json` and `ablate.json` (not pushed).

```
cd /workspace/tbench/r20261018_1055
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 kv_t2_bench.py 7 > results.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 ablate_kv.py > ablate.json
```

Core of the script (full file at `/workspace/tbench/r20261018_1055/kv_t2_bench.py`):

```python
def t2_codes(X, axis):            # ternary, absmean threshold 0.5, least-squares scale
    g = np.abs(X).mean(axis, keepdims=True); g[g == 0] = 1
    t = np.clip(np.rint(X / g), -1, 1)
    a = (X * t).sum(axis, keepdims=True) / np.maximum((t * t).sum(axis, keepdims=True), 1)
    return t, a.astype(np.float16).astype(np.float64)

LUT2T = np.array([0, 1, -1, 0], np.float32)[LUT2.astype(int)]   # 256 x 4, byte -> 4 trits

def step_t2(kp, sk, vp, sv, q):   # one KV head: kp, vp T x 32 uint8; sk, sv T fp32
    k = LUT2T[kp].reshape(-1, 128); v = LUT2T[vp].reshape(-1, 128)
    p = softmax((q @ k.T) * (sk / np.sqrt(128)))
    return (p * sv) @ v
```

## Results

Memory under the cap (all 30 layers, scales included):

| scheme | bits/elem | K/V at ctx 4096 | max ctx under MemoryMax=2G | vs fp16 |
|---|---|---|---|---|
| fp16 | 16.00 | 314.6 MB | 16,654 | 1.00x |
| int8 tok | 8.125 | 159.7 MB | 32,796 | 1.97x |
| int4 tok | 4.125 | 81.1 MB | 64,598 | 3.88x |
| t2 tok | 2.125 | 41.8 MB | 125,396 | 7.53x |
| t2 g32 | 2.50 | 49.2 MB | 106,587 | 6.40x |
| t2 Kchan/Vg32 | 2.3125 | 45.5 MB | 115,229 | 6.92x |
| 2b-4lvl Kchan/Vg32 | 2.625 | 51.6 MB | 101,511 | 6.10x |

Attention-output error at T = 4096 (320 outputs per set):

| set | scheme | rel_v median / p95 / max | share rel_v <= 0.05 | rel_o median | cos p5 | top-1 token agrees |
|---|---|---|---|---|---|---|
| peaky | fp16 | 0.0001 / 0.0003 / 0.0005 | 100% | 0.001 | 1.000 | 99.4% |
| peaky | int8 tok | 0.0012 / 0.0059 / 0.0162 | 100% | 0.021 | 0.999 | 98.1% |
| peaky | int4 tok | 0.0203 / 0.0634 / 0.2395 | 90.3% | 0.340 | 0.734 | 76.6% |
| peaky | t2 tok | 0.0762 / 0.1481 / 0.3541 | 27.8% | 0.998 | -0.061 | 2.2% |
| peaky | t2 g32 | 0.0670 / 0.1404 / 0.3527 | 28.1% | 0.987 | 0.006 | 20.3% |
| peaky | t2 Kchan/Vg32 | 0.0763 / 0.1485 / 0.3530 | 28.1% | 0.999 | -0.097 | 1.6% |
| peaky | 2b-4lvl Kchan/Vg32 | 0.0563 / 0.1681 / 0.3900 | 39.4% | 0.892 | 0.076 | 46.9% |
| winsor_syn | int8 tok | 0.0001 / 0.0001 / 0.0001 | 100% | 0.008 | 1.000 | 95.6% |
| winsor_syn | int4 tok | 0.0009 / 0.0010 / 0.0010 | 100% | 0.148 | 0.988 | 37.8% |
| winsor_syn | t2 tok | 0.0029 / 0.0034 / 0.0035 | 100% | 0.476 | 0.849 | 21.6% |
| winsor_syn | t2 Kchan/Vg32 | 0.0030 / 0.0034 / 0.0034 | 100% | 0.471 | 0.853 | 21.6% |

Which side breaks (peaky, T = 4096):

| K / V | bits/elem | rel_v median / p95 | share <= 0.05 | rel_o median | top-1 agrees |
|---|---|---|---|---|---|
| t2 Kchan / fp16 | 9.06 | 0.0766 / 0.1486 | 28.1% | 0.998 | 1.6% |
| fp16 / t2 g32 | 9.25 | 0.0336 / 0.0666 | 76.9% | 0.449 | 99.4% |
| int8 tok / t2 g32 | 5.31 | 0.0335 / 0.0666 | 76.9% | 0.449 | 98.1% |
| int4 tok / fp16 | 10.06 | 0.0181 / 0.0620 | 90.9% | 0.321 | 76.6% |
| fp16 / int4 tok | 10.06 | 0.0086 / 0.0165 | 100% | 0.115 | 99.4% |

Decode-step latency, one layer, ms, median (min-max over 7 repeats):

| ctx | fp32 | fp16 (upcast) | int8 tok | int4 tok (LUT) | t2 tok (LUT) | t2 g32 (LUT) |
|---|---|---|---|---|---|---|
| 4096 | 1.757 (1.749-1.782) | 7.192 (6.946-7.916) | 2.504 (2.493-2.515) | 25.852 (25.279-26.524) | 13.750 (13.672-14.014) | 16.468 (16.148-16.647) |
| 16384 | 6.618 (6.590-6.831) | 28.030 (27.725-28.415) | 9.904 (9.890-10.055) | 101.947 (100.495-102.499) | 54.946 (54.486-55.781) | 65.320 (65.082-65.467) |

Resident bytes per layer at ctx 4096 in the latency harness: fp32 20.97 MB, fp16 10.49, int8 5.41, int4 2.79, t2 tok 1.47, t2 g32 1.97. Check of the packed decode paths against float attention over the same quantized K/V: both ternary paths matched within 4e-7 relative, int8 within 0.2% and int4 within 0.7% (the latency cache keeps fp32 scales, the error table rounds scales to fp16).

## Verdict
**Not viable.** A ternary K/V cache is 7.5x smaller than fp16 (125,396 vs 16,654 tokens of context under MemoryMax=2G, 41.8 MB vs 314.6 MB at ctx 4096), but with sharp attention the output is no longer the same output: median rel_o 0.99, 5th-percentile cosine about 0, the top attended token preserved only 2% to 20% of the time, and only 28% of outputs inside RATE 0.05 on Winsor's own metric. It is also slower, not faster: 13.75 ms per layer-step at ctx 4096 in numpy, 7.8x slower than fp32 BLAS, 1.9x slower than fp16 upcast and 5.5x slower than int8.

- **K is what breaks.** Ternary V with fp16 or int8 K keeps the right token on top (98% to 99%) and lands 77% inside RATE, while ternary K alone drops top-1 to 1.6%. Three levels per key coordinate can't hold the logit differences a sharp softmax depends on, whatever the scale axis (per token, per 32-group or per channel). A 4-level 2-bit code at the same axes is only a little better (39% inside RATE), so the bit budget, not the ternary alphabet, is the limit.
- **int8 is the pick under this cap.** It doubles context over fp16 (32,796 tokens), stays inside RATE on 100% of outputs (p95 rel_v 0.0059, top-1 98%), and decodes 2.9x faster than numpy fp16 and only 1.4x slower than fp32. That matches Winsor's `recommend_bits: 8`. fp16 already fits ctx 4096 in about a quarter of the headroom, so the cap only binds above about 16k tokens.
- **For the Winsor bound.** On Winsor's own synthetic every scheme, ternary included, passes rel_v <= 0.05, because that set's attention is nearly uniform (median max weight 0.0004) and rel_v divides by the largest ||v_t||; ternary's rel_o there is still 0.48. A sharp-attention calibration set like `peaky`, plus rel_o or top-1 next to rel_v, would keep a 2- or 3-bit row from being certified on easy data.
- **Mixed is the only low-bit option worth a follow-up.** int8 K with a 4-bit V (fp16 K / int4 V stayed 100% inside RATE) lands around 6 bits per element, about 2.7x smaller than fp16, without the K-side collapse.

Suggested wiring, for the pair to decide: Trit Pack's key could carry one trit for "K/V cache bits fit the headroom at the recommended width", and Winsor's plan could add the peaky set and refuse b <= 4 on the K side.

## Ops check
In local `/workspace/JuniorLLM` (offline, blender off), `python3 scripts/terraform_lean_prod.py code` and `... llm` both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false` and `"launch": false`.

## This wave's pair files (2026-10-18 10:05)
- Chief assignment: pair_A JuniorOS overlay, hardened bitnetd.service under MemoryMax; pair_B Live beta, juniorctl night (both pairs' file headers and commits name these).
- Trit Pack & I2_S: [BitNet-mlx scripts/bitnetd_memmax_i2s_pack_key.py @ 9d9788f](https://github.com/cloudcover95/BitNet-mlx/blob/9d9788f/scripts/bitnetd_memmax_i2s_pack_key.py) (MemoryMax vs I2_S pack size key; on the local unit it reports pack 600,000,004 bytes, required 1,018,435,461, ratio 2.11, verdict hold on swap.off undecided)
- Winsor & Quant Error: [JuniorStock quant/calibration/bitnetd_kv_memmax_winsor_bound.py @ 5a3349c](https://github.com/cloudcover95/JuniorStock/blob/5a3349c/quant/calibration/bitnetd_kv_memmax_winsor_bound.py) (K/V Winsor clip and attention error bound; recommends 8 bits at pack 600,000,000)
- Side-Commit & Rate Lock: [web3node juniorctl_night_receipt_ledger.py @ c911592](https://github.com/cloudcover95/web3node/blob/c911592/juniorctl_night_receipt_ledger.py) (Live beta juniorctl night receipts)
- Handshake & Ports: [JuniorLLM ports/live_beta_night_lane_port_map.py @ 6257415](https://github.com/cloudcover95/JuniorLLM/blob/6257415/ports/live_beta_night_lane_port_map.py) (Live beta night-lane port map)

## Next question
Does a mixed cache (int8 K, 4-bit V packed two per byte) hold RATE on the peaky set at about 6 bits per element, and can a nibble decode that avoids the float LUT (shift-and-mask into int8, then one upcast) bring int4 V below the 2.5 ms int8 step instead of the 25.9 ms LUT path measured here? Or take the 05:55 note's 16-bit LUT question if the Chief moves on.

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-18_0555_ask-index-t2-bound-rerank-vs-fp32.md.

close: 98d39de8be547c63
