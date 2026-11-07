---
local: true
date: 2026-11-07
wave: "2026-11-07 05:05 (bench 05:55, ran ~05:56–06:05 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 6 / JuniorOS overlay: bitnetd on 127.0.0.1:8765 (fallback Handler = rails/linux/i2sd.py UNPACK); JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767"
verdict: viable
tags: [osai, vault, findings, ternary, bitnetd, i2sd, handler, i2s, int8, decode, token-batch, numpy, loopback, viable, memory]
---

# bitnetd/i2sd handler buffer: I2_S 2-bit block decode is 4.0x smaller than int8 but 4.5–11x slower to expand

## Question

Prior note [[2026-11-07_0255_terraform-lean-trit-energy-ternary-count-vs-float-path]] closed the Terraform lean surface (not compute-bound) and asked for the bitnetd/i2sd in-process path. Quant Chief 05:05 wrote both pairs no gap (67th idle wave); no pair item to prefer. Does I2_S 2-bit block decode into the 127.0.0.1 bitnetd/i2sd handler float32 buffer beat int8 at a realistic per-request token batch (2B4T H=2560, B=1..128)?

## Method

`bitnetd.sh` advertises `127.0.0.1:8765` and falls through to `rails/linux/i2sd.py` (Handler PORT 8767). Bench uses the same `junior_bitnet.i2s` pack bit order (4 trits/byte, MSB-first codes 0/1/2) and fills a preallocated float32 handler buffer:

- **lut**: numpy 256→4 int8 trit LUT + scale (in-process path a Handler should use).
- **int8**: absmax int8 payload → float32 × mult (same buffer shape).
- **pure**: today's `i2s.unpack` list path (B≤8 only).
- **loopback**: real `i2sd.Handler` UNPACK over 127.0.0.1 ephemeral port 18765 (advertised :8765 left alone); hex wire as shipped.

Activations: Student-t(4)×0.05, seed `202611070555`. I2_S absmean gamma; int8 absmax. 3 warm + 7 timed repeats; passes 80/40/20/10 by B. numpy 2.5.4, python3 only, offline. Local tree `/workspace/JuniorLLM` (tip inputs match vault parent `594a03b` for `junior_bitnet/i2s.py` + `rails/linux/i2sd.py`). Command (scratch `/workspace/bench0555_1107/`):

```
PYTHONPATH=/workspace/JuniorLLM python3 bench.py
```

## Results

LUT decode is **exact** vs trit reference (0.0 max abs) on every B. Int8 reconstruct exact vs its codes.

| B | N=B×2560 | I2_S B | int8 B | mem vs int8 | lut µs med (min–max) | int8 µs med (min–max) | speed lut/int8 | I2_S mae vs f32 | int8 mae vs f32 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2560 | 640 | 2560 | **4.00x** | 10.15 (9.37–12.20) | 2.22 (2.21–2.32) | 0.22x | 0.0246 | 0.00106 |
| 8 | 20480 | 5120 | 20480 | **4.00x** | 56.03 (53.37–66.32) | 6.50 (6.49–6.68) | 0.12x | 0.0247 | 0.00170 |
| 32 | 81920 | 20480 | 81920 | **4.00x** | 246.87 (218.46–261.00) | 21.94 (21.00–28.09) | 0.09x | 0.0252 | 0.00494 |
| 64 | 163840 | 40960 | 163840 | **4.00x** | 430.25 (413.81–463.68) | 40.46 (39.94–47.47) | 0.09x | 0.0250 | 0.00369 |
| 128 | 327680 | 81920 | 327680 | **4.00x** | 876.55 (862.07–914.47) | 111.73 (110.34–124.43) | 0.13x | 0.0251 | 0.00336 |

Pure-Python `unpack` (current Handler): B=1 **201.0 µs** (~20x slower than lut), B=8 **1604 µs**. Loopback UNPACK B=1 median **0.47 ms** (0.43–0.93) for 640 packed bytes on a 1280-char hex line — wire+text dominates in-process lut (~10 µs). Handler float32 buffer is 4N bytes either way; the 4.0x win is the packed resident/wire form.

## Verdict

**Viable on memory.** Packed I2_S is exactly **4.00x** smaller than int8 at every batch (2 bits/trit vs 8), above the 1.5x bar, with exact trit decode into the handler buffer. **Not viable on decode speed:** numpy LUT expand is only 0.09–0.22x of int8→float32 (about 4.5–11x slower), and the shipped pure-Python UNPACK is another ~20x behind lut. Ternarization mae vs f32 is ~0.025 vs int8 ~0.001–0.005 — acceptable for this stack's I2_S path, but the efficiency win is storage/hand-off, not per-request expand. Keep weights/activations I2_S-resident for the 4x byte win; do not expect the decode-into-float32 step itself to beat int8. Bounded to H=2560 Student-t activations on a shared box.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-07T12:10:56Z (~05:10 MT) wrote pair_A/pair_B **no gap** (notes: 67th idle wave). Roadmap blobs still JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`. JuniorLLM tip before this push: [594a03b](https://github.com/cloudcover95/JuniorLLM/commit/594a03b) (prior vault note). Pair A bases still Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) / Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Winsor-clipped activation error vs speed at the bitnetd boundary (the other branch left open by [[2026-11-07_0255_terraform-lean-trit-energy-ternary-count-vs-float-path]]), specifically: does a Winsor clip before absmean-ternarize of the per-request activation tile reduce mae vs f32 enough to matter without losing the I2_S 4x packed-byte win, versus absmax int8 at B=1/8/32?

close: 8b087ac93e3d3bd9 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-07_0255)
