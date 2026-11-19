---
local: true
date: 2026-11-18
wave: "2026-11-18 18:55 (ran ~18:42–18:58 MT; Chief last_run 2026-11-19T01:10:00Z ~18:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, kv-cache, a4-g32, a8-g64, recent-window, mixed-precision-kv, gqa, u32-lut, int8-control, ctx-8192, total-bytes, numpy, viable]
---

# Keeping only the last 1024 tokens at A8 g64 and storing older tokens as A4 g32 nibbles brings the ternary total-bytes edge at B = 8 / ctx 8192 back above the bar: **71.0 MiB (2-bit weights + 8 windowed KV caches) vs 151.3 MiB for int8 weights + A8 KV = 2.13x** (was 1.49x at 14:55), and **1.70x** even when the int8 control gets the same windowed KV (120.7 MiB). Per-stream KV shrinks 1.56x (85.0 → 54.4 MiB at B = 8). Error stays tiny: min per-stream layer-delta cos **0.9999973** vs f32-KV whole softmax. The cost is speed: A4 nibble unpack in numpy makes attention 0.78–0.79x and the step 0.83–0.85x of all-A8 at B = 8 / ctx 8192 (130.5 / 139.0 vs 153.1 / 167.4 tok/s). Viable on memory, not on speed.

## Question

[[2026-11-18_1455_multi-stream-batched-decode-b1-b4-b8-per-stream-a8-g64-kv-ctx-2048-8192-u32-lut-vs-int8-weights-total-bytes-full-bitnetd-layer]] found that at B = 8 / ctx 8192 the eight A8 g64 KV caches are ~84% of the bytes and ~65% of the step, so the ternary total-bytes edge over int8 weights fell to 1.49x. Earlier KV notes (2026-11-10_0555 value-only A4, 2026-11-10_1055 Hadamard A4, 2026-11-10_1855 f16 recent window + A8) were single-stream attention only, never against weight bytes. This run asks: in the multi-stream decode layer, does **A4 g32 for tokens older than a 1024-token A8 window** restore a ≥ 1.5x ternary total-bytes edge over int8 weights, and what does it do to multi-stream attention time, at layer cos ≥ 0.9999 per stream?

## Method

- Same full bitnetd decoder layer and multi-stream step as 14:55 (hidden 2560, 20 q heads × 128, 5 KV heads, FFN 6912; `base.py`, `par.py`, `tiled.py`, `cache.py` byte-identical to `/workspace/bench1455_1118`, md5-checked). One shared weight pass per step for B = 1 / 4 / 8 streams; each stream has its own 8192-token history projected through that weight set's qkv.
- KV formats: `a8` (all tokens A8 g64, 14:55 control); `a4w1024` (tokens older than the last 1024 stored as A4 g32: symmetric codes in [−7, 7], two per byte, f32 scale per 32 dims, requantised from their A8 values as they would be when aging out; the last 1024 at A8 g64; each step requantises the one aged-out token per stream, counted in step time); `a4all` (every token A4 g32, error floor).
- Attention: per (stream, KV head) online softmax on a W = 8 pool; old region in 1024-key tiles with A4 unpack via a 256-entry u16 LUT (byte → two int8), window as one A8 tile.
- Weights: 2-bit pack + u32-LUT expand (`par.fmm`, T = 256, W = 8, exact W1.58) vs int8-row control (`par.fmm8`).
- Bytes: KV bytes per token per layer = 2 × 5 × (128 + 8) = 1360 B at A8 g64, 2 × 5 × (64 + 16) = 800 B at A4 g32 (1.70x smaller). Total = weight bytes + B × KV bytes.
- Error: per-stream cos of the layer delta (y − x) vs the same weights with f32 KV and whole-context softmax; also max abs.
- Timing: 5 rounds × 8 timed steps (+2 warm) per config, config order rotated each round, 40 samples per config; two full independent runs. Median and min–max shown.

Command (from `/workspace/bench1855_1118`):

```
OPENBLAS_NUM_THREADS=1 python3 a4win.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 a4win.py results_run2.json > run2.txt
```

Weight bytes: u32 path 16.65 MiB, int8 rows 66.34 MiB.

## Results

Step and attention ms are run1 / run2 medians; the range is min–max over both runs; share and bytes from run1 (bytes are identical across runs).

| cfg | step ms med | step range | attn ms med | share | tok/s | KV MiB | total MiB | layer cos min |
|---|---|---|---|---|---|---|---|---|
| u32_a8_ctx2048_B1 | 22.43 / 22.11 | 20.36–40.87 | 1.67 / 1.69 | 0.074 | 44.6 / 45.2 | 2.66 | 19.31 | 1.0000000 |
| u32_a4w1024_ctx2048_B1 | 22.48 / 21.84 | 20.07–38.51 | 1.78 / 1.77 | 0.079 | 44.5 / 45.8 | 2.11 | 18.76 | 0.9999938 |
| u32_a4all_ctx2048_B1 | 22.54 / 21.78 | 20.13–46.22 | 1.99 / 1.93 | 0.088 | 44.4 / 45.9 | 1.56 | 18.21 | 0.9999881 |
| u32_a8_ctx2048_B4 | 19.92 / 17.99 | 16.61–38.74 | 5.52 / 5.16 | 0.277 | 200.9 / 222.3 | 10.62 | 27.27 | 1.0000000 |
| u32_a4w1024_ctx2048_B4 | 21.67 / 18.80 | 17.12–36.23 | 6.32 / 5.72 | 0.292 | 184.6 / 212.8 | 8.44 | 25.09 | 0.9999938 |
| u32_a4all_ctx2048_B4 | 20.98 / 19.94 | 17.90–32.03 | 6.76 / 6.51 | 0.322 | 190.7 / 200.6 | 6.25 | 22.90 | 0.9999881 |
| u32_a8_ctx2048_B8 | 25.57 / 23.99 | 21.62–52.23 | 10.44 / 9.61 | 0.408 | 312.8 / 333.5 | 21.25 | 37.90 | 1.0000000 |
| u32_a4w1024_ctx2048_B8 | 27.19 / 26.28 | 23.61–41.59 | 11.56 / 11.37 | 0.425 | 294.2 / 304.4 | 16.88 | 33.52 | 0.9999938 |
| u32_a4all_ctx2048_B8 | 27.13 / 26.64 | 23.81–37.53 | 12.43 / 12.01 | 0.458 | 294.9 / 300.3 | 12.50 | 29.15 | 0.9999881 |
| u32_a8_ctx8192_B1 | 24.48 / 24.02 | 22.52–46.08 | 4.44 / 4.30 | 0.181 | 40.8 / 41.6 | 10.62 | 27.27 | 1.0000000 |
| u32_a4w1024_ctx8192_B1 | 26.71 / 26.06 | 23.43–33.24 | 5.46 / 5.48 | 0.205 | 37.4 / 38.4 | 6.80 | 23.45 | 0.9999974 |
| u32_a4all_ctx8192_B1 | 27.16 / 25.84 | 23.79–35.89 | 5.92 / 5.60 | 0.218 | 36.8 / 38.7 | 6.25 | 22.90 | 0.9999971 |
| u32_a8_ctx8192_B4 | 30.45 / 30.43 | 27.46–52.47 | 16.56 / 16.63 | 0.544 | 131.3 / 131.5 | 42.50 | 59.15 | 1.0000000 |
| u32_a4w1024_ctx8192_B4 | 34.94 / 35.72 | 31.83–46.52 | 21.02 / 21.01 | 0.601 | 114.5 / 112.0 | 27.19 | 43.84 | 0.9999974 |
| u32_a4all_ctx8192_B4 | 38.87 / 35.03 | 32.52–60.12 | 22.06 / 21.08 | 0.568 | 102.9 / 114.2 | 25.00 | 41.65 | 0.9999971 |
| u32_a8_ctx8192_B8 | 52.25 / 47.79 | 44.27–71.98 | 33.56 / 32.35 | 0.642 | 153.1 / 167.4 | 85.00 | 101.65 | 1.0000000 |
| u32_a4w1024_ctx8192_B8 | 61.32 / 57.55 | 50.30–76.88 | 42.93 / 40.86 | 0.700 | 130.5 / 139.0 | 54.38 | 71.02 | 0.9999973 |
| u32_a4all_ctx8192_B8 | 58.18 / 56.68 | 52.01–94.48 | 41.77 / 41.50 | 0.718 | 137.5 / 141.2 | 50.00 | 66.65 | 0.9999970 |
| int8_a8_ctx2048_B1 | 17.21 / 15.95 | 14.16–23.92 | 1.89 / 1.72 | 0.110 | 58.1 / 62.7 | 2.66 | 68.99 | 0.9999998 |
| int8_a4w1024_ctx2048_B1 | 17.96 / 16.18 | 14.62–29.32 | 2.31 / 1.97 | 0.128 | 55.7 / 61.8 | 2.11 | 68.45 | 0.9999765 |
| int8_a8_ctx2048_B4 | 13.99 / 13.98 | 12.24–20.26 | 5.31 / 5.45 | 0.380 | 285.9 / 286.1 | 10.62 | 76.96 | 0.9999998 |
| int8_a4w1024_ctx2048_B4 | 15.38 / 14.82 | 12.81–34.58 | 6.14 / 6.09 | 0.399 | 260.1 / 269.8 | 8.44 | 74.77 | 0.9999765 |
| int8_a8_ctx2048_B8 | 21.37 / 20.64 | 16.86–38.32 | 10.32 / 10.25 | 0.483 | 374.3 / 387.6 | 21.25 | 87.59 | 0.9999998 |
| int8_a4w1024_ctx2048_B8 | 22.76 / 20.90 | 18.72–38.22 | 11.46 / 11.15 | 0.504 | 351.5 / 382.7 | 16.88 | 83.21 | 0.9999756 |
| int8_a8_ctx8192_B1 | 19.96 / 19.10 | 17.83–31.66 | 4.83 / 4.51 | 0.242 | 50.1 / 52.3 | 10.62 | 76.96 | 1.0000000 |
| int8_a4w1024_ctx8192_B1 | 20.78 / 20.37 | 18.64–42.43 | 5.78 / 5.83 | 0.278 | 48.1 / 49.1 | 6.80 | 73.13 | 0.9999894 |
| int8_a8_ctx8192_B4 | 29.41 / 26.09 | 22.77–37.02 | 17.77 / 16.76 | 0.604 | 136.0 / 153.3 | 42.50 | 108.84 | 1.0000000 |
| int8_a4w1024_ctx8192_B4 | 31.64 / 30.02 | 26.53–42.87 | 21.75 / 20.72 | 0.688 | 126.4 / 133.2 | 27.19 | 93.52 | 0.9999894 |
| int8_a8_ctx8192_B8 | 45.05 / 41.40 | 39.23–81.92 | 33.78 / 31.76 | 0.750 | 177.6 / 193.2 | 85.00 | 151.34 | 1.0000000 |
| int8_a4w1024_ctx8192_B8 | 56.12 / 51.69 | 46.83–67.29 | 43.58 / 40.51 | 0.777 | 142.6 / 154.8 | 54.38 | 120.71 | 0.9999894 |

Ratios (bytes = ref_total / new_total; speed = ref_ms / new_ms, run1 / run2; > 1 means the new config is better):

| comparison | bytes | step speed | attention speed |
|---|---|---|---|
| u32a4w_vs_int8a8_ctx2048_B1 | 3.68x | 0.77x / 0.73x | 1.06x / 0.97x |
| u32a4w_vs_int8a4w_ctx2048_B1 | 3.65x | 0.80x / 0.74x | 1.29x / 1.11x |
| u32a4w_vs_u32a8_ctx2048_B1 | 1.03x | 1.00x / 1.01x | 0.94x / 0.96x |
| u32a4w_vs_int8a8_ctx2048_B4 | 3.07x | 0.65x / 0.74x | 0.84x / 0.95x |
| u32a4w_vs_int8a4w_ctx2048_B4 | 2.98x | 0.71x / 0.79x | 0.97x / 1.07x |
| u32a4w_vs_u32a8_ctx2048_B4 | 1.09x | 0.92x / 0.96x | 0.87x / 0.90x |
| u32a4w_vs_int8a8_ctx2048_B8 | 2.61x | 0.79x / 0.79x | 0.89x / 0.90x |
| u32a4w_vs_int8a4w_ctx2048_B8 | 2.48x | 0.84x / 0.80x | 0.99x / 0.98x |
| u32a4w_vs_u32a8_ctx2048_B8 | 1.13x | 0.94x / 0.91x | 0.90x / 0.84x |
| u32a4w_vs_int8a8_ctx8192_B1 | 3.28x | 0.75x / 0.73x | 0.88x / 0.82x |
| u32a4w_vs_int8a4w_ctx8192_B1 | 3.12x | 0.78x / 0.78x | 1.06x / 1.06x |
| u32a4w_vs_u32a8_ctx8192_B1 | 1.16x | 0.92x / 0.92x | 0.81x / 0.78x |
| u32a4w_vs_int8a8_ctx8192_B4 | 2.48x | 0.84x / 0.73x | 0.85x / 0.80x |
| u32a4w_vs_int8a4w_ctx8192_B4 | 2.13x | 0.91x / 0.84x | 1.04x / 0.99x |
| u32a4w_vs_u32a8_ctx8192_B4 | 1.35x | 0.87x / 0.85x | 0.79x / 0.79x |
| u32a4w_vs_int8a8_ctx8192_B8 | 2.13x | 0.73x / 0.72x | 0.79x / 0.78x |
| u32a4w_vs_int8a4w_ctx8192_B8 | 1.70x | 0.92x / 0.90x | 1.01x / 0.99x |
| u32a4w_vs_u32a8_ctx8192_B8 | 1.43x | 0.85x / 0.83x | 0.78x / 0.79x |

Takeaways:
- **Memory:** u32 + a4w1024 vs int8 + a8 is 2.13x at B = 8 / ctx 8192 (1.49x at 14:55) and 2.48x at B = 4 / ctx 8192. Against int8 + the same a4w1024 KV it is 1.70x / 2.13x, so the ternary edge stays above 1.5x on equal terms at every tested point.
- **Error:** a4w1024 min layer cos 0.9999973 (ctx 8192) / 0.9999938 (ctx 2048) on u32 weights; a4all is 0.9999970 / 0.9999881, so at ctx ≤ 8192 the window buys little accuracy over all-A4 on this Gaussian history. (int8 weights + a4w1024: 0.9999894 / 0.9999756.)
- **Speed:** numpy A4 unpack (LUT take → int8 → f32 × scale) costs more than A8 dequant, so the smaller KV does not turn into faster attention: 0.78–0.79x at B = 8 / ctx 8192, about 0.9x at ctx 2048. u32-LUT weights stay 0.72–0.79x the step speed of int8 + A8 KV.

## Verdict

**viable** on memory: ≥ 1.5x total-bytes edge for 2-bit weights + windowed A4/A8 KV over int8 weights at every tested point (min 1.70x on equal KV, 2.13x against the 14:55 int8 + A8 control at B = 8 / ctx 8192), layer cos ≥ 0.99999. **not viable on speed**: attention 0.78x and step 0.83–0.85x of all-A8 at B = 8 / ctx 8192 in numpy; the byte win would need a fused nibble-dot kernel (no f32 dequant) to show up as time.

## Script (`/workspace/bench1855_1118/a4win.py`)

```python
#!/usr/bin/env python3
"""A4 g32 nibble KV for old tokens + A8 g64 recent window, multi-stream decode (18:55 11/18). Same layer, weights and
multi-stream step as /workspace/bench1455_1118/mstream.py (base/par/tiled/cache byte-identical). KV formats per stream:
  a8     : all n tokens A8 g64 (int8 + f32 scale per 64-dim group)  [14:55 control]
  a4w1024: tokens older than the last 1024 at A4 g32 (two 4-bit codes per byte, symmetric [-7,7], f32 scale per 32 dims),
           requantised from their A8 g64 values (as they would be when aging out of the window); last 1024 at A8 g64.
           Each step one aged-out token per stream is requantised A8->A4 (counted in step time).
  a4all  : every token A4 g32 (no window), for the error floor.
Attention: per (stream, KV head) online softmax, old region in 1024-key tiles (A4 unpack via 256-entry u16 LUT -> 2 int8),
window as one A8 tile, W=8 pool. Weights: 2-bit u32-LUT (par.fmm) vs int8 rows (par.fmm8).
Error: per-stream layer-delta cos vs the f32 KV whole-context softmax. Usage: OPENBLAS_NUM_THREADS=1 python3 a4win.py out.json"""
import json, sys, time, statistics
import numpy as np
import base, par
from base import rms, silu, D, KVH, QH, HD
TU, W = 256, 8; G = QH // KVH; SC = np.float32(1 / np.sqrt(HD)); NMAX = 8192; TK = 1024; WIN = 1024
CTXS = (2048, 8192); BS = (1, 4, 8); ROUNDS, STEPS, WARM = 5, 8, 2
CFGS = [("u32", "a8"), ("u32", "a4w1024"), ("u32", "a4all"), ("int8", "a8"), ("int8", "a4w1024")]
base.build(); u32_bytes = par.prep(); i8_bytes = par.prep8()
for n in base.PW:
    for f in ("Wd", "Wt", "wm", "ws", "wm8", "ws8", "PT", "q8"): base.PW[n].pop(f, None)
MM = {"u32": par.fmm, "int8": par.fmm8}; WB = {"u32": u32_bytes, "int8": i8_bytes}
LUT16 = np.array([[(c & 15) - 8, (c >> 4) - 8] for c in range(256)], np.int8).view(np.uint16).ravel()
def q8(x, gs=64):
    g = x.reshape(*x.shape[:-1], HD // gs, gs); m = np.abs(g).max(-1) / 127.0 + 1e-12
    return np.clip(np.rint(g / m[..., None]), -127, 127).astype(np.int8).reshape(x.shape), m.astype(np.float32)
def dq8(c, s): return (c.reshape(*c.shape[:-1], HD // 64, 64).astype(np.float32) * s[..., None]).reshape(c.shape)
def q4(x):  # (..., HD) f32 -> (..., HD/2) uint8 nibbles, (..., HD/32) f32
    g = x.reshape(*x.shape[:-1], HD // 32, 32); m = np.abs(g).max(-1) / 7.0 + 1e-12
    c = (np.clip(np.rint(g / m[..., None]), -7, 7).astype(np.int16) + 8).reshape(*x.shape[:-1], HD // 2, 2)
    return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8), m.astype(np.float32)
def dq4(c, s):
    n = c.shape[0]; u = np.take(LUT16, c).view(np.int8).reshape(n, HD // 32, 32)
    return (u.astype(np.float32) * s[:, :, None]).reshape(n, HD)
class KV:
    def __init__(s, k, v, fmt):   # k, v: (KVH, NMAX, HD) f32 history
        s.fmt = fmt; s.n = 0
        if fmt == "f32": s.k, s.v = k.copy(), v.copy(); return
        (s.k, s.ks), (s.v, s.vs) = q8(k), q8(v)
        if fmt != "a8":
            (s.k4, s.k4s), (s.v4, s.v4s) = q4(dq8(s.k, s.ks)), q4(dq8(s.v, s.vs))
            if fmt == "a4all": del s.k, s.ks, s.v, s.vs
    def wb(s): return s.n if s.fmt == "a4all" else max(0, s.n - WIN) if s.fmt == "a4w1024" else s.n
    def nbytes(s, n):
        n8 = n if s.fmt == "a8" else (0 if s.fmt == "a4all" else min(n, WIN)); n4 = n - n8
        return 2 * KVH * (n8 * (HD + HD // 64 * 4) + n4 * (HD // 2 + HD // 32 * 4))
    def append1(s, k, v):
        kh = k.reshape(KVH, 1, HD); vh = v.reshape(KVH, 1, HD); a = s.n
        if s.fmt == "f32": s.k[:, a:a+1] = kh; s.v[:, a:a+1] = vh
        elif s.fmt == "a4all": (s.k4[:, a:a+1], s.k4s[:, a:a+1]), (s.v4[:, a:a+1], s.v4s[:, a:a+1]) = q4(kh), q4(vh)
        else:
            (s.k[:, a:a+1], s.ks[:, a:a+1]), (s.v[:, a:a+1], s.vs[:, a:a+1]) = q8(kh), q8(vh)
            o = a - WIN   # token aging out of the A8 window -> requantise A8 -> A4
            if s.fmt == "a4w1024" and o >= 0:
                (s.k4[:, o:o+1], s.k4s[:, o:o+1]) = q4(dq8(s.k[:, o:o+1], s.ks[:, o:o+1]))
                (s.v4[:, o:o+1], s.v4s[:, o:o+1]) = q4(dq8(s.v[:, o:o+1], s.vs[:, o:o+1]))
        s.n = a + 1
    def tiles(s):
        wb = s.wb(); t = [(a, min(a + TK, wb), 4) for a in range(0, wb, TK)]
        if s.fmt == "a4all": return t
        return t + [(wb, s.n, 8)] if s.fmt == "a4w1024" else [(a, min(a + TK, s.n), 8) for a in range(0, s.n, TK)]
    def get(s, h, a, b, bits):
        if s.fmt == "f32": return s.k[h, a:b], s.v[h, a:b]
        if bits == 4: return dq4(s.k4[h, a:b], s.k4s[h, a:b]), dq4(s.v4[h, a:b], s.v4s[h, a:b])
        return dq8(s.k[h, a:b], s.ks[h, a:b]), dq8(s.v[h, a:b], s.vs[h, a:b])
def head(kv, Q, h):
    m = np.full((G, 1), -np.inf, np.float32); l = np.zeros((G, 1), np.float32); acc = np.zeros((G, HD), np.float32)
    for a, b, bits in kv.tiles():
        K, V = kv.get(h, a, b, bits); S = Q @ K.T
        mn = np.maximum(m, S.max(1, keepdims=True)); al = np.exp(m - mn); S -= mn; np.exp(S, out=S)
        l *= al; l += S.sum(1, keepdims=True); acc *= al; acc += S @ V; m = mn
    return acc / l
def attn(kvs, q):
    B = q.shape[0]; qh = q.reshape(B, KVH, G, HD) * SC
    futs = [par.pool(W).submit(head, kvs[s], qh[s, h], h) for s in range(B) for h in range(KVH)]
    return np.stack([f.result() for f in futs]).reshape(B, D)
def attn_ref(kvs, q):
    B = q.shape[0]; qh = q.reshape(B, KVH, G, HD) * SC; out = np.empty((B, KVH, G, HD), np.float32)
    for s in range(B):
        for h in range(KVH):
            K, V = kvs[s].get(h, 0, kvs[s].n, 32); S = qh[s, h] @ K.T; S -= S.max(1, keepdims=True); np.exp(S, out=S)
            S /= S.sum(1, keepdims=True); out[s, h] = S @ V
    return out.reshape(B, D)
def step(mm, kvs, x, af):
    t0 = time.perf_counter(); h1 = rms(x); q, k, v = mm("qkv", h1, TU, W)
    for s in range(x.shape[0]): kvs[s].append1(k[s], v[s])
    t1 = time.perf_counter(); att = af(kvs, q); t2 = time.perf_counter()
    o = mm("o", att, TU, W)[0]; x2 = x + o; h2 = rms(x2)
    g, u = mm("gu", h2, TU, W); z = (silu(g) * u).astype(np.float32); y = x2 + mm("down", z, TU, W)[0]
    t3 = time.perf_counter(); return y, t2 - t1, t3 - t0
def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def main():
    rng = np.random.default_rng(202611181855); BM = max(BS)
    XS = rng.standard_normal((STEPS, BM, D)).astype(np.float32)
    out = {"weight_bytes": WB, "cfg": {}}
    for wk in MM:
        mm = MM[wk]; kfs = [kf for w_, kf in CFGS if w_ == wk]; streams = []
        for s in range(BM):
            X0 = rng.standard_normal((NMAX, D)).astype(np.float32); K = np.empty((NMAX, KVH * HD), np.float32); V = np.empty_like(K)
            for c0 in range(0, NMAX, 512): _, K[c0:c0+512], V[c0:c0+512] = mm("qkv", rms(X0[c0:c0 + 512]), TU, W)
            K = K.reshape(NMAX, KVH, HD).transpose(1, 0, 2); V = V.reshape(NMAX, KVH, HD).transpose(1, 0, 2)
            streams.append({kf: KV(K, V, kf) for kf in kfs + ["f32"]}); del X0, K, V
        keys = []
        for n in CTXS:
            for B in BS:
                f32 = [streams[s]["f32"] for s in range(B)]
                for c in f32: c.n = n - 1
                ry, _, _ = step(mm, f32, XS[0, :B], attn_ref)
                for kf in kfs:
                    cs = [streams[s][kf] for s in range(B)]
                    for c in cs: c.n = n - 1
                    y, _, _ = step(mm, cs, XS[0, :B], attn)
                    kk = f"{wk}_{kf}_ctx{n}_B{B}"; keys.append(kk)
                    out["cfg"][kk] = dict(w=wk, kv=kf, B=B, ctx=n, kv_bytes=B * cs[0].nbytes(n), attn_ms=[], step_ms=[],
                        layer_cos_min=min(cos(y[s] - XS[0, s], ry[s] - XS[0, s]) for s in range(B)), layer_maxabs=float(np.abs(y - ry).max()))
        for s in range(BM): del streams[s]["f32"]
        for rd in range(ROUNDS):
            r = rd % len(keys)
            for kk in keys[r:] + keys[:r]:
                d = out["cfg"][kk]; cs = [streams[s][d["kv"]] for s in range(d["B"])]
                for i in range(WARM + STEPS):
                    for c in cs: c.n = d["ctx"] - 1
                    _, ta, ts = step(mm, cs, XS[i % STEPS, :d["B"]], attn)
                    if i >= WARM: d["attn_ms"].append(ta * 1e3); d["step_ms"].append(ts * 1e3)
        del streams
    print(f"weight bytes u32 {WB['u32']/2**20:.2f} MiB, int8 {WB['int8']/2**20:.2f} MiB")
    print(f"{'cfg':26s} {'step ms med (min-max)':24s} {'attn med':9s} share  tok/s   kv_MiB  total_MiB  layer_cos_min")
    for kk, d in out["cfg"].items():
        A, S_ = d["attn_ms"], d["step_ms"]; ma, ms = statistics.median(A), statistics.median(S_)
        d.update(step_med=ms, step_min=min(S_), step_max=max(S_), attn_med=ma, attn_min=min(A), attn_max=max(A), share=ma / ms,
                 tok_s=d["B"] * 1e3 / ms, total_bytes=WB[d["w"]] + d["kv_bytes"], n=len(S_))
        print(f"{kk:26s} {ms:7.2f} ({min(S_):.2f}-{max(S_):.2f})    {ma:7.2f}  {ma/ms:.3f}  {d['tok_s']:6.1f}  {d['kv_bytes']/2**20:6.2f}  {d['total_bytes']/2**20:8.2f}   {d['layer_cos_min']:.7f}")
        del d["attn_ms"], d["step_ms"]
    print(f"{'compare':26s} bytes_ratio  speed_ratio(ref_ms/new_ms)  attn_ratio")
    C = out["cfg"]
    for n in CTXS:
        for B in BS:
            g = lambda w, kf: C[f"{w}_{kf}_ctx{n}_B{B}"]
            for name, new, ref in (("u32a4w_vs_int8a8", g("u32", "a4w1024"), g("int8", "a8")), ("u32a4w_vs_int8a4w", g("u32", "a4w1024"), g("int8", "a4w1024")),
                                   ("u32a4w_vs_u32a8", g("u32", "a4w1024"), g("u32", "a8"))):
                r = dict(bytes=ref["total_bytes"] / new["total_bytes"], speed=ref["step_med"] / new["step_med"], attn=ref["attn_med"] / new["attn_med"])
                out.setdefault("compare", {})[f"{name}_ctx{n}_B{B}"] = r
                print(f"{name+f'_ctx{n}_B{B}':26s} {r['bytes']:6.2f}x     {r['speed']:6.2f}x                    {r['attn']:5.2f}x")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
main()
```

## Links

- previous: [[2026-11-18_1455_multi-stream-batched-decode-b1-b4-b8-per-stream-a8-g64-kv-ctx-2048-8192-u32-lut-vs-int8-weights-total-bytes-full-bitnetd-layer]]
- related: [[2026-11-10_1055_hadamard-rotated-kv-a4-g32-k4v4-trit-v-vs-all-a8-attention-512-4096]], [[2026-11-10_1855_rolling-f16-f32-recent-window-plus-a8-g64-kv-vs-all-a8-all-f16-1024-4096]]
- this wave's pair files: none. Quant Chief 18:05 assignment was pair_A no gap and pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-19T01:10:00Z). Roadmap blobs: `JuniorHome docs/ROADMAP.md` 4efb666, `JuniorLLM docs/BETA_TO_OS.md` bac6923.
- bench dir: `/workspace/bench1855_1118` (a4win.py, run1.txt, run2.txt, results_run1.json, results_run2.json)

## Next question

Can the A4 old-token attention avoid the f32 dequant entirely, by scoring Q·K directly on int8-expanded nibble codes with the per-32-dim scales folded into Q (and S·V accumulated per group), so that the 1.56x smaller KV gives ≥ 1.5x faster multi-stream attention at B = 8 / ctx 8192 at the same layer cos?

close: sha256(prev || this)[:16] = d29a3cd3d5961e76
