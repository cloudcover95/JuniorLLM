---
local: true
date: 2026-11-23
wave: "2026-11-23 02:55 (fired 02:42 MT, ran ~02:43–03:00 MT; no pair reports reached chat this wave, quant builders idle)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (K-blocked u32-LUT weight GEMM never beats the current path: best K-blocked kbR256x1280 1.008/0.923x at B = 32, strided K blocks 0.38–0.67x, f32-LUT take 0.47–0.70x; same 2-bit bytes (3.984x vs int8), error unchanged (max abs vs current ≤ 2.4e-06); even the int8 control is only 1.506/1.381x)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, weights, u32-lut, k-blocking, scratch, l2, gemm, b32, numpy, not-viable]
---

# K-blocked u32-LUT 2-bit weight GEMM with per-worker L2-sized f32 unpack scratch, full weight pass at B = 32 (and B = 8): not viable. The best K-blocked layout (kbR256x1280) is 1.008/0.923x of the current u32-LUT path, the strided and f32-LUT variants are 0.38–0.70x, and the only gain is a smaller row tile (rowT128 1.176/1.078x). The int8 control (no LUT) is only 1.506/1.381x, so no unpack-side numpy change can reach 1.5x

The 18:55 note ([[2026-11-22_1855_decode-roofline-f32-copy-vs-int8-cast-vs-scrd-int8-x-scale-dequant-and-v4-u16-take-per-tile-on-8-thread-pool-b32-ctx-8192]]) closed out numpy attention tuning and asked: at B = 32 / ctx 2048, qkv/o/gate-up/down are ~50% of the step. Does a K-blocked u32-LUT weight GEMM with per-worker preallocated f32 unpack scratch (the trick that cleared 1.5x on attention at 05:55 on 11-22) speed up the 2-bit weight pass by ≥ 1.5x vs the current u32-LUT path, at unchanged error and the same 4.00x weight bytes vs int8?

Answers:
- **No.** K blocking makes the weight pass slower. Over a strided PR slice (kb*), it runs at 0.38–0.67x (B = 32 medians): each take reads a non-contiguous (T, KB/4) window, and each extra K block adds a (B, T) partial matmul plus an add. Re-laying PR K-block-major (kbR*, same bytes, contiguous slabs) recovers most of that. The best setting, kbR256x1280 (2 K blocks at in = 2560), is 1.008/0.923x (per-round 0.80–1.15), roughly break-even and nowhere near 1.5x. A direct (256,4) f32 LUT take (one pass, no u32 + cast) is 0.69/0.70x.
- **Preallocated scratch was already harvested here.** par.fmm (05:55 11/16) has given each worker a private u32 + f32 tile since it landed. The 05:55 11-22 attention win came from removing per-call allocation, which this path never had. What remained to test was cache residency: the T = 256 tile is 8.44 MiB per worker, far above the 2 MiB L2, against 0.33–1.59 MiB K-blocked. Fitting the tile in L2 does not help. Per-call numpy and BLAS overhead on the smaller pieces outweighs the cache benefit.
- **Row tile:** rowT128 is 1.176/1.078x at B = 32 and 1.091/1.026x at B = 8, a small, noisy gain (per-round low 0.86). rowT64 is worse (0.957/0.950x).
- **Ceiling:** the int8 control (same pool and tiling, cast only, no LUT, 4x the bytes) is 1.506/1.381x at B = 32 and 1.406/1.556x at B = 8. A perfect zero-cost 2-bit unpack in numpy would still need the copy/cast into an f32 tile plus BLAS, so ~1.4–1.5x is the upper bound for any unpack-side change. Clearing 1.5x on the weight pass needs native code (a fused 2-bit dot product) or a different decomposition.
- **Bytes:** resident 2-bit weights are 16.65 MiB vs 66.34 MiB int8 (3.984x; packed weights alone 4.00x, plus per-row gamma). K-block-major is a re-layout of the same bytes (both copies were held for this bench only). Per-worker scratch drops from 8.44 MiB to 0.33–1.59 MiB, but that is not resident model memory.
- **Error:** max abs vs the current path ≤ 2.4e-06 (summation order only) and vs the float64 trit reference ≤ 1.9e-06; worst cos 1.00000000. The int8 control's 3–4 mab and cos 0.886 against the trit reference mean different weights (int8 absmax of the f32 W, not trits), not an error in the 2-bit path.
- **Verdict: not-viable.** No K-blocked variant reaches 1.5x (or even 1.1x) on speed, and memory is unchanged.

## Method

- Weights: base.build() (bitnetd layer: hidden 2560, 20 q heads × 128, 5 KV heads, FFN 6912, absmean trits, seed 202611141055) + par.prep()/par.prep8() from /workspace/bench1455_1118 (fused sites qkv 2560→3840, o 2560→2560, gate|up 2560→13824, down 6912→2560), unchanged from every B = 32 decode step since 11/21.
- Timed op: one full weight pass of a decode step = the four fused site GEMMs on rms-normalized random (B, in) activations, B = 32 (target) and B = 8, 8-thread pool (W = 8), OPENBLAS_NUM_THREADS=1.
- Variants (same resident PR bytes, same pool): base256 = par.fmm T 256 (baseline); rowT128/rowT64; kb{T}x{KB} = K-blocked over strided PR slices, take + cast into a per-worker (T, KB) scratch, matmul into a per-worker (B, T) partial, then add; kbR{T}x{KB} = the same over a K-block-major copy of PR (contiguous slabs); f4kb = K-blocked with a direct (256,4) f32 LUT take; int8_256 = par.fmm8 control.
- Timing: 5 rounds × (2 warm + 5 timed) passes per variant, variant order rotated each round. Median (min–max) over 25 passes; speedup = base256 median / variant median, per-round range from per-round medians. Two activation seeds (XSEED 1, 2) in separate processes.
- Error: max abs vs base256 output, max abs and min cos vs a float64 x @ (trits·γ)ᵀ reference, worst site.
- 8-core Xeon (2 MiB L2 per core, 320 MiB L3), numpy only, no network, single host.
- Command: `cd /workspace/bench0255_1123 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 kb.py run$s.json > run$s.txt 2>&1; done` (~40 s per seed).

## Results

Full weight pass (qkv + o + gate|up + down), ms, median (min–max) of 25; speedup vs base256 (per-round range); scratch = per-worker unpack scratch.

### B = 32

| variant | what | run1 ms | run2 ms | run1 vs base | run2 vs base | scratch MiB/worker | mab vs base | mab vs f64 | min cos f64 |
|---|---|---|---|---|---|---|---|---|---|
| base256 | current u32-LUT path (par.fmm, T 256, full-width f32 tile) | 20.15 (16.23–32.44) | 18.22 (16.13–20.53) | 1.000 (1.00–1.00) | 1.000 (1.00–1.00) | 8.44 | 0.0e+00 | 1.9e-06 | 1.00000000 |
| rowT128 | same, row tile 128 | 17.13 (14.62–24.90) | 16.90 (14.98–19.91) | 1.176 (0.86–1.38) | 1.078 (0.93–1.15) | 4.22 | 0.0e+00 | 1.9e-06 | 1.00000000 |
| rowT64 | same, row tile 64 | 21.05 (17.84–27.78) | 19.19 (17.21–22.39) | 0.957 (0.78–1.00) | 0.950 (0.84–1.03) | 2.11 | 0.0e+00 | 1.9e-06 | 1.00000000 |
| int8_256 | int8 control (4x bytes, cast only, no LUT) | 13.38 (11.67–18.89) | 13.20 (11.23–17.05) | 1.506 (1.30–1.57) | 1.381 (1.12–1.58) | 8.44 | 3.6e+00 | 3.6e+00 | 0.88613310 |
| kb256x512 | K-blocked, strided PR slice | 29.90 (27.28–39.70) | 27.69 (25.51–31.15) | 0.674 (0.51–0.73) | 0.658 (0.62–0.69) | 0.66 | 2.1e-06 | 1.5e-06 | 1.00000000 |
| kb256x256 | K-blocked, strided | 49.23 (44.12–62.41) | 46.53 (43.21–51.17) | 0.409 (0.35–0.46) | 0.392 (0.36–0.42) | 0.34 | 2.3e-06 | 1.6e-06 | 1.00000000 |
| kb128x512 | K-blocked, strided | 47.83 (44.90–70.90) | 44.88 (43.17–47.37) | 0.421 (0.35–0.47) | 0.406 (0.37–0.43) | 0.33 | 2.1e-06 | 1.5e-06 | 1.00000000 |
| kb128x1024 | K-blocked, strided | 31.87 (29.35–39.96) | 29.61 (27.47–34.43) | 0.632 (0.58–0.73) | 0.615 (0.55–0.65) | 0.64 | 2.1e-06 | 1.7e-06 | 1.00000000 |
| kb64x1024 | K-blocked, strided | 51.28 (45.62–71.76) | 48.34 (45.97–50.47) | 0.393 (0.35–0.42) | 0.377 (0.34–0.40) | 0.32 | 2.1e-06 | 1.7e-06 | 1.00000000 |
| kbR256x512 | K-blocked, K-block-major PR | 28.69 (25.84–38.73) | 28.07 (25.81–32.12) | 0.702 (0.62–0.75) | 0.649 (0.57–0.67) | 0.66 | 2.1e-06 | 1.5e-06 | 1.00000000 |
| kbR256x1280 | K-blocked, K-block-major PR | 19.98 (18.36–21.40) | 19.74 (17.69–26.14) | 1.008 (0.90–1.15) | 0.923 (0.80–0.98) | 1.59 | 2.1e-06 | 1.9e-06 | 1.00000000 |
| kbR128x1280 | K-blocked, K-block-major PR | 25.30 (22.27–38.19) | 23.96 (21.87–27.71) | 0.796 (0.71–0.90) | 0.761 (0.70–0.83) | 0.80 | 2.1e-06 | 1.9e-06 | 1.00000000 |
| kbR512x512 | K-blocked, K-block-major PR | 23.22 (19.28–28.50) | 20.02 (18.81–24.05) | 0.868 (0.72–1.03) | 0.910 (0.83–0.92) | 1.31 | 2.1e-06 | 1.5e-06 | 1.00000000 |
| f4kb256x512 | K-blocked, direct (256,4) f32 LUT take | 29.30 (26.77–34.25) | 26.06 (23.94–29.16) | 0.688 (0.60–0.78) | 0.699 (0.62–0.74) | 0.66 | 2.1e-06 | 1.5e-06 | 1.00000000 |
| f4kb128x512 | K-blocked, f32 LUT take | 42.46 (38.18–50.53) | 37.98 (36.01–53.75) | 0.474 (0.42–0.55) | 0.480 (0.44–0.49) | 0.33 | 2.1e-06 | 1.5e-06 | 1.00000000 |

### B = 8

| variant | what | run1 ms | run2 ms | run1 vs base | run2 vs base | scratch MiB/worker | mab vs base | mab vs f64 | min cos f64 |
|---|---|---|---|---|---|---|---|---|---|
| base256 | current u32-LUT path (par.fmm, T 256, full-width f32 tile) | 14.57 (13.21–17.92) | 15.47 (14.14–20.15) | 1.000 (1.00–1.00) | 1.000 (1.00–1.00) | 8.44 | 0.0e+00 | 1.8e-06 | 1.00000000 |
| rowT128 | same, row tile 128 | 13.36 (12.44–17.66) | 15.07 (12.36–22.13) | 1.091 (0.89–1.14) | 1.026 (0.98–1.12) | 4.22 | 0.0e+00 | 1.8e-06 | 1.00000000 |
| rowT64 | same, row tile 64 | 17.38 (15.55–21.11) | 18.68 (15.99–21.71) | 0.838 (0.70–0.92) | 0.828 (0.77–1.05) | 2.11 | 0.0e+00 | 1.8e-06 | 1.00000000 |
| int8_256 | int8 control (4x bytes, cast only, no LUT) | 10.36 (9.09–14.23) | 9.94 (8.72–13.38) | 1.406 (1.18–1.58) | 1.556 (1.34–1.86) | 8.44 | 3.8e+00 | 3.8e+00 | 0.88606372 |
| kb256x512 | K-blocked, strided PR slice | 26.24 (23.86–36.48) | 27.27 (24.22–33.42) | 0.555 (0.44–0.60) | 0.567 (0.50–0.72) | 0.63 | 2.4e-06 | 1.3e-06 | 1.00000000 |
| kb256x256 | K-blocked, strided | 42.86 (40.57–55.89) | 46.03 (43.37–49.85) | 0.340 (0.31–0.37) | 0.336 (0.33–0.44) | 0.32 | 2.4e-06 | 1.3e-06 | 1.00000000 |
| kb128x512 | K-blocked, strided | 42.95 (39.26–54.16) | 42.51 (39.64–47.24) | 0.339 (0.31–0.37) | 0.364 (0.34–0.47) | 0.32 | 1.9e-06 | 7.4e-07 | 1.00000000 |
| kb128x1024 | K-blocked, strided | 29.49 (26.36–34.91) | 27.93 (26.27–37.98) | 0.494 (0.47–0.53) | 0.554 (0.42–0.68) | 0.63 | 2.4e-06 | 1.5e-06 | 1.00000000 |
| kb64x1024 | K-blocked, strided | 47.51 (44.03–57.95) | 47.89 (44.79–55.10) | 0.307 (0.28–0.32) | 0.323 (0.28–0.43) | 0.31 | 1.9e-06 | 7.3e-07 | 1.00000000 |
| kbR256x512 | K-blocked, K-block-major PR | 26.08 (23.43–31.90) | 26.53 (23.74–28.91) | 0.559 (0.54–0.60) | 0.583 (0.54–0.74) | 0.63 | 2.4e-06 | 1.3e-06 | 1.00000000 |
| kbR256x1280 | K-blocked, K-block-major PR | 16.96 (14.91–39.62) | 17.56 (15.70–33.78) | 0.859 (0.66–0.95) | 0.881 (0.82–0.99) | 1.57 | 2.4e-06 | 1.6e-06 | 1.00000000 |
| kbR128x1280 | K-blocked, K-block-major PR | 21.45 (19.78–27.35) | 23.79 (20.61–26.17) | 0.679 (0.53–0.74) | 0.650 (0.61–0.92) | 0.79 | 2.4e-06 | 1.6e-06 | 1.00000000 |
| kbR512x512 | K-blocked, K-block-major PR | 18.20 (15.35–22.32) | 18.25 (17.17–19.64) | 0.800 (0.77–0.86) | 0.848 (0.81–1.07) | 1.27 | 2.4e-06 | 1.3e-06 | 1.00000000 |
| f4kb256x512 | K-blocked, direct (256,4) f32 LUT take | 23.49 (21.45–28.44) | 25.68 (22.16–27.81) | 0.620 (0.57–0.67) | 0.603 (0.55–0.72) | 0.63 | 2.4e-06 | 1.3e-06 | 1.00000000 |
| f4kb128x512 | K-blocked, f32 LUT take | 35.33 (33.14–44.78) | 38.14 (33.67–46.49) | 0.412 (0.36–0.44) | 0.406 (0.37–0.51) | 0.32 | 1.9e-06 | 7.4e-07 | 1.00000000 |

Resident weights: 2-bit 17,458,176 B (16.65 MiB) vs int8 69,559,296 B (66.34 MiB), 3.984x.

## Pair files this wave

No pair reports reached chat this wave (quant builders idle for many waves). Previous findings: [[2026-11-22_1855_decode-roofline-f32-copy-vs-int8-cast-vs-scrd-int8-x-scale-dequant-and-v4-u16-take-per-tile-on-8-thread-pool-b32-ctx-8192]].

## Inline script (kb.py)

    #!/usr/bin/env python3
    """K-blocked u32-LUT 2-bit weight GEMM with per-worker preallocated cache-sized f32 unpack scratch (02:55 11/23).
    Baseline = par.fmm (05:55 11/16; the weight path inside every B=32 decode step since 11/21): per worker, per T=256 row tile,
    take LUT_U32 over the full input width (T x in/4 u32) -> cast int8 view to a (T, in) f32 tile -> one matmul. The f32 tile is
    2.6 MiB (in=2560) / 7.1 MiB (in=6912) per worker, above the 2 MiB L2. Variants (same resident PR bytes, same pool W=8):
      rowT{T}     = baseline with a smaller row tile (the cheap alternative to K blocking)
      kb{T}x{KB}  = K-blocked: per row tile, loop K blocks of KB inputs; take+cast into a (T, KB) per-worker scratch, matmul against
                    a pre-split contiguous x block into a per-worker (B, T) partial, add into y (tile stays L2-resident)
      kbR{T}x{KB} = K-blocked over a K-block-major copy of the same PR bytes (PRB[j] = PR[:, j*KB/4:(j+1)*KB/4], contiguous) so the
                    take reads a contiguous slab (layout change only; resident bytes identical, the strided-slice variant above keeps PR)
      f4kb{T}x{KB}= K-blocked with a direct (256,4) f32 LUT take (one pass byte -> 4 f32, no u32 + cast)
    Timed op = one full weight pass of a decode step at B: qkv(2560->3840) + o(2560->2560) + gate|up(2560->13824) + down(6912->2560).
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=... python3 kb.py out.json"""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par, tiled
    from base import D, F
    W = 8; ROUNDS, REPS, WARM = 5, 5, 2
    BS = [32, 8]
    LUT_U32 = par.LUT_U32; LUT4 = tiled.LUT4
    base.build(); WB = par.prep(); WB8 = par.prep8()
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT"): base.PW[nm].pop(f, None)
    SITES = [("qkv", D), ("o", D), ("gu", D), ("down", F)]
    _ks = {}
    def kscr(w, T, KB, B):
        k = (w, T, KB, B)
        if k not in _ks: _ks[k] = (np.empty((T, KB // 4), np.uint32), np.empty((T, KB), np.float32), np.empty((B, T), np.float32))
        return _ks[k]
    _prb = {}
    def prb(key, KB):
        if (key, KB) not in _prb:
            PR = par.FP[key]["PR"]; qb = KB // 4
            _prb[(key, KB)] = [np.ascontiguousarray(PR[:, c:c + qb]) for c in range(0, PR.shape[1], qb)]
        return _prb[(key, KB)]
    def run_kbR(PB, xb, y, r0, r1, T, KB, w):
        B = y.shape[0]; s8, s, part = kscr(w, T, KB, B)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0); yt = y[:, o0:o0+n]
            for j, xj in enumerate(xb):
                m = xj.shape[1]; q = m // 4
                np.take(LUT_U32, PB[j][o0:o0+n], out=s8[:n, :q])
                np.copyto(s[:n, :m], s8[:n, :q].view(np.int8), casting="unsafe")
                if j == 0: np.matmul(xj, s[:n, :m].T, out=yt)
                else: np.matmul(xj, s[:n, :m].T, out=part[:, :n]); yt += part[:, :n]
    def fmm_kbR(key, x, T, KB):
        P = par.FP[key]; PB = prb(key, KB); out = P["PR"].shape[0]; i = P["PR"].shape[1] * 4
        xb = [np.ascontiguousarray(x[:, k0:k0+KB]) for k0 in range(0, i, KB)]
        y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
        futs = [par.pool(W).submit(run_kbR, PB, xb, y, w * per * T, min(out, (w + 1) * per * T), T, KB, w) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
        y *= P["g"]; return y
    def run_kb(PR, xb, y, r0, r1, T, KB, w, f4):
        B = y.shape[0]; s8, s, part = kscr(w, T, KB, B); qb = KB // 4
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0); yt = y[:, o0:o0+n]; first = True
            for j, xj in enumerate(xb):
                c0 = j * qb; m = xj.shape[1]; c1 = c0 + m // 4
                if f4: np.take(LUT4, PR[o0:o0+n, c0:c1], axis=0, out=s[:n, :m].reshape(n, m // 4, 4))
                else:
                    np.take(LUT_U32, PR[o0:o0+n, c0:c1], out=s8[:n, :m // 4])
                    np.copyto(s[:n, :m], s8[:n, :m // 4].view(np.int8), casting="unsafe")
                if first: np.matmul(xj, s[:n, :m].T, out=yt); first = False
                else:
                    np.matmul(xj, s[:n, :m].T, out=part[:, :n]); yt += part[:, :n]
    def fmm_kb(key, x, T, KB, f4=False):
        P = par.FP[key]; PR = P["PR"]; out = PR.shape[0]; i = PR.shape[1] * 4
        xb = [np.ascontiguousarray(x[:, k0:k0+KB]) for k0 in range(0, i, KB)]
        y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
        futs = [par.pool(W).submit(run_kb, PR, xb, y, w * per * T, min(out, (w + 1) * per * T), T, KB, w, f4) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
        y *= P["g"]; return y
    def fmm_base(key, x, T): return np.concatenate(par.fmm(key, x, T, W), axis=1)
    def fmm_i8(key, x, T): return np.concatenate(par.fmm8(key, x, T, W), axis=1)
    V = {"base256": lambda k, x: fmm_base(k, x, 256), "rowT128": lambda k, x: fmm_base(k, x, 128), "rowT64": lambda k, x: fmm_base(k, x, 64),
         "int8_256": lambda k, x: fmm_i8(k, x, 256)}
    for T, KB in [(256, 512), (256, 256), (128, 512), (128, 1024), (64, 1024)]:
        V[f"kb{T}x{KB}"] = (lambda T, KB: lambda k, x: fmm_kb(k, x, T, KB))(T, KB)
    for T, KB in [(256, 512), (256, 1280), (128, 1280), (512, 512)]:
        V[f"kbR{T}x{KB}"] = (lambda T, KB: lambda k, x: fmm_kbR(k, x, T, KB))(T, KB)
    for T, KB in [(256, 512), (128, 512)]:
        V[f"f4kb{T}x{KB}"] = (lambda T, KB: lambda k, x: fmm_kb(k, x, T, KB, True))(T, KB)
    def wref(key):  # float64 trit reference W1.58 (gamma folded), same rows as FP[key]
        return np.concatenate([base.PW[n]["Wt"].astype(np.float64) for n in par.FUSE[key]])
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    def main():
        rng = np.random.default_rng(int(os.environ.get("XSEED", "1"))); out = {"xseed": int(os.environ.get("XSEED", "1")), "w_bytes_2bit": WB, "w_bytes_int8": WB8, "B": {}}
        print(f"xseed {out['xseed']} | resident 2-bit {WB/2**20:.2f} MiB int8 {WB8/2**20:.2f} MiB ({WB8/WB:.3f}x) | W {W} rounds {ROUNDS} x reps {REPS}", flush=True)
        for B in BS:
            X = {k: base.rms(rng.standard_normal((B, i)).astype(np.float32)) for k, i in SITES}
            REF = {k: X[k].astype(np.float64) @ wref(k).T for k, _ in SITES}
            Y0 = {k: V["base256"](k, X[k]) for k, _ in SITES}
            res = {}
            for nm, fn in V.items():
                Y = {k: fn(k, X[k]) for k, _ in SITES}
                err = dict(mab_vs_base=max(float(np.abs(Y[k] - Y0[k]).max()) for k, _ in SITES),
                           mab_vs_f64=max(float(np.abs(Y[k] - REF[k]).max()) for k, _ in SITES),
                           cos_f64=min(cos(Y[k], REF[k]) for k, _ in SITES))
                res[nm] = dict(t=[], **err)
            names = list(V)
            for rd in range(ROUNDS):
                for nm in names[rd % len(names):] + names[:rd % len(names)]:
                    fn = V[nm]
                    for r in range(WARM + REPS):
                        t0 = time.perf_counter()
                        for k, _ in SITES: fn(k, X[k])
                        dt = (time.perf_counter() - t0) * 1e3
                        if r >= WARM: res[nm]["t"].append(dt)
            bm = [float(np.median(res["base256"]["t"][r * REPS:(r + 1) * REPS])) for r in range(ROUNDS)]
            for nm, d in res.items():
                T = np.array(d["t"]); rm = [float(np.median(T[r * REPS:(r + 1) * REPS])) for r in range(ROUNDS)]
                sp = [a / b for a, b in zip(bm, rm)]
                d.update(med=float(np.median(T)), lo=float(T.min()), hi=float(T.max()), iqr=float(np.percentile(T, 75) - np.percentile(T, 25)),
                         speedup=float(np.median(res["base256"]["t"])) / float(np.median(T)), sp_lo=min(sp), sp_hi=max(sp), n=len(T))
                print(f"B{B:<3d} {nm:12s} {d['med']:7.2f} ms ({d['lo']:.2f}-{d['hi']:.2f}) iqr {d['iqr']:5.2f} | vs base256 {d['speedup']:.3f}x (rounds {d['sp_lo']:.2f}-{d['sp_hi']:.2f}) | mab vs base {d['mab_vs_base']:.1e} vs f64 {d['mab_vs_f64']:.1e} cos {d['cos_f64']:.8f}", flush=True)
            out["B"][str(B)] = res
        out["scratch_bytes_kb_per_worker"] = {f"{k[1]}x{k[2]}_B{k[3]}": sum(a.nbytes for a in v) for k, v in _ks.items() if k[0] == 0}
        out["scratch_bytes_base_per_worker"] = par.live_scratch_bytes() // max(1, len(par._scr))
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Next question

The numpy unpack lever on the weight side is exhausted: even int8 with no LUT is only ~1.4–1.5x. The remaining question is amortization. With one shared 2-bit unpack per step, at B = 32 / 64 / 128, how does the base256 weight pass split between LUT take, int8→f32 cast and BLAS matmul? Does growing B from 32 to 128 raise aggregate weight-pass tokens/s by ≥ 1.5x while the 2-bit/int8 time gap drops below 1.1x, at the same 4.00x weight bytes and with KV kept at ctx 2048?

close: 89aa41d4008752cb
