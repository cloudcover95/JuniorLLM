---
local: true
date: 2026-11-11
wave: "2026-11-11 14:55 (ran ~14:50–15:00 MT; Chief last_run 2026-11-11T21:13:00Z ~14:13 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, w158, i2s, lut-expand, shift-mask, gather-free, row-block, numpy, not-viable]
---

# Gather-free shift/mask decode of the blk32 packed W1.58 layer is 0.60–0.64x of the LUT `take` it was meant to replace, so the 256-entry LUT gather stays the fastest numpy expand

## Question

The prior note [[2026-11-11_1055_threaded-blk32-packed-w158-layer-b1-pool-1-8-vs-int8-threaded]] showed threads do not help the blk32 packed layer and asked: does a gather-free expand (decode the 2-bit codes with vectorized shifts and masks, no LUT `take`) cut the blk32 layer below 0.67x of its ~29–34 ms, so packed W1.58 beats int8-resident by at least 1.5x at B=1 single-threaded while keeping the 3.8x memory win? Quant Chief `last_tips.json` `last_run` 2026-11-11T21:13:00Z (~14:13 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same layer as the prior two notes: B=1, hidden 2560, FFN 6912 (gate, up, down), 20 query heads × 128, KV heads 20 (MHA) and 5 (GQA), assumed shapes. Absmean ternary weights, one f32 scale per matrix, 2-bit codes (trit+1) four per byte; the layer time is the sum of the seven GEMVs. All forms are single-threaded Python loops over 32-row blocks into one reused scratch (OpenBLAS left at default threads inside `matmul`).

- `blk32_lut`: prior winner; `take` from a 256×4 int8 LUT, cast to f32 scratch, `matmul`.
- `blk32_lutf`: `take` straight from a 256×4 f32 LUT into the f32 scratch (skips the cast pass).
- `blk32_shift_u8`: broadcast `right_shift(P[...,None], [0,2,4,6])` and `& 3` into uint8 scratch, cast to f32, subtract 1, `matmul`.
- `blk32_shift_ux`: same, but the −1 is folded out of the expand: `W·x = C·x − sum(x)`.
- `blk32_shift_u32`: decode on a uint32 view of each block, 4 shift passes with mask `0x03030303` (four bytes per op), strided byte store into the uint8 scratch, cast, `matmul`, `− sum(x)`.
- `int8`: int8-resident trits, cast + GEMV. `f32`: pre-scaled f32-resident GEMV (memory-heavy ceiling).

numpy 2.5.4 (scipy-openblas 0.3.34), 8-core shared box (load ~0.6–1.0 at start), seed `202611111455`, 2 warm-ups, 7 repeats × 3 passes, median and min–max, run three times back to back. Error is max |y − ref| / max |ref| against an f64 GEMV on the same trits.

Commands, from `/workspace/bench1455_1111`: `python3 bench.py results_run1.json`, `python3 bench.py results_run2.json`, `python3 bench.py results_run3.json`.

```python
# Full bitnetd-shaped layer at B=1 (hidden 2560, FFN 6912, 20 q heads x 128, KV heads 20 and 5), single-threaded:
# blk32 row-blocked expand of W1.58 2-bit packed weights, LUT gather (prior winner) vs gather-free shift/mask decode
# variants that write trits straight into a reused scratch, vs int8-resident (cast+GEMV) and f32-resident GEMV.
import numpy as np, time, json, sys
SEED = 202611111455
Hd, F, D, BS = 2560, 6912, 128, 32
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
LUT = np.array([[((b >> (2 * k)) & 3) for k in range(4)] for b in range(256)], dtype=np.int8) - 1
LUTF = LUT.astype(np.float32)
SH = np.array([0, 2, 4, 6], np.uint8)
def tern(o, i):
    W = rng.standard_normal((o, i)).astype(np.float32); s = np.abs(W).mean()
    return np.clip(np.rint(W / s), -1, 1).astype(np.int8), np.float32(s)
def pack(T):  # code = trit+1, 4 codes per byte, little-endian 2-bit fields
    c = (T + 1).astype(np.uint8).reshape(T.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
class Scr:
    def __init__(s):
        q = F // 4
        s.u8 = np.empty((BS, q, 4), np.uint8); s.i8 = s.u8.view(np.int8)
        s.f = np.empty((BS, F), np.float32); s.f4 = s.f.reshape(BS, q, 4)
        s.w32 = np.empty((BS, q // 4), np.uint32)
        s.u8b = np.empty((BS, q, 4), np.uint8)
SC = Scr()
def blk_lut(P, s, x, y):  # prior note: take(LUT int8) -> cast f32 -> matmul
    q = P.shape[1]; i = q * 4
    for r in range(0, P.shape[0], BS):
        n = min(BS, P.shape[0] - r)
        b8 = SC.i8[:n, :q]; np.take(LUT, P[r:r + n], axis=0, out=b8)
        bf = SC.f[:n, :i]; np.copyto(bf, b8.reshape(n, i), casting='unsafe')
        np.matmul(bf, x, out=y[r:r + n])
    y *= s
def blk_lutf(P, s, x, y):  # take straight from f32 LUT (no cast pass)
    q = P.shape[1]; i = q * 4
    for r in range(0, P.shape[0], BS):
        n = min(BS, P.shape[0] - r)
        bf = SC.f4[:n, :q]; np.take(LUTF, P[r:r + n], axis=0, out=bf)
        np.matmul(SC.f[:n, :i], x, out=y[r:r + n])
    y *= s
def blk_shift_u8(P, s, x, y):  # gather-free: broadcast shift/mask uint8 -> (code) ; subtract 1 in the f32 cast
    q = P.shape[1]; i = q * 4
    for r in range(0, P.shape[0], BS):
        n = min(BS, P.shape[0] - r)
        u = SC.u8[:n, :q]
        np.right_shift(P[r:r + n, :, None], SH, out=u); np.bitwise_and(u, 3, out=u)
        bf = SC.f[:n, :i]; np.copyto(bf, u.reshape(n, i), casting='unsafe'); bf -= 1.0
        np.matmul(bf, x, out=y[r:r + n])
    y *= s
def blk_shift_ux(P, s, x, y):  # gather-free, fold the -1 into the GEMV: W x = C x - sum(x)
    q = P.shape[1]; i = q * 4; sx = np.float32(x.sum())
    for r in range(0, P.shape[0], BS):
        n = min(BS, P.shape[0] - r)
        u = SC.u8[:n, :q]
        np.right_shift(P[r:r + n, :, None], SH, out=u); np.bitwise_and(u, 3, out=u)
        bf = SC.f[:n, :i]; np.copyto(bf, u.reshape(n, i), casting='unsafe')
        np.matmul(bf, x, out=y[r:r + n])
    y -= sx; y *= s
def blk_shift_u32(P, s, x, y):  # gather-free on a uint32 view: 4 shift/mask passes of 4 bytes at once, then byte-scatter
    q = P.shape[1]; i = q * 4; sx = np.float32(x.sum()); P32 = P.view(np.uint32)
    for r in range(0, P.shape[0], BS):
        n = min(BS, P.shape[0] - r)
        u = SC.u8[:n, :q]; uv = u.reshape(n, q // 4, 4, 4)  # [word, byte-in-word, field]
        w = SC.w32[:n, :q // 4]
        src = P32[r:r + n]
        for k in range(4):
            np.right_shift(src, np.uint32(2 * k), out=w); np.bitwise_and(w, np.uint32(0x03030303), out=w)
            uv[:, :, :, k] = w.view(np.uint8).reshape(n, q // 4, 4)
        bf = SC.f[:n, :i]; np.copyto(bf, u.reshape(n, i), casting='unsafe')
        np.matmul(bf, x, out=y[r:r + n])
    y -= sx; y *= s
def i8(W8, s, x, y): np.matmul(W8.astype(np.float32), x, out=y); y *= s
def f32(Wf, s, x, y): np.matmul(Wf, x, out=y)
def bench(f, *a):
    for _ in range(WARM): f(*a)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for _ in range(PASS): f(*a)
        ts.append((time.perf_counter() - t0) / PASS * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))
forms = {"f32": f32, "int8": i8, "blk32_lut": blk_lut, "blk32_lutf": blk_lutf, "blk32_shift_u8": blk_shift_u8,
         "blk32_shift_ux": blk_shift_ux, "blk32_shift_u32": blk_shift_u32}
scr_bytes = {"blk32_lut": BS * F * 5, "blk32_lutf": BS * F * 4, "blk32_shift_u8": BS * F * 5,
             "blk32_shift_ux": BS * F * 5, "blk32_shift_u32": BS * F * 5 + BS * F // 4}
out = {"seed": SEED, "numpy": np.__version__, "layers": {}}
x = rng.standard_normal(Hd).astype(np.float32); xf = rng.standard_normal(F).astype(np.float32)
for kvh in (20, 5):
    shapes = {"q": (Hd, Hd), "k": (kvh * D, Hd), "v": (kvh * D, Hd), "o": (Hd, Hd), "gate": (F, Hd), "up": (F, Hd), "down": (Hd, F)}
    acc = {k: [0.0, 0.0, 0.0] for k in forms}; by = {k: 0 for k in forms}; err = {k: 0.0 for k in forms}
    for nm, (o, i) in shapes.items():
        T8, s = tern(o, i); P = pack(T8); Wf = T8.astype(np.float32) * s; v = xf if i == F else x
        ref = (T8.astype(np.float64) @ v.astype(np.float64)) * float(s); y = np.empty(o, np.float32)
        for k, f in forms.items():
            W = T8 if k == "int8" else (Wf if k == "f32" else P)
            m, lo, hi = bench(f, W, s, v, y)
            acc[k] = [acc[k][0] + m, acc[k][1] + lo, acc[k][2] + hi]; by[k] += W.nbytes
            f(W, s, v, y); err[k] = max(err[k], float(np.max(np.abs(y - ref)) / np.max(np.abs(ref))))
        del Wf, T8, P
    out["layers"][str(kvh)] = dict(ms=acc, bytes=by, scratch={k: scr_bytes.get(k, 0) for k in forms}, max_rel_err=err)
    print("kvh", kvh, flush=True)
    for k in forms:
        print(f"  {k:16s} {acc[k][0]:9.3f} ms [{acc[k][1]:.3f}-{acc[k][2]:.3f}] {by[k]/2**20:7.2f} MiB scratch {scr_bytes.get(k,0)/2**20:5.2f} MiB err {err[k]:.1e} vs_int8 {acc['int8'][0]/acc[k][0]:.2f}x vs_lut {acc['blk32_lut'][0]/acc[k][0]:.2f}x", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Seven-matrix layer weight stage, ms median for run2 / run3 (run3 min–max); run1 agrees within 3% on every packed form, but its 5-KV-head `int8` row had a 36.8–104.1 ms spread (median 53.3 ms) from box contention, so run1 is left out of the int8 ratios.

| KV heads | form | resident MiB | scratch MiB | ms run2 / run3 (run3 min–max) | vs blk32_lut (run2 / run3) | vs int8 (run2 / run3) | max rel err |
|---|---|---|---|---|---|---|---|
| 20 | f32 | 302.50 | 0 | 1.34 / 1.60 (1.57–1.89) | n/a | 30.2x / 21.5x | 3.5e-07 |
| 20 | int8 | 75.62 | 0 | 40.53 / 34.40 (31.72–41.72) | 0.77x / 0.91x | 1.00x | 3.9e-07 |
| 20 | **blk32_lut** | 18.91 | 1.05 | **31.29 / 31.27** (30.70–33.64) | 1.00x | **1.30x / 1.10x** | 3.9e-07 |
| 20 | blk32_lutf | 18.91 | 0.84 | 40.09 / 39.55 (38.43–40.97) | 0.78x / 0.79x | 1.01x / 0.87x | 3.9e-07 |
| 20 | blk32_shift_u8 | 18.91 | 1.05 | 232.70 / 236.08 (231.56–242.19) | 0.13x / 0.13x | 0.17x / 0.15x | 3.9e-07 |
| 20 | blk32_shift_ux | 18.91 | 1.05 | 214.68 / 220.66 (215.07–231.30) | 0.15x / 0.14x | 0.19x / 0.16x | 7.0e-07 |
| 20 | blk32_shift_u32 | 18.91 | 1.11 | 51.40 / 52.13 (50.68–55.03) | 0.61x / 0.60x | 0.79x / 0.66x | 7.0e-07 |
| 5 | f32 | 265.00 | 0 | 1.12 / 1.87 (1.67–2.18) | n/a | 30.4x / 16.7x | 3.7e-07 |
| 5 | int8 | 66.25 | 0 | 33.90 / 31.18 (28.98–43.57) | 0.79x / 0.88x | 1.00x | 3.8e-07 |
| 5 | **blk32_lut** | 16.56 | 1.05 | **26.83 / 27.33** (26.88–27.96) | 1.00x | **1.26x / 1.14x** | 3.8e-07 |
| 5 | blk32_lutf | 16.56 | 0.84 | 33.94 / 34.83 (33.58–37.62) | 0.79x / 0.78x | 1.00x / 0.90x | 3.8e-07 |
| 5 | blk32_shift_u8 | 16.56 | 1.05 | 204.54 / 208.06 (202.62–217.92) | 0.13x / 0.13x | 0.17x / 0.15x | 3.8e-07 |
| 5 | blk32_shift_ux | 16.56 | 1.05 | 191.47 / 197.57 (188.85–203.91) | 0.14x / 0.14x | 0.18x / 0.16x | 7.0e-07 |
| 5 | blk32_shift_u32 | 16.56 | 1.11 | 44.68 / 45.71 (44.66–47.50) | 0.60x / 0.60x | 0.76x / 0.68x | 7.0e-07 |

Resident packed weights are 4.00x smaller than int8 (3.79x with the ~1 MiB scratch) for every packed form. The broadcast uint8 shift is 7–8x slower than the LUT because numpy's 3-D broadcast `right_shift` with a stride-0 source runs an inner loop of length 4; the uint32 view removes that (4 full-length passes of 4 bytes per op) but the strided byte store back into the interleaved uint8 scratch costs more than one contiguous `take`, leaving it at 0.60–0.61x of the LUT. Folding the −1 into `− sum(x)` saves only ~7% and doubles the relative error to 7.0e-07 (still float32-noise level). Taking straight from an f32 LUT is slower (0.78–0.79x) because the f32 gather moves 4x the bytes of the int8 gather plus cast. The blk32 LUT result reproduces the prior notes (26.8–31.3 ms, 1.10–1.30x of int8-resident).

## Verdict

**Not viable: gather-free shift/mask decode does not beat the LUT.** The best gather-free form (`blk32_shift_u32`) is 0.60–0.61x of `blk32_lut` and 0.66–0.79x of int8-resident, far from the 0.67x-of-current target and the 1.5x bar. In numpy the single contiguous 256-entry int8 LUT `take` is the cheapest expand; keep `blk32_lut` (1.10–1.30x of int8 at 3.79x fewer bytes) as the packed B=1 path. Closing the remaining gap to 1.5x needs work numpy cannot express in one pass (a fused decode-and-dot kernel in C or SIMD), which is outside this lane's numpy-only budget.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-11T21:13:00Z (~14:13 MT) wrote pair_A/pair_B **no gap** ("14:05 11/11: roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `25c653b` (15:18Z) re-advances landed ROADMAP #5 UE5, left to LLC. JuniorLLM tip at clone time: [8f9b0ef](https://github.com/cloudcover95/JuniorLLM/commit/8f9b0ef) (the prior findings note).

## Next question

Since the expand cost is now pinned at the LUT `take`, does skipping expand for the activation side help instead: at B=1, does grouping the f32 input into 4-element chunks and precomputing all 81 signed partial sums per chunk (a T-MAC-style activation LUT, `x4·{-1,0,1}^4`), then gathering by 8-bit trit-pair index from the packed weights, beat `blk32_lut` by at least 1.5x on the 2560×6912 FFN matrices while keeping the 4.00x weight-memory win?

close: f74f450456d01b9f (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-11_1055)
