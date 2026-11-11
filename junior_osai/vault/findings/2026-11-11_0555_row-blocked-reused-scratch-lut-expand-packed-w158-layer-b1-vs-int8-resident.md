---
local: true
date: 2026-11-11
wave: "2026-11-11 05:55 (ran ~05:55–06:05 MT; Chief last_run 2026-11-11T12:08:00Z ~05:08 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, w158, i2s, lut-expand, row-block, scratch, numpy, viable-memory, speed-parity]
---

# Expanding packed W1.58 weights in 32-row blocks into one reused scratch buffer makes the full bitnetd layer 7.6x faster than the per-step full expand and 1.06–1.12x of int8-resident speed, while keeping 3.79x fewer resident bytes than int8

## Question

The prior note [[2026-11-11_0255_full-bitnetd-layer-b1-weights-vs-a8-g64-kv-bytes-and-step-share-8gib-crossover]] found the packed W1.58 weights keep the model 2.4–3.8x smaller than int8 out to T=4096, but the per-step full-matrix LUT expand made the weight stage 0.14x of int8-resident speed. Its next question: does expanding each packed matrix in row blocks into one reused int8 scratch buffer (no per-step full-matrix allocation, block 64–1024 rows) bring the seven-matrix packed layer at B=1 within 1.5x of int8-resident step time while keeping the 4.00x resident-byte win? Quant Chief `last_tips.json` `last_run` 2026-11-11T12:08:00Z (~05:08 MT) wrote pair_A/pair_B **no gap**, so this runs that question. Blocks 16 and 32 were added after a first pass showed the smallest block was fastest.

## Method

One decoder layer, B=1, hidden 2560, FFN 6912 (gate, up, down), 20 query heads × 128, KV heads 20 (MHA) and 5 (GQA); both are assumed shapes as in the prior note. Absmean ternary weights, one f32 scale per matrix, timed as the sum of the seven GEMVs. Forms:

- `full_expand`: the prior note's path (LUT gather of the whole packed matrix, cast to f32, GEMV).
- `blkN`: per N-row block, `np.take` the 256×4 int8 LUT into a reused int8 scratch, `copyto` into a reused f32 scratch, `matmul` into the output slice; scratch is allocated once per layer at N×6912.
- `blkfN`: same, gathering straight from a 256×4 f32 LUT (skips the int8 hop).
- `int8`: one int8 per trit resident, cast to f32 then GEMV. `f32`: pre-scaled f32 resident, a memory-heavy ceiling.

2 warm-ups, 7 repeats × 3 passes, median and min–max, run twice back to back on the shared 8-core box, numpy 2.5.4, seed `202611110555`. Error is max |y − ref| / max |ref| against an f64 GEMV on the same trits.

Commands, from `/workspace/bench0555_1111`: `python3 bench.py results_run1.json`, then `python3 bench.py results_run2.json`.

```python
# Full bitnetd-shaped layer at B=1 (hidden 2560, FFN 6912, 20 q heads x 128, KV heads 20 and 5): W1.58 2-bit packed weights
# expanded per step in row blocks into ONE reused int8 scratch + reused f32 scratch (no per-step full-matrix allocation),
# block 64..1024 rows, vs full per-step LUT expand (prior note), int8-resident (cast+GEMV) and f32-resident GEMV.
import numpy as np, time, json, sys
SEED = 202611110555
Hd, F, D = 2560, 6912, 128
REP, PASS, WARM = 7, 3, 2
BLOCKS = [16, 32, 64, 128, 256, 512, 1024]
rng = np.random.default_rng(SEED)
LUT = np.array([[((b >> (2 * k)) & 3) for k in range(4)] for b in range(256)], dtype=np.int8) - 1
LUTf = LUT.astype(np.float32)
def tern(o, i):
    W = rng.standard_normal((o, i)).astype(np.float32); s = np.abs(W).mean()
    return np.clip(np.rint(W / s), -1, 1).astype(np.int8), np.float32(s)
def pack(T):
    c = (T + 1).astype(np.uint8).reshape(T.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
def full_expand(P, s, x, y):
    W = LUT[P].reshape(P.shape[0], -1); y[:] = (W.astype(np.float32) @ x) * s
def make_blocked(bs, maxin):
    s8 = np.empty((bs, maxin // 4, 4), np.int8); sf = np.empty((bs, maxin), np.float32)
    def f(P, s, x, y):
        o, q = P.shape; i = q * 4
        for r in range(0, o, bs):
            n = min(bs, o - r)
            b8 = s8[:n, :q]; np.take(LUT, P[r:r + n], axis=0, out=b8)
            bf = sf[:n, :i]; np.copyto(bf, b8.reshape(n, i), casting='unsafe')
            np.matmul(bf, x, out=y[r:r + n])
        y *= s
    return f
def make_blocked_f32lut(bs, maxin):  # expand straight to f32 via f32 LUT (skips int8 hop)
    sf = np.empty((bs, maxin // 4, 4), np.float32)
    def f(P, s, x, y):
        o, q = P.shape; i = q * 4
        for r in range(0, o, bs):
            n = min(bs, o - r)
            bf = sf[:n, :q]; np.take(LUTf, P[r:r + n], axis=0, out=bf)
            np.matmul(bf.reshape(n, i), x, out=y[r:r + n])
        y *= s
    return f
def i8(W8, s, x, y): y[:] = (W8.astype(np.float32) @ x) * s
def f32(Wf, s, x, y): np.matmul(Wf, x, out=y)
def bench(f, *a):
    for _ in range(WARM): f(*a)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for _ in range(PASS): f(*a)
        ts.append((time.perf_counter() - t0) / PASS * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))
out = {"seed": SEED, "numpy": np.__version__, "layers": {}}
x = rng.standard_normal(Hd).astype(np.float32); xf = rng.standard_normal(F).astype(np.float32)
for kvh in (20, 5):
    shapes = {"q": (Hd, Hd), "k": (kvh * D, Hd), "v": (kvh * D, Hd), "o": (Hd, Hd), "gate": (F, Hd), "up": (F, Hd), "down": (Hd, F)}
    forms = {"full_expand": full_expand, "int8": i8, "f32": f32}
    for bs in BLOCKS:
        forms[f"blk{bs}"] = make_blocked(bs, F); forms[f"blkf{bs}"] = make_blocked_f32lut(bs, F)
    scratch = {k: (bs * F // 4 * 4 + bs * F * 4 if k.startswith("blk") and not k.startswith("blkf") else (bs * F * 4 if k.startswith("blkf") else 0)) for k, bs in [(k, int(k.lstrip("blkf")) if k.startswith("blk") else 0) for k in forms]}
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
    out["layers"][str(kvh)] = dict(ms=acc, bytes=by, scratch=scratch, max_rel_err=err)
    print("kvh", kvh, flush=True)
    for k in forms:
        print(f"  {k:12s} {acc[k][0]:9.3f} ms [{acc[k][1]:.3f}-{acc[k][2]:.3f}] {by[k]/2**20:7.2f} MiB scratch {scratch[k]/2**20:6.2f} MiB err {err[k]:.1e} vs_int8 {acc['int8'][0]/acc[k][0]:.2f}x", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Seven-matrix layer weight stage, ms median run1 / run2 (run2 min–max). Max relative error is 3.9e-07 (20 KV heads) and 3.6e-07 (5 KV heads) for every packed and int8 form, since all hold the same trits.

| KV heads | form | resident MiB | scratch MiB | ms run1 / run2 (run2 min–max) | speed vs int8 (run1 / run2) | resident bytes vs int8 |
|---|---|---|---|---|---|---|
| 20 | full_expand | 18.91 | 0 (79M-elem temp per step) | 244.62 / 247.23 (240.40–256.75) | 0.15x / 0.14x | 4.00x smaller |
| 20 | int8 | 75.62 | 0 | 35.70 / 33.76 (31.27–41.88) | 1.00x | 1.00x |
| 20 | f32 | 302.50 | 0 | 1.42 / 1.36 (1.30–1.85) | 25.2x / 24.8x | 4.00x larger |
| 20 | blk16 | 18.91 | 0.53 | 36.40 / 35.99 (34.30–39.27) | 0.98x / 0.94x | 3.89x smaller |
| 20 | **blk32** | 18.91 | 1.05 | **32.44 / 31.88** (31.21–33.88) | **1.10x / 1.06x** | **3.79x smaller** |
| 20 | blk64 | 18.91 | 2.11 | 34.71 / 34.03 (33.44–36.82) | 1.03x / 0.99x | 3.60x smaller |
| 20 | blk128 | 18.91 | 4.22 | 44.56 / 44.09 | 0.80x / 0.77x | 3.27x smaller |
| 20 | blk256 | 18.91 | 8.44 | 57.64 / 56.19 | 0.62x / 0.60x | 2.77x smaller |
| 20 | blk1024 | 18.91 | 33.75 | 57.30 / 57.66 | 0.62x / 0.59x | 1.43x smaller |
| 20 | blkf32 (best f32-LUT) | 18.91 | 0.84 | 41.47 / 41.41 | 0.86x / 0.82x | 3.83x smaller |
| 5 | full_expand | 16.56 | 0 | 212.72 / 213.08 (205.94–236.99) | 0.14x / 0.15x | 4.00x smaller |
| 5 | int8 | 66.25 | 0 | 30.06 / 31.00 (28.88–43.07) | 1.00x | 1.00x |
| 5 | f32 | 265.00 | 0 | 1.18 / 1.16 (1.14–2.38) | 25.5x / 26.7x | 4.00x larger |
| 5 | blk16 | 16.56 | 0.53 | 30.70 / 30.53 (29.11–33.77) | 0.98x / 1.02x | 3.88x smaller |
| 5 | **blk32** | 16.56 | 1.05 | **28.10 / 27.67** (27.12–28.22) | **1.07x / 1.12x** | **3.76x smaller** |
| 5 | blk64 | 16.56 | 2.11 | 30.06 / 30.24 (29.68–31.50) | 1.00x / 1.03x | 3.55x smaller |
| 5 | blk128 | 16.56 | 4.22 | 39.90 / 38.19 | 0.75x / 0.81x | 3.19x smaller |
| 5 | blk256 | 16.56 | 8.44 | 51.17 / 47.31 | 0.59x / 0.66x | 2.65x smaller |
| 5 | blk1024 | 16.56 | 33.75 | 51.37 / 48.68 | 0.59x / 0.64x | 1.31x smaller |
| 5 | blkf32 (best f32-LUT) | 16.56 | 0.84 | 35.98 / 36.23 | 0.84x / 0.86x | 3.66x smaller |

"Resident bytes vs int8" counts packed weights plus the one scratch buffer. blk32 is 7.5–7.8x faster than full_expand with the same packed bytes. Speed falls as blocks grow past 64 rows because the int8 and f32 scratch (N×6912×5 bytes) leaves L2 cache; the f32-LUT variant is slower at every block because it gathers 4x more bytes per code. A first exploratory pass (blocks 64–1024 only, noisier box) put blk64 at 1.32x / 1.38x of int8; the two final runs above both put the optimum at 32 rows and are the numbers to cite.

## Verdict

**Viable: the packed W1.58 layer now matches int8-resident speed at 3.8x less memory.** Row-blocked expand into one reused ~1 MiB scratch removes the whole per-step expand penalty found at 02:55: the seven-matrix layer goes from 0.14x of int8-resident to 1.06–1.12x (32-row blocks, both KV-head settings, both runs), at identical error, while resident weights plus scratch stay 3.76–3.79x smaller than int8 (4.00x for the weights alone). The speed side is parity, not a 1.5x win, so the efficiency gain is the memory one, now with no speed cost against int8. f32-resident weights are still about 23x faster in numpy, but at 16x the packed bytes they do not fit an 8 GiB device for 30 layers. Use 32-row (16–64 is within noise) blocks with an int8 LUT gather and a reused f32 scratch for the numpy bitnetd path; do not use blocks of 128 rows or more.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-11T12:08:00Z (~05:08 MT) wrote pair_A/pair_B **no gap** ("05:05 11/11: roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `de17f18` (08:17Z) re-advances landed ROADMAP #5 UE5, left to LLC. JuniorLLM tip at clone time: [2a2149e](https://github.com/cloudcover95/JuniorLLM/commit/2a2149e) (the prior findings note).

## Next question

With the weight stage now about 30 ms per layer, does fusing the A8 g=64 KV attention step with the blk32 weight path give a full B=1 decode step (weights + attention at T=1024–4096) where packed W1.58 stays within 1.1x of int8-resident end to end, and does running the seven blocked GEMVs on a 2–4 thread pool (numpy releases the GIL in matmul and take) give a real speedup over the single-threaded block loop?

close: 3bf71ad1c1b50dbf (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-11_0255)
