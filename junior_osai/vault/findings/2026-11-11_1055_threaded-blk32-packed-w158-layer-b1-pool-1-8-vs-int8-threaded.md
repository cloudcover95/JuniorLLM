---
local: true
date: 2026-11-11
wave: "2026-11-11 10:55 (ran ~10:55–11:00 MT; Chief last_run 2026-11-11T17:13:00Z ~10:13 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, w158, i2s, lut-expand, row-block, threads, threadpool, numpy, not-viable]
---

# Running the blk32 packed W1.58 layer on a 2–8 thread pool gives no speedup (0.89–1.09x of single-thread), while a 4-thread int8-resident split reaches up to 1.5x, so threads widen the speed gap against packed

## Question

The prior note [[2026-11-11_0555_row-blocked-reused-scratch-lut-expand-packed-w158-layer-b1-vs-int8-resident]] found 32-row blocked LUT expand puts the seven-matrix packed W1.58 layer at 1.06–1.12x of int8-resident speed at 3.79x fewer bytes. Its next question had two halves; this note runs the second: does splitting each layer's rows across a 2/4/8-worker thread pool (each worker with its own scratch; numpy releases the GIL inside `take`, `copyto` and `matmul`) give a real speedup over the single-threaded block loop, and does the packed path keep parity once int8-resident gets the same thread split? Quant Chief `last_tips.json` `last_run` 2026-11-11T17:13:00Z (~10:13 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same layer as the prior note: B=1, hidden 2560, FFN 6912 (gate, up, down), 20 query heads × 128, KV heads 20 (MHA) and 5 (GQA), assumed shapes. Absmean ternary weights, one f32 scale per matrix; the layer time is the sum of the seven GEMVs.

- `blk32_tN`: rows of each matrix split into N contiguous 32-row-aligned ranges, one per `ThreadPoolExecutor` worker; each worker gathers its 32-row blocks from the 256×4 int8 LUT into its own int8 scratch, casts into its own f32 scratch, and `matmul`s into its output slice. Scratch is N × 32 × 6912 × 5 bytes.
- `int8_tN`: int8-resident trits, same row split, each worker casts its slice to f32 and GEMVs.
- `f32`: pre-scaled f32-resident single `matmul` (memory-heavy ceiling; OpenBLAS threads it internally).

N in {1, 2, 4, 8} on the shared 8-core box, numpy 2.5.4 with scipy-openblas 0.3.34 (OpenBLAS left at its default threads), seed `202611111055`, 2 warm-ups, 7 repeats × 3 passes, median and min–max, run three times back to back. Error is max |y − ref| / max |ref| against an f64 GEMV on the same trits.

Commands, from `/workspace/bench1055_1111`: `python3 bench.py results_run1.json`, `python3 bench.py results_run2.json`, `python3 bench.py results_run3.json`.

```python
# Full bitnetd-shaped layer at B=1 (hidden 2560, FFN 6912, 20 q heads x 128, KV heads 20 and 5):
# blk32 row-blocked LUT expand of W1.58 2-bit packed weights (prior note's winner), run on a thread pool of 1/2/4/8 workers
# (each worker owns its own int8+f32 scratch, row ranges split evenly; numpy releases the GIL in take/copyto/matmul),
# vs int8-resident (cast+GEMV) single and the same thread split, and f32-resident GEMV.
import numpy as np, time, json, sys
from concurrent.futures import ThreadPoolExecutor
SEED = 202611111055
Hd, F, D, BS = 2560, 6912, 128, 32
REP, PASS, WARM = 7, 3, 2
THREADS = [1, 2, 4, 8]
rng = np.random.default_rng(SEED)
LUT = np.array([[((b >> (2 * k)) & 3) for k in range(4)] for b in range(256)], dtype=np.int8) - 1
def tern(o, i):
    W = rng.standard_normal((o, i)).astype(np.float32); s = np.abs(W).mean()
    return np.clip(np.rint(W / s), -1, 1).astype(np.int8), np.float32(s)
def pack(T):
    c = (T + 1).astype(np.uint8).reshape(T.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
def blk_range(P, x, y, r0, r1, s8, sf):
    q = P.shape[1]; i = q * 4
    for r in range(r0, r1, BS):
        n = min(BS, r1 - r)
        b8 = s8[:n, :q]; np.take(LUT, P[r:r + n], axis=0, out=b8)
        bf = sf[:n, :i]; np.copyto(bf, b8.reshape(n, i), casting='unsafe')
        np.matmul(bf, x, out=y[r:r + n])
def i8_range(W8, x, y, r0, r1):
    y[r0:r1] = W8[r0:r1].astype(np.float32) @ x
def splits(o, t):  # BS-aligned even split
    nb = -(-o // BS); per = -(-nb // t)
    return [(a * per * BS, min(o, (a + 1) * per * BS)) for a in range(t) if a * per * BS < o]
def make_pool_blk(t, pool):
    sc = [(np.empty((BS, F // 4, 4), np.int8), np.empty((BS, F), np.float32)) for _ in range(t)]
    def f(P, s, x, y):
        sp = splits(P.shape[0], t)
        if t == 1: blk_range(P, x, y, 0, P.shape[0], *sc[0])
        else: list(pool.map(lambda a: blk_range(P, x, y, sp[a][0], sp[a][1], *sc[a]), range(len(sp))))
        y *= s
    return f
def make_pool_i8(t, pool):
    def f(W8, s, x, y):
        sp = splits(W8.shape[0], t)
        if t == 1: i8_range(W8, x, y, 0, W8.shape[0])
        else: list(pool.map(lambda a: i8_range(W8, x, y, *sp[a]), range(len(sp))))
        y *= s
    return f
def f32(Wf, s, x, y): np.matmul(Wf, x, out=y)
def bench(f, *a):
    for _ in range(WARM): f(*a)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for _ in range(PASS): f(*a)
        ts.append((time.perf_counter() - t0) / PASS * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))
pools = {t: ThreadPoolExecutor(t) for t in THREADS if t > 1}
forms = {"f32": f32}
for t in THREADS:
    forms[f"int8_t{t}"] = make_pool_i8(t, pools.get(t)); forms[f"blk32_t{t}"] = make_pool_blk(t, pools.get(t))
out = {"seed": SEED, "numpy": np.__version__, "layers": {}}
x = rng.standard_normal(Hd).astype(np.float32); xf = rng.standard_normal(F).astype(np.float32)
for kvh in (20, 5):
    shapes = {"q": (Hd, Hd), "k": (kvh * D, Hd), "v": (kvh * D, Hd), "o": (Hd, Hd), "gate": (F, Hd), "up": (F, Hd), "down": (Hd, F)}
    acc = {k: [0.0, 0.0, 0.0] for k in forms}; by = {k: 0 for k in forms}; err = {k: 0.0 for k in forms}
    for nm, (o, i) in shapes.items():
        T8, s = tern(o, i); P = pack(T8); Wf = T8.astype(np.float32) * s; v = xf if i == F else x
        ref = (T8.astype(np.float64) @ v.astype(np.float64)) * float(s); y = np.empty(o, np.float32)
        for k, f in forms.items():
            W = T8 if k.startswith("int8") else (Wf if k == "f32" else P)
            m, lo, hi = bench(f, W, s, v, y)
            acc[k] = [acc[k][0] + m, acc[k][1] + lo, acc[k][2] + hi]; by[k] += W.nbytes
            f(W, s, v, y); err[k] = max(err[k], float(np.max(np.abs(y - ref)) / np.max(np.abs(ref))))
        del Wf, T8, P
    scratch = {k: (int(k.split("_t")[1]) * BS * F * 5 if k.startswith("blk") else 0) for k in forms}
    out["layers"][str(kvh)] = dict(ms=acc, bytes=by, scratch=scratch, max_rel_err=err)
    print("kvh", kvh, flush=True)
    for k in forms:
        print(f"  {k:10s} {acc[k][0]:9.3f} ms [{acc[k][1]:.3f}-{acc[k][2]:.3f}] {by[k]/2**20:7.2f} MiB scratch {scratch[k]/2**20:5.2f} MiB err {err[k]:.1e} vs_int8_t1 {acc['int8_t1'][0]/acc[k][0]:.2f}x vs_blk32_t1 {acc['blk32_t1'][0]/acc[k][0]:.2f}x", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Seven-matrix layer weight stage, ms median for run2 / run3 (run3 min–max). Run1 is left out of the table: its `int8_t1` and `int8_t2` medians were 160–185 ms with 120–269 ms spread (box contention at start, load average 4.6), while its packed rows matched runs 2 and 3 (blk32_t1 32.66 / 29.28 ms). Max relative error is 4.0e-07 (20 KV heads) and 3.3e-07 (5 KV heads) for every packed and int8 form.

| KV heads | form | resident MiB | scratch MiB | ms run2 / run3 (run3 min–max) | vs own t1 (run2 / run3) | vs int8_t1 (run2 / run3) |
|---|---|---|---|---|---|---|
| 20 | f32 | 302.50 | 0 | 3.40 / 1.98 (1.85–2.15) | n/a | 11.6x / 19.1x |
| 20 | int8_t1 | 75.62 | 0 | 39.33 / 37.86 (34.58–51.29) | 1.00x | 1.00x |
| 20 | int8_t2 | 75.62 | 0 | 57.84 / 45.11 (40.06–70.67) | 0.68x / 0.84x | 0.68x / 0.84x |
| 20 | int8_t4 | 75.62 | 0 | 47.51 / 32.26 (24.44–47.60) | 0.83x / 1.17x | 0.83x / 1.17x |
| 20 | int8_t8 | 75.62 | 0 | 45.81 / 36.66 (29.09–59.49) | 0.86x / 1.03x | 0.86x / 1.03x |
| 20 | blk32_t1 | 18.91 | 1.05 | 35.40 / 34.39 (32.86–39.25) | 1.00x | 1.11x / 1.10x |
| 20 | blk32_t2 | 18.91 | 2.11 | 36.21 / 33.21 (29.12–44.71) | 0.98x / 1.04x | 1.09x / 1.14x |
| 20 | blk32_t4 | 18.91 | 4.22 | 32.38 / 35.17 (30.96–42.71) | 1.09x / 0.98x | 1.21x / 1.08x |
| 20 | blk32_t8 | 18.91 | 8.44 | 39.63 / 38.23 (31.05–52.03) | 0.89x / 0.90x | 0.99x / 0.99x |
| 5 | f32 | 265.00 | 0 | 1.41 / 1.19 (1.13–1.36) | n/a | 22.5x / 23.6x |
| 5 | int8_t1 | 66.25 | 0 | 31.66 / 28.01 (25.91–36.58) | 1.00x | 1.00x |
| 5 | int8_t2 | 66.25 | 0 | 37.73 / 34.24 (29.59–43.38) | 0.84x / 0.82x | 0.84x / 0.82x |
| 5 | **int8_t4** | 66.25 | 0 | **22.55 / 18.50** (16.41–30.10) | **1.40x / 1.51x** | **1.40x / 1.51x** |
| 5 | int8_t8 | 66.25 | 0 | 23.07 / 19.91 (17.53–32.07) | 1.37x / 1.41x | 1.37x / 1.41x |
| 5 | blk32_t1 | 16.56 | 1.05 | 30.21 / 28.95 (27.72–37.19) | 1.00x | 1.05x / 0.97x |
| 5 | blk32_t2 | 16.56 | 2.11 | 30.33 / 29.24 (26.08–32.00) | 1.00x / 0.99x | 1.04x / 0.96x |
| 5 | blk32_t4 | 16.56 | 4.22 | 30.35 / 28.83 (24.43–36.42) | 1.00x / 1.00x | 1.04x / 0.97x |
| 5 | blk32_t8 | 16.56 | 8.44 | 33.58 / 30.04 (25.84–38.45) | 0.89x / 0.96x | 0.94x / 0.93x |

Packed bytes plus scratch stay 3.79x (t1) to 2.75x (t8) smaller than int8 at 20 KV heads. The blocked packed path is flat across 1–4 workers and slower at 8, which points to the per-block LUT gather and cast being serialized or memory-bound rather than spread across cores; the extra workers only add scratch and dispatch cost. The int8 split is noisy and mixed at 20 KV heads (0.68–1.17x) but at 5 KV heads 4 workers are 1.40–1.51x faster, so the best threaded int8 beats the best threaded packed form by up to 1.56x (run3, 18.50 vs 28.83 ms).

## Verdict

**Not viable: a thread pool does not speed up the blk32 packed layer.** Across both KV-head settings and two clean runs, 2–8 workers give 0.89–1.09x of the single-threaded blk32 loop, far below the 1.5x bar. The packed path's speed parity with int8 from the prior note holds only against single-threaded int8; once int8-resident rows are split across 4 workers at 5 KV heads, int8 runs 1.40–1.51x faster and packed falls to about 0.64x of it. The memory win (about 3.8x at one worker) is unchanged, so keep the single-threaded blk32 loop for the numpy bitnetd path and do not add a Python thread pool around it; if threads are spent anywhere, spend them on the f32 cast and GEMV, not on the LUT gather.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-11T17:13:00Z (~10:13 MT) wrote pair_A/pair_B **no gap** ("10:05 11/11: roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). `llc_skip_paths`: JuniorHome `25c653b` (15:18Z) re-advances landed ROADMAP #5 UE5, left to LLC. JuniorLLM tip at clone time: [7857193](https://github.com/cloudcover95/JuniorLLM/commit/7857193) (the prior findings note).

## Next question

Does a gather-free expand (decode the 2-bit codes with vectorized shifts and masks on a uint32 view of each 32-row block straight into the f32 scratch, no LUT `take`) cut the blk32 layer below 0.67x of its current ~29–34 ms, so packed W1.58 beats int8-resident by at least 1.5x at B=1 single-threaded while keeping the 3.8x memory win?

close: 9bb9b8cfa45a20f3 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-11_0555)
