---
local: true
date: 2026-11-16
wave: "2026-11-16 05:55 (ran ~05:45–06:05 MT; Chief last_run 2026-11-16T12:07:00Z ~05:07 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (batched decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, batched, b8, b16, tiled-expand, u32-lut, threads, site-fusion, int8-expand, int8-control, numpy, viable]
---

# Running the 02:55 tiled u32-LUT expand on worker threads (each owning a row range and a private tile, BLAS at 1 thread) makes the packed batched path **2.0–3.3x faster** at exact W1.58: B=8 drops from 35.4 ms to **14.3 ms** (T=128, 8 workers) or **17.7 ms** (T=64, 4 workers), interleaved. T=64 × 4 workers uses **26.3 MB** total and roughly matches int8-expand + 8-thread BLAS (B=8 medians 18.6–23.9 ms at 140.25 MB) at **5.3x less memory**. Site fusion alone (q|k|v, gate|up as one pack) gains nothing. The honest control, int8 rows tiled the same way, is still **1.5–1.9x faster** (7.7 ms at B=8 interleaved), so ternary's win here is memory, not speed against equally-parallel int8.

## Question

The prior note [[2026-11-16_0255_tiled-u32-lut-expand-batched-packed-gemm-b1-b16-crossover-vs-int8-expand-and-popcount-full-bitnetd-layer]] found that the tiled u32-LUT expand runs B=8 in 37.4–39.4 ms at 18.43 MB but is only 0.73x of int8-expand's speed. Its next question: can the tiled expand get cheaper than int8-expand at B=8–16 (int8 tiles, fusing q/k/v and gate/up, or a hot-tile cache)? This run measures fusion and thread parallelism, and adds an int8 control that uses the same tiling so the ternary effect is separated from the parallelism effect.

Quant Chief `last_tips.json` `last_run` 2026-11-16T12:07:00Z (~05:07 MT) set pair_A and pair_B to **no gap** ("05:05 11/16: blobs unchanged; both no gap"), so this note uses the lane's own next question.

## Method

- **Unchanged:** `base.py` and `tiled.py` are copied byte for byte from 02:55 (same layer: hidden 2560, GQA 20 q / 5 KV heads, FFN 6912, absmean ternary weights, seed 202611141055).
- **New (`par.py`):**
  - *Site fusion:* the q, k, v row packs are concatenated into one (3840 × 640 B) pack, and gate/up into one (13824 × 640 B), with a per-row gamma vector. Resident = 17.46 MB (2-bit pack + 22,784 f32 gammas).
  - *Threads:* W worker threads each take a contiguous block of tiles. Per tile: `np.take` on the u32 LUT, int8 → f32 cast into a private tile, `x @ tileᵀ` into the output slice. numpy releases the GIL in all three. Scratch is one u32 + one f32 buffer per worker sized for in = 6912 and shared across sites: W·T·6912·5 bytes.
  - *int8 control:* the same tiling and threading, but int8 rows (absmax int8, 69.56 MB resident) cast straight to the f32 tile, with no LUT take.
- **BLAS threads:** with OpenBLAS at its default 8 threads, the worker threads oversubscribe the cores (fused T=128 × 8 workers: 40–79 ms). All thread-parallel results are therefore taken with `OPENBLAS_NUM_THREADS=1`. int8-expand and `f32_resident` are reported at their best setting, which is default (8) BLAS threads; at 1 BLAS thread int8-expand is 34–40 ms at B=4–16.
- **Error:** layer output vs the weight-only W1.58 reference (`wo_mm`) on 32 Gaussian tokens. The int8 control is checked against int8-expand (same int8 model).
- **Timing:** 3 runs with `OPENBLAS_NUM_THREADS=1` at B = 1/4/8/16 (9 repeats × 5 passes, 3 warm; run 1 used a slightly different config list), 2 runs at default BLAS threads, and a 7-round interleaved check at B=8 and B=16. 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench0555_1116`):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py results_b1_run1.json   # run 1 (configs: T16W1, T64W1, T64W4, T64W8, T128W8, T256W8)
OPENBLAS_NUM_THREADS=1 python3 bench.py results_b1_run2.json   # runs 2-3: current bench.py below (adds int8 control)
OPENBLAS_NUM_THREADS=1 python3 bench.py results_b1_run3.json
python3 bench.py results_run1.json                             # default BLAS threads (first config list) 
python3 bench.py results_bdef_run2.json                        # default BLAS threads
OPENBLAS_NUM_THREADS=1 python3 tinter.py                       # interleaved B=8 / B=16
```

<details><summary>par.py</summary>

```python
"""Thread-parallel + site-fused tiled u32-LUT expand (05:55 11/16). Resident bytes = the same 2-bit pack (PR rows), just
concatenated per fused site (q|k|v, gate|up) with a per-row gamma vector. Worker threads (numpy take/copyto/matmul release
the GIL) each own a contiguous row range and a private (T, in) scratch, so nothing larger than W*T*in*8 bytes is ever live."""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import base, tiled
from base import PW, rms, silu, D, KVH, QH, HD
LUT_U32 = tiled.LUT_U32
FUSE = {"qkv": ("q", "k", "v"), "o": ("o",), "gu": ("gate", "up"), "down": ("down",)}
FP = {}
def prep():
    tiled.prep()
    for k, names in FUSE.items():
        PR = np.ascontiguousarray(np.concatenate([PW[n]["PR"] for n in names]))
        g = np.concatenate([np.full(PW[n]["PR"].shape[0], PW[n]["gw"], np.float32) for n in names])
        FP[k] = dict(PR=PR, g=g, split=np.cumsum([PW[n]["PR"].shape[0] for n in names])[:-1])
    return sum(P["PR"].nbytes + P["g"].nbytes for P in FP.values())
_pool = {}; _scr = {}
def pool(W):
    if W not in _pool: _pool[W] = ThreadPoolExecutor(W)
    return _pool[W]
def scratch(w, T, q):
    """one u32 + one f32 buffer per worker, sized for the widest input (F) and viewed down for narrower sites"""
    k = (w, T)
    if k not in _scr: _scr[k] = (np.empty(T * base.F // 4, np.uint32), np.empty(T * base.F, np.float32))
    a, b = _scr[k]; return a[:T * q].reshape(T, q), b[:T * q * 4].reshape(T, q * 4)
def live_scratch_bytes(): return sum(a.nbytes + b.nbytes for a, b in _scr.values())
def scratch_bytes(W, T): return W * T * (base.F // 4) * 4 + W * T * base.F * 4   # = live, shared across sites
def run_rows(PR, x, y, r0, r1, T, w):
    q = PR.shape[1]; s8, s = scratch(w, T, q)
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        np.take(LUT_U32, PR[o0:o0+n], out=s8[:n])
        np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe")
        np.matmul(x, s[:n].T, out=y[:, o0:o0+n])
def fmm(key, x, T, W):
    P = FP[key]; PR = P["PR"]; out = PR.shape[0]
    y = np.empty((x.shape[0], out), np.float32)
    if W == 1: run_rows(PR, x, y, 0, out, T, 0)
    else:
        nt = -(-out // T); per = -(-nt // W)
        futs = [pool(W).submit(run_rows, PR, x, y, w * per * T, min(out, (w + 1) * per * T), T, w) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
    y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
def layer(x, T, W):
    h1 = rms(x); q, k, v = fmm("qkv", h1, T, W)
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    o = fmm("o", att, T, W)[0]; x2 = x + o; h2 = rms(x2)
    g, u = fmm("gu", h2, T, W); z = (silu(g) * u).astype(np.float32)
    return x2 + fmm("down", z, T, W)[0]

# int8 control: identical threading/tiling, but int8 resident rows (4x the bytes) cast straight to the f32 tile (no LUT take)
F8 = {}
def prep8():
    for k, names in FUSE.items():
        F8[k] = dict(Q=np.ascontiguousarray(np.concatenate([PW[n]["q8"] for n in names])),
                     g=np.concatenate([np.full(PW[n]["q8"].shape[0], PW[n]["m8"], np.float32) for n in names]), split=FP[k]["split"])
    return sum(P["Q"].nbytes + P["g"].nbytes for P in F8.values())
def run_rows8(Q, x, y, r0, r1, T, w):
    i = Q.shape[1]; _, s = scratch(w, T, i // 4)
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        np.copyto(s[:n], Q[o0:o0+n], casting="unsafe")
        np.matmul(x, s[:n].T, out=y[:, o0:o0+n])
def fmm8(key, x, T, W):
    P = F8[key]; Q = P["Q"]; out = Q.shape[0]
    y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
    futs = [pool(W).submit(run_rows8, Q, x, y, w * per * T, min(out, (w + 1) * per * T), T, w) for w in range(W) if w * per * T < out]
    for f in futs: f.result()
    y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
def layer8(x, T, W):
    h1 = rms(x); q, k, v = fmm8("qkv", h1, T, W)
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    o = fmm8("o", att, T, W)[0]; x2 = x + o; h2 = rms(x2)
    g, u = fmm8("gu", h2, T, W); z = (silu(g) * u).astype(np.float32)
    return x2 + fmm8("down", z, T, W)[0]
```

</details>

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Can the tiled packed expand beat int8-expand at B=8/16 within <=20 MB? (05:55 11/16). base.py/tiled.py verbatim from 02:55.
New (par.py): site fusion (q|k|v and gate|up as one concatenated 2-bit pack, one pass) and W worker threads over row ranges.
Baselines: 02:55 u32_T16 (unfused, serial), int8_expand (int8 resident + 70.8 MB f32 scratch + BLAS), f32_resident.
Exactness vs weight-only W1.58 layer (wo_mm). numpy-only, offline. Usage: python3 bench.py results_runN.json"""
import json, os, sys
import numpy as np
import base, tiled, par
from base import layer, timed, cos, wo_mm, i8scr, f32res, SEED
BS = (1, 4, 8, 16)
CFG = [("fused_T16_W1", 16, 1), ("fused_T64_W4", 64, 4), ("fused_T64_W8", 64, 8), ("fused_T128_W8", 128, 8), ("fused_T256_W8", 256, 8)]
CFG8 = [("int8_par_T64_W8", 64, 8), ("int8_par_T128_W8", 128, 8)]
def main():
    _, by = base.build(); fused_bytes = par.prep(); i8_bytes = par.prep8()
    res = {"seed": SEED, "numpy": np.__version__, "blas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS", "default"),
           "bytes": by, "fused_resident_bytes": fused_bytes, "int8_par_resident_bytes": i8_bytes, "err": {}, "time": {}, "ratios": {},
           "scratch_bytes": {"int8_expand": base.SCR.nbytes, "u32_T16": 16 * base.F * 8, **{n: par.scratch_bytes(W, T) for n, T, W in CFG}}}
    xe = np.random.default_rng(SEED + 2).standard_normal((32, base.D)).astype(np.float32)
    yw = layer(xe, wo_mm)[0]; d0 = yw - xe
    for n, T, W in CFG + [("u32_T16", 16, 0)]:
        y = par.layer(xe, T, W) if W else layer(xe, tiled.u32_tiled(16))[0]; d1 = y - xe
        res["err"][n] = {"max_abs": float(np.max(np.abs(y - yw))), "delta_cos": cos(d1, d0),
                         "rel_l2": float(np.linalg.norm((d1 - d0).astype(np.float64)) / np.linalg.norm(d0.astype(np.float64)))}
        print("err", n, res["err"][n], flush=True)
    for n, T, W in CFG8:
        y = par.layer8(xe, T, W); y8 = layer(xe, i8scr)[0]
        res["err"][n] = {"max_abs_vs_int8_expand": float(np.max(np.abs(y - y8)))}; print("err", n, res["err"][n], flush=True)
    for B in BS:
        xb = np.random.default_rng(SEED + 9 + B).standard_normal((B, base.D)).astype(np.float32)
        fns = {"int8_expand": lambda: layer(xb, i8scr), "f32_resident": lambda: layer(xb, f32res), "u32_T16": lambda: layer(xb, tiled.u32_tiled(16))}
        for n, T, W in CFG: fns[n] = (lambda T=T, W=W: par.layer(xb, T, W))
        for n, T, W in CFG8: fns[n] = (lambda T=T, W=W: par.layer8(xb, T, W))
        Tm = {}
        for n, fn in fns.items(): Tm[n] = timed(fn); print(B, n, Tm[n], flush=True)
        i8 = Tm["int8_expand"]["median_ms"]; u = Tm["u32_T16"]["median_ms"]
        res["time"][f"B{B}"] = Tm
        res["ratios"][f"B{B}"] = {n: {"vs_int8_expand": i8 / Tm[n]["median_ms"], "vs_u32_T16": u / Tm[n]["median_ms"]} for n in Tm}
    res["live_scratch_bytes_all_configs"] = par.live_scratch_bytes()
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
if __name__ == "__main__": main()
```

</details>

<details><summary>tinter.py</summary>

```python
"""Interleaved timing at B=8 and B=16 (OPENBLAS_NUM_THREADS=1): 7 rounds, each times every config (REPEATS=5 x PASSES=5) in rotated order."""
import json, statistics
import numpy as np
import base, tiled, par
base.build(); par.prep(); par.prep8(); base.REPEATS = 5; out = {}
for B in (8, 16):
    xb = np.random.default_rng(base.SEED + 9 + B).standard_normal((B, base.D)).astype(np.float32)
    fns = {"u32_T16": lambda: base.layer(xb, tiled.u32_tiled(16)), "fused_T64_W4": lambda: par.layer(xb, 64, 4),
           "fused_T128_W8": lambda: par.layer(xb, 128, 8), "int8_par_T128_W8": lambda: par.layer8(xb, 128, 8), "int8_expand_1thr": lambda: base.layer(xb, base.i8scr)}
    names = list(fns); rounds = {n: [] for n in names}
    for r in range(7):
        for n in names[r % len(names):] + names[:r % len(names)]: rounds[n].append(base.timed(fns[n])["median_ms"])
    o = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
    for n in names: o[n]["vs_u32_T16"] = o["u32_T16"]["median_ms"] / o[n]["median_ms"]
    out[f"B{B}"] = o
print(json.dumps(out, indent=1)); json.dump(out, open("tinter.json", "w"), indent=1)
```

</details>

## Results

### Error (run 2)

| kernel | max abs vs W1.58 layer | delta cos | rel-L2 |
|---|---|---|---|
| fused_T16_W1 | 1.91e-06 | 1.000000000000 | 7.1e-07 |
| fused_T64_W4 | 1.91e-06 | 1.000000000000 | 7.1e-07 |
| fused_T64_W8 | 1.91e-06 | 1.000000000000 | 7.1e-07 |
| fused_T128_W8 | 1.91e-06 | 1.000000000000 | 7.1e-07 |
| fused_T256_W8 | 1.91e-06 | 1.000000000000 | 7.1e-07 |
| u32_T16 | 1.91e-06 | 1.000000000000 | 7.1e-07 |
| int8_par_T64_W8 | 0.0e+00 vs int8_expand (same int8 model) | — | — |
| int8_par_T128_W8 | 0.0e+00 vs int8_expand (same int8 model) | — | — |

Every packed path computes the same W1.58 layer (max abs 1.9e-06, float summation order only). The int8 control is bit-identical to int8-expand.

### Bytes

| path | resident | scratch | total |
|---|---|---|---|
| 02:55 u32_T16 | 17.37 MB | 0.88 MB | 18.25 MB |
| fused T=64, 4 workers | 17.46 MB | 8.85 MB | 26.31 MB |
| fused T=64, 8 workers | 17.46 MB | 17.69 MB | 35.15 MB |
| fused T=128, 8 workers | 17.46 MB | 35.39 MB | 52.85 MB |
| int8 control T=128, 8 workers | 69.56 MB | 35.39 MB | 104.95 MB |
| int8 expand + BLAS | 69.47 MB | 70.78 MB | 140.25 MB |
| `f32_resident` | 277.87 MB | 0 | 277.87 MB |

### Timing, `OPENBLAS_NUM_THREADS=1`

B=1

| config | total MB | ms median (min–max), runs 1 / 2 / 3 |
|---|---|---|
| 02:55 tiled u32-LUT T=16 (unfused, serial) | 18.25 | 32.54 (32.00–33.72) / 28.67 (28.50–29.51) / 28.67 (28.60–35.25) |
| fused T=16, 1 worker | 18.01 | 30.30 (29.22–32.60) / 28.48 (28.30–30.86) / 28.52 (28.41–33.75) |
| fused T=64, 1 worker | 19.67 | 28.69 (28.21–29.98) / — / — |
| **fused T=64, 4 workers** | 26.31 | 17.50 (16.62–21.15) / 15.88 (15.75–16.35) / 15.95 (15.58–15.99) |
| fused T=64, 8 workers | 35.15 | 23.48 (21.55–26.54) / 16.86 (15.99–19.25) / 16.17 (15.87–16.49) |
| **fused T=128, 8 workers** | 52.85 | 21.64 (20.95–25.41) / 14.88 (14.62–17.25) / 14.80 (14.56–15.62) |
| fused T=256, 8 workers | 88.24 | 20.28 (17.91–23.27) / 17.59 (17.54–18.05) / 17.78 (17.73–17.98) |
| int8 control: same tiling, int8 resident, T=64, 8 workers | 87.25 | — / 8.75 (8.31–9.25) / 10.15 (9.68–11.29) |
| int8 control: same tiling, int8 resident, T=128, 8 workers | 104.95 | — / 9.84 (9.41–10.08) / 11.02 (10.46–12.33) |
| int8 expand + BLAS (full f32 scratch) | 140.25 | 20.86 (20.47–25.45) / 20.72 (19.57–22.80) / 20.39 (20.23–20.78) |
| `f32_resident` | 277.87 | 11.18 (10.95–12.14) / 10.93 (10.19–12.53) / 11.02 (10.39–12.75) |

B=4

| config | total MB | ms median (min–max), runs 1 / 2 / 3 |
|---|---|---|
| 02:55 tiled u32-LUT T=16 (unfused, serial) | 18.25 | 31.31 (31.10–32.14) / 31.59 (31.41–32.09) / 31.65 (31.45–33.22) |
| fused T=16, 1 worker | 18.01 | 31.40 (31.24–31.68) / 31.72 (31.44–31.94) / 31.50 (31.35–33.34) |
| fused T=64, 1 worker | 19.67 | 35.29 (35.14–35.46) / — / — |
| **fused T=64, 4 workers** | 26.31 | 24.40 (23.53–26.71) / 22.77 (22.59–23.86) / 24.74 (22.85–27.05) |
| fused T=64, 8 workers | 35.15 | 27.37 (26.90–29.49) / 23.70 (23.00–26.13) / 27.54 (27.20–28.26) |
| **fused T=128, 8 workers** | 52.85 | 14.37 (13.08–15.21) / 11.64 (11.13–12.86) / 13.55 (13.11–16.94) |
| fused T=256, 8 workers | 88.24 | 14.65 (13.21–15.74) / 12.02 (11.39–12.59) / 12.79 (12.52–14.37) |
| int8 control: same tiling, int8 resident, T=64, 8 workers | 87.25 | — / 16.34 (16.05–18.39) / 16.13 (15.83–16.88) |
| int8 control: same tiling, int8 resident, T=128, 8 workers | 104.95 | — / 6.31 (6.09–7.85) / 7.57 (7.33–8.47) |
| int8 expand + BLAS (full f32 scratch) | 140.25 | 35.46 (34.36–36.39) / 34.23 (34.08–37.30) / 34.44 (34.31–34.76) |
| `f32_resident` | 277.87 | 25.04 (24.55–26.48) / 24.96 (24.35–25.70) / 24.85 (24.60–25.46) |

B=8

| config | total MB | ms median (min–max), runs 1 / 2 / 3 |
|---|---|---|
| 02:55 tiled u32-LUT T=16 (unfused, serial) | 18.25 | 36.43 (36.14–38.12) / 36.83 (36.65–39.00) / 37.32 (37.06–38.26) |
| fused T=16, 1 worker | 18.01 | 37.31 (36.85–38.21) / 37.53 (37.31–37.72) / 37.50 (37.46–40.05) |
| fused T=64, 1 worker | 19.67 | 52.26 (52.23–53.66) / — / — |
| **fused T=64, 4 workers** | 26.31 | 19.52 (17.89–21.88) / 18.00 (17.37–21.48) / 17.53 (17.16–18.01) |
| fused T=64, 8 workers | 35.15 | 18.84 (17.98–20.95) / 15.05 (14.75–16.45) / 19.45 (18.55–20.94) |
| **fused T=128, 8 workers** | 52.85 | 14.48 (13.42–15.48) / 12.16 (11.92–12.96) / 16.51 (15.57–17.16) |
| fused T=256, 8 workers | 88.24 | 15.35 (14.67–16.52) / 13.22 (12.33–17.12) / 14.95 (14.28–16.69) |
| int8 control: same tiling, int8 resident, T=64, 8 workers | 87.25 | — / 7.53 (6.95–8.44) / 10.23 (9.05–14.57) |
| int8 control: same tiling, int8 resident, T=128, 8 workers | 104.95 | — / 7.15 (6.75–7.93) / 9.69 (8.75–10.58) |
| int8 expand + BLAS (full f32 scratch) | 140.25 | 35.87 (35.72–36.90) / 35.72 (35.53–37.73) / 35.48 (35.41–36.90) |
| `f32_resident` | 277.87 | 26.37 (26.11–26.91) / 25.91 (25.56–27.02) / 25.90 (25.62–28.14) |

B=16

| config | total MB | ms median (min–max), runs 1 / 2 / 3 |
|---|---|---|
| 02:55 tiled u32-LUT T=16 (unfused, serial) | 18.25 | 53.10 (52.79–55.31) / 53.16 (52.95–53.99) / 53.47 (53.14–53.75) |
| fused T=16, 1 worker | 18.01 | 53.34 (53.06–54.97) / 53.37 (53.14–53.55) / 53.92 (53.55–54.57) |
| fused T=64, 1 worker | 19.67 | 56.76 (56.40–58.02) / — / — |
| **fused T=64, 4 workers** | 26.31 | 18.68 (18.39–20.05) / 19.11 (18.82–19.77) / 19.06 (18.55–19.25) |
| fused T=64, 8 workers | 35.15 | 15.50 (15.14–16.29) / 15.70 (15.61–16.65) / 16.02 (15.76–16.55) |
| **fused T=128, 8 workers** | 52.85 | 12.72 (12.37–13.46) / 12.57 (12.30–13.24) / 13.00 (12.49–13.48) |
| fused T=256, 8 workers | 88.24 | 13.21 (12.97–13.87) / 13.15 (12.72–13.99) / 13.45 (13.09–13.83) |
| int8 control: same tiling, int8 resident, T=64, 8 workers | 87.25 | — / 8.11 (7.39–8.78) / 8.41 (7.88–9.43) |
| int8 control: same tiling, int8 resident, T=128, 8 workers | 104.95 | — / 8.24 (7.80–8.73) / 8.27 (7.92–8.83) |
| int8 expand + BLAS (full f32 scratch) | 140.25 | 39.08 (38.77–39.60) / 39.33 (39.20–43.10) / 40.46 (39.86–41.47) |
| `f32_resident` | 277.87 | 29.28 (28.85–32.44) / 29.88 (29.59–32.68) / 30.47 (30.11–31.38) |

### Timing, default (8) BLAS threads

| B | int8_expand ms, runs 1 / 2 | f32_resident ms, runs 1 / 2 | fused T=64 W=8 ms, runs 1 / 2 | fused T=128 W=8 ms, run 2 |
|---|---|---|---|---|
| 1 | 25.98 (25.02–28.90) / 17.79 (16.19–18.42) | 1.29 (1.27–2.11) / 1.40 (1.35–1.47) | 19.65 (18.59–20.64) / 16.63 (16.31–17.06) | 40.27 (35.34–44.21) |
| 4 | 22.24 (21.45–24.97) / 17.53 (17.04–18.38) | 4.12 (3.90–5.46) / 3.79 (3.73–4.48) | 28.98 (24.06–31.69) / 24.37 (24.08–24.73) | 53.16 (39.58–67.47) |
| 8 | 23.86 (22.79–25.12) / 18.60 (17.04–19.33) | 4.71 (4.32–46.70) / 4.67 (4.06–5.59) | 27.17 (25.87–28.82) / 26.40 (25.90–26.60) | 70.45 (51.24–155.72) |
| 16 | 19.99 (18.09–22.59) / 23.03 (22.13–24.84) | 5.35 (5.28–8.08) / 5.76 (5.65–6.76) | 34.38 (33.92–36.89) / 32.07 (30.58–36.54) | 79.15 (62.32–118.84) |

### Interleaved, `OPENBLAS_NUM_THREADS=1`

| B | config | ms median of 7 round medians (min–max) | vs 02:55 u32_T16 |
|---|---|---|---|
| 8 | u32_T16 | 35.43 (35.28–35.55) | 1.00x |
| 8 | fused_T64_W4 | 17.69 (17.56–18.35) | 2.00x |
| 8 | fused_T128_W8 | 14.34 (12.14–14.57) | 2.47x |
| 8 | int8_par_T128_W8 | 7.70 (7.38–8.68) | 4.60x |
| 8 | int8_expand_1thr | 36.88 (35.65–37.99) | 0.96x |
| 16 | u32_T16 | 47.73 (47.53–47.86) | 1.00x |
| 16 | fused_T64_W4 | 19.18 (18.42–19.42) | 2.49x |
| 16 | fused_T128_W8 | 14.58 (12.80–15.71) | 3.27x |
| 16 | int8_par_T128_W8 | 9.32 (8.43–12.15) | 5.12x |
| 16 | int8_expand_1thr | 39.83 (39.55–40.58) | 1.20x |

## Verdict

**Viable** for the batched packed path, on both speed (vs the prior packed path) and memory (vs int8-expand). **Not** a speed win over int8 tiled and threaded the same way.

1. **Threads give the gain, fusion does not.** Fused T=16 with 1 worker ties the unfused 02:55 kernel (B=8 37.3–37.5 vs 36.4–37.3 ms). With BLAS at 1 thread and 4–8 workers, B=8 is 12.2–19.5 ms and B=16 is 12.6–19.1 ms across runs. Interleaved: **2.00x** (T=64 W=4) and **2.47x** (T=128 W=8) at B=8, **2.49x** and **3.27x** at B=16, all vs u32_T16.
2. **Larger tiles pay off once threaded.** Serially T=16 was best (02:55), but with 8 workers T=128 beats T=64 at B ≥ 4 (B=8 12.2–16.5 vs 15.1–19.5 ms). T=256 adds nothing.
3. **vs int8-expand (its best, 8-thread BLAS, 140.25 MB):** int8-expand medians are 18.6–23.9 ms at B=8 and 20.0–23.0 at B=16. Fused T=64 W=4 (26.3 MB) runs 17.5–19.5 / 18.7–19.1 ms: 0.95–1.36x at B=8 and 1.05–1.23x at B=16, at **5.3x less memory**. Fused T=128 W=8 (52.9 MB) runs 12.2–16.5 / 12.6–13.0 ms: 1.13–1.96x and 1.54–1.83x, at 2.7x less memory. The two int8-expand runs are taken at different BLAS settings from the threaded runs, so these ratios cross processes.
4. **The int8 control is the honest ceiling.** Same tiling/threads with int8 rows: B=8 7.2–10.2 ms, B=16 8.1–9.3 ms (incl. interleaved), so ternary at the same T/W is 0.54–0.66x of its speed. The LUT take costs about as much as the matmul at these shapes. Ternary keeps a 3.98x smaller resident footprint (17.46 vs 69.56 MB) and 2.0x smaller total at the same T/W (52.9 vs 105.0 MB).
5. **B=1:** fused T=64–128 with 4–8 workers runs 14.8–16.9 ms in runs 2–3 (17.5–23.5 in run 1), 1.7–1.9x faster than serial u32_T16 in runs 2–3, but still slower than the 18:55 popcount default (12.4–12.6 ms), so B=1 dispatch is unchanged.

Recommended dispatch for i2sd/bitnetd batched decode (B ≥ 3), with `OPENBLAS_NUM_THREADS=1` in the worker:
- RSS-capped (~26 MB/layer): fused tiled u32-LUT expand, T=64, 4 workers.
- ~53 MB/layer: T=128, 8 workers.
- int8 tiled + threaded only where ~4x the resident bytes is acceptable; it is the fastest batched path measured.

**Caveat:** weights and tokens are synthetic, and this is single-process numpy on 8 cores. Threaded timings move with box load (spread up to ±20% between runs). The ≤20 MB total budget from the prior next question was **not** met by any threaded config: the scratch alone is 8.8 MB at T=64 × 4 workers.

## Pair files this wave

At Quant Chief 05:05 MT (`last_run` 12:07Z), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs are unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was 7ca8fa0 (the 02:55 note). Since then nothing new; the only other commit in the last 6 hours is 3eaf332 (`rails/scratchdraft/`, unattributed cloudcover95).
- No Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours. Chief's LLC skip path (JuniorHome 0226f9b) is in another repo.
- This note adds one file under `junior_osai/vault/findings/`.

## Next question

Can the LUT take be cut so ternary closes on the threaded int8 control? Options:
- Cache each worker's expanded int8 (not f32) tiles for the hottest sites across decode steps, within a fixed RSS budget.
- Expand to int8 once per tile and use a bitplane-split matmul (x @ M − 2·x @ (M∧S)) on 0/1 f32 tiles built with `np.unpackbits` per worker.
- Use a 16-bit LUT (2 bytes → 8 trits, 64K × 8 B = 512 KB) to halve the take count.

Measure at B=8/16 against int8 tiled with the same T/W, at ≤ 30 MB total.

close: sha256(prev || this)[:16] = f0371baa7b814e5f
