---
local: true
date: 2026-11-16
wave: "2026-11-16 18:55 (ran ~18:45–19:00 MT; Chief last_run 2026-11-17T01:05:00Z ~18:05 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (batched decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, batched, b1, b8, b16, a8, int8-activations, int32-matmul, w158a8, threads, numpy, not-viable]
---

# A8 activations into an int8 × int8 → int32 tile matmul make the threaded ternary layer **much slower** in numpy, not faster: **0.11–0.26x** of the f32 tile matmul on resident int8 trit tiles and **0.17–0.42x** on the 2-bit u32-LUT path, worst at B=16 (numpy's integer matmul has no BLAS kernel). Holding the same A8 integers in f32 (`a8f`) is bit-identical to int32 (max abs diff 0.0; |acc| ≤ 127·6912 < 2²⁴) and runs at the f32 speed (0.95–1.02x / 0.95–1.01x), so W1.58A8 costs nothing here but buys nothing either. A8 error vs weight-only W1.58: rel L2 0.0085, cos 0.999964, max abs 0.0460. Memory is unchanged (resident 69.6 MB int8 / 17.5 MB 2-bit; int32 scratch = f32 scratch).

## Question

[[2026-11-16_1455_partial-int8-tile-cache-hot-fused-sites-vs-threaded-u32-expand-and-all-int8-trit-tiles-45mb-budget-full-bitnetd-layer-b1-b8-b16]] showed that the 2-bit ↔ int8 trade on this layer is linear (a partial int8 tile cache only gets 1.16–1.40x over u32). Its next question: does **A8 activation quantization into an int8 × int8 → int32 matmul** speed up the threaded path at B=8/16, or does numpy's lack of an int8 BLAS make it slower than the f32 cast-and-matmul?

Quant Chief `last_tips.json` `last_run` 2026-11-17T01:05:00Z (~18:05 MT): pair_A and pair_B both **no gap**, so this note uses the lane's own next question.

## Method

- **Unchanged:** `base.py`, `tiled.py`, `par.py`, `cache.py` copied byte for byte from 14:55 (`/workspace/bench1455_1116`, sha256 verified): hidden 2560, GQA 20 q / 5 KV heads, FFN 6912, absmean ternary weights, seed 202611141055, fused q|k|v and gate|up, W worker threads, private per-worker scratch, `OPENBLAS_NUM_THREADS=1`.
- **New (`a8.py`):** per-token absmax A8 on every site input (x ≈ sx·xq, xq ∈ [−127, 127]). Three tile kinds on two resident layouts (all-int8 trit tiles from `cache.CT`, or 2-bit pack + u32 LUT take):
  - `f32`: trit tile → f32, x_f32 @ tileᵀ (W1.58, no A8; the 14:55 `int8all` / `u32` paths).
  - `i32`: trit tile → int32, xq_int32 @ tileᵀ in int32, then y = sx·acc·γ.
  - `a8f`: the same xq held in f32, f32 matmul (exact-integer control).
- **Error:** vs weight-only W1.58 reference (`base.wo_mm`) on 32 Gaussian tokens; i32 vs a8f checked bit-for-bit.
- **Timing:** 5 interleaved rounds per B (each round times all 12 configs in rotated order, 5 repeats × 5 passes, 3 warm), median of round medians with min–max; two full runs. 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench1855_1116`):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py results_run1.json
OPENBLAS_NUM_THREADS=1 python3 bench.py results_run2.json
```

<details><summary>a8.py</summary>

```python
"""A8 activations into int8 x int8 -> int32 matmul vs f32 tile matmul (18:55 11/16). Per-token absmax A8 on each site
input (x ~= sx * xq, xq int8 in [-127,127]). Resident weights: all-int8 trit tiles (cache.CT) or 2-bit pack (par.FP, u32 LUT).
Tile math per kind:  f32  = cast trit tile -> f32, x_f32 @ tile^T (W1.58, no A8; the 14:55 int8all / u32 paths)
                     i32  = cast trit tile -> int32, xq_i32 @ tile^T in int32 (numpy has no int BLAS), y = sx * acc * g
                     a8f  = xq held as f32, f32 tile matmul; |acc| <= 127*6912 < 2^24 so it is bit-identical to i32."""
import numpy as np
import base, par, cache
from base import rms, silu, D, KVH, QH, HD
_s = {}
def scr(w, T, dt):
    k = (w, T, dt)
    if k not in _s: _s[k] = np.empty(T * base.F, dt)
    return _s[k]
def a8(x):
    sx = np.maximum(np.abs(x).max(axis=1, keepdims=True), 1e-8) / 127.0
    return np.clip(np.rint(x / sx), -127, 127), sx.astype(np.float32)
def rows(src, packed, xin, y, r0, r1, T, w, dt):
    i = xin.shape[1]; s = scr(w, T, dt)[:T * i].reshape(T, i)
    s8 = par.scratch(w, T, i // 4)[0] if packed else None
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if packed:
            np.take(par.LUT_U32, src[o0:o0+n], out=s8[:n]); np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe")
        else: np.copyto(s[:n], src[o0:o0+n], casting="unsafe")
        np.matmul(xin, s[:n].T, out=y[:, o0:o0+n])
def mm(key, x, T, W, res, kind):
    packed = res == "u32"
    P = par.FP[key] if packed else cache.CT[key]; src = P["PR"] if packed else P["Q"]; out = src.shape[0]
    if kind == "f32": xin, sx, dt = x, None, np.float32
    else:
        q, sx = a8(x); dt = np.int32 if kind == "i32" else np.float32; xin = q.astype(dt)
    y = np.empty((x.shape[0], out), dt); nt = -(-out // T); per = -(-nt // W)
    futs = [par.pool(W).submit(rows, src, packed, xin, y, w * per * T, min(out, (w + 1) * per * T), T, w, dt) for w in range(W) if w * per * T < out]
    for f in futs: f.result()
    y = y.astype(np.float32, copy=False)
    if sx is not None: y *= sx
    y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
def layer(res, kind, x, T, W):
    f = lambda k, h: mm(k, h, T, W, res, kind)
    h1 = rms(x); q, k, v = f("qkv", h1)
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    o = f("o", att)[0]; x2 = x + o; h2 = rms(x2)
    g, u = f("gu", h2); z = (silu(g) * u).astype(np.float32)
    return x2 + f("down", z)[0]
```

</details>

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""A8 int8 x int8 -> int32 tile matmul vs f32 tile matmul on the threaded full bitnetd layer, resident int8 trit tiles
and 2-bit u32-LUT pack, B=1/8/16. base/tiled/par/cache verbatim from 14:55.
Usage: OPENBLAS_NUM_THREADS=1 python3 bench.py out.json"""
import json, os, sys, statistics
import numpy as np
import base, par, cache, a8
base.build(); par.prep(); cache.prep()
CFG = [(f"{r}_{k}_T{T}_W{W}", r, k, T, W) for T, W in ((64, 4), (128, 8)) for r in ("int8", "u32") for k in ("f32", "i32", "a8f")]
res = {"seed": base.SEED, "numpy": np.__version__, "blas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS", "default"), "cpus": os.cpu_count(),
       "resident_bytes": {"int8": cache.resident_bytes(tuple(par.FP)), "u32": cache.resident_bytes(())}, "err": {}, "time": {}}
xe = np.random.default_rng(base.SEED + 2).standard_normal((32, base.D)).astype(np.float32)
yw = base.layer(xe, base.wo_mm)[0]; outs = {}
for n, r, k, T, W in CFG:
    y = a8.layer(r, k, xe, T, W); outs[n] = y
    res["err"][n] = {"max_abs_vs_w158": float(np.max(np.abs(y - yw))), "rel_l2_vs_w158": float(np.linalg.norm(y - yw) / np.linalg.norm(yw)), "cos_vs_w158": base.cos(y, yw)}
    print("err", n, res["err"][n], flush=True)
for T, W in ((64, 4), (128, 8)):
    res["err"][f"i32_eq_a8f_T{T}_W{W}"] = float(np.max(np.abs(outs[f"int8_i32_T{T}_W{W}"] - outs[f"int8_a8f_T{T}_W{W}"])))
print("i32 vs a8f", {k: v for k, v in res["err"].items() if k.startswith("i32_eq")}, flush=True)
base.REPEATS = 5; ROUNDS = 5
for B in (1, 8, 16):
    xb = np.random.default_rng(base.SEED + 9 + B).standard_normal((B, base.D)).astype(np.float32)
    names = [c[0] for c in CFG]; fns = {n: (lambda r=r, k=k, T=T, W=W: a8.layer(r, k, xb, T, W)) for n, r, k, T, W in CFG}
    rounds = {n: [] for n in names}
    for rr in range(ROUNDS):
        for n in names[rr % len(names):] + names[:rr % len(names)]: rounds[n].append(base.timed(fns[n])["median_ms"])
    o = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
    for n in names:
        r, k, T, W = n.split("_"); o[n]["vs_same_res_f32"] = o[f"{r}_f32_{T}_{W}"]["median_ms"] / o[n]["median_ms"]
    res["time"][f"B{B}"] = o
    for n in names: print(B, n, {kk: round(v, 3) for kk, v in o[n].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
```

</details>

## Results

Error (identical for both resident layouts and both T/W):

| kind | max abs vs W1.58 | rel L2 vs W1.58 | cos vs W1.58 |
|---|---|---|---|
| f32 | 1.9e-06 | 3.3e-07 | 1.000000 |
| i32 (A8) | 0.0460 | 0.0085 | 0.999964 |
| a8f (A8) | 0.0460 | 0.0085 | 0.999964 |

i32 vs a8f max abs diff: 0.0 (T64 W4), 0.0 (T128 W8).

Layer ms, median (min–max of 5 rounds); last column is speed vs the f32 kind on the same resident layout and T/W.

| B | config | resident MB | run 1 ms | run 2 ms | vs f32 (r1 / r2) |
|---|---|---|---|---|---|
| 1 | `int8_f32_T64_W4` | 69.6 | 8.13 (8.08–10.14) | 8.04 (7.73–8.16) | 1.00 / 1.00 |
| 1 | `int8_i32_T64_W4` | 69.6 | 34.43 (33.98–35.43) | 34.46 (33.83–34.76) | 0.24 / 0.23 |
| 1 | `int8_a8f_T64_W4` | 69.6 | 8.21 (8.18–9.98) | 8.31 (8.07–8.71) | 0.99 / 0.97 |
| 1 | `u32_f32_T64_W4` | 17.5 | 17.02 (16.51–17.21) | 16.83 (16.36–17.45) | 1.00 / 1.00 |
| 1 | `u32_i32_T64_W4` | 17.5 | 40.33 (40.01–40.86) | 40.05 (39.49–40.58) | 0.42 / 0.42 |
| 1 | `u32_a8f_T64_W4` | 17.5 | 16.91 (16.63–17.04) | 17.16 (16.59–17.52) | 1.01 / 0.98 |
| 1 | `int8_f32_T128_W8` | 69.6 | 10.23 (9.93–10.45) | 9.91 (9.71–10.21) | 1.00 / 1.00 |
| 1 | `int8_i32_T128_W8` | 69.6 | 39.45 (39.02–39.94) | 39.74 (39.55–40.20) | 0.26 / 0.25 |
| 1 | `int8_a8f_T128_W8` | 69.6 | 10.16 (9.90–10.26) | 10.36 (9.87–10.70) | 1.01 / 0.96 |
| 1 | `u32_f32_T128_W8` | 17.5 | 15.10 (15.02–15.33) | 15.19 (15.06–15.38) | 1.00 / 1.00 |
| 1 | `u32_i32_T128_W8` | 17.5 | 41.42 (40.95–41.82) | 41.65 (41.38–42.81) | 0.36 / 0.36 |
| 1 | `u32_a8f_T128_W8` | 17.5 | 15.25 (15.20–16.07) | 15.16 (15.11–15.43) | 0.99 / 1.00 |
| 8 | `int8_f32_T64_W4` | 69.6 | 11.19 (10.77–11.58) | 10.48 (10.35–13.09) | 1.00 / 1.00 |
| 8 | `int8_i32_T64_W4` | 69.6 | 58.69 (56.61–59.48) | 58.81 (57.38–60.04) | 0.19 / 0.18 |
| 8 | `int8_a8f_T64_W4` | 69.6 | 11.15 (10.81–11.83) | 10.78 (10.71–11.51) | 1.00 / 0.97 |
| 8 | `u32_f32_T64_W4` | 17.5 | 18.75 (17.54–19.01) | 17.71 (17.56–23.79) | 1.00 / 1.00 |
| 8 | `u32_i32_T64_W4` | 17.5 | 62.40 (61.89–64.20) | 61.85 (60.99–64.74) | 0.30 / 0.29 |
| 8 | `u32_a8f_T64_W4` | 17.5 | 19.35 (18.10–20.40) | 18.45 (17.99–23.69) | 0.97 / 0.96 |
| 8 | `int8_f32_T128_W8` | 69.6 | 8.59 (7.55–8.88) | 8.55 (7.70–10.16) | 1.00 / 1.00 |
| 8 | `int8_i32_T128_W8` | 69.6 | 38.41 (37.47–40.67) | 39.86 (38.83–49.31) | 0.22 / 0.21 |
| 8 | `int8_a8f_T128_W8` | 69.6 | 8.84 (8.19–9.05) | 9.04 (8.08–11.97) | 0.97 / 0.95 |
| 8 | `u32_f32_T128_W8` | 17.5 | 15.03 (12.67–15.50) | 14.52 (12.77–16.53) | 1.00 / 1.00 |
| 8 | `u32_i32_T128_W8` | 17.5 | 43.36 (40.19–44.44) | 42.78 (39.56–51.96) | 0.35 / 0.34 |
| 8 | `u32_a8f_T128_W8` | 17.5 | 15.24 (12.51–15.56) | 15.33 (12.64–17.16) | 0.99 / 0.95 |
| 16 | `int8_f32_T64_W4` | 69.6 | 11.96 (11.91–12.04) | 12.41 (12.08–12.58) | 1.00 / 1.00 |
| 16 | `int8_i32_T64_W4` | 69.6 | 110.39 (109.01–110.90) | 114.64 (111.80–117.57) | 0.11 / 0.11 |
| 16 | `int8_a8f_T64_W4` | 69.6 | 12.39 (12.13–12.62) | 12.50 (12.39–12.84) | 0.97 / 0.99 |
| 16 | `u32_f32_T64_W4` | 17.5 | 19.06 (18.59–19.47) | 19.39 (18.98–19.98) | 1.00 / 1.00 |
| 16 | `u32_i32_T64_W4` | 17.5 | 115.26 (112.70–119.87) | 116.80 (114.59–118.44) | 0.17 / 0.17 |
| 16 | `u32_a8f_T64_W4` | 17.5 | 19.61 (19.50–20.22) | 19.76 (19.33–22.62) | 0.97 / 0.98 |
| 16 | `int8_f32_T128_W8` | 69.6 | 8.65 (8.13–8.67) | 8.78 (8.21–9.23) | 1.00 / 1.00 |
| 16 | `int8_i32_T128_W8` | 69.6 | 69.85 (68.67–71.11) | 71.04 (69.67–73.96) | 0.12 / 0.12 |
| 16 | `int8_a8f_T128_W8` | 69.6 | 8.50 (8.33–8.79) | 8.79 (8.66–8.98) | 1.02 / 1.00 |
| 16 | `u32_f32_T128_W8` | 17.5 | 13.06 (12.89–13.54) | 14.01 (13.03–14.48) | 1.00 / 1.00 |
| 16 | `u32_i32_T128_W8` | 17.5 | 70.65 (69.92–72.70) | 72.18 (70.94–72.73) | 0.18 / 0.19 |
| 16 | `u32_a8f_T128_W8` | 17.5 | 13.58 (13.37–13.91) | 13.82 (13.42–15.95) | 0.96 / 1.01 |

## Verdict

**not-viable.** int8 × int8 → int32 in numpy is 0.11–0.26x the speed of the f32 tile matmul on int8 trit tiles and 0.17–0.42x on the 2-bit path. It gets worse as B grows, because numpy's integer matmul is a plain loop while f32 goes through OpenBLAS. A8 saves no resident bytes (weights are already int8 or 2-bit; activations are ≤ 16 × 6912). The exact-integer f32 path (`a8f`) matches int32 bit for bit at f32 speed (0.95–1.02x), so A8 can be used in this stack without a speed penalty, but it does not change the speed side. A real int8 GEMM (VNNI/dotprod kernel in bitnetd/i2sd C) is the only way A8 pays off; numpy can't show it.

**Caveat:** synthetic weights/tokens; single-process numpy on 8 cores; threaded timings shift with box load.

## Pair files this wave

At Quant Chief 18:05 MT (`last_run` 01:05Z Nov 17), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was 7f87243 (the 14:55 note); no other commit in the last 6 hours.
- No Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours; Chief's LLC skip paths (JuniorHome 612f903, 3249cbb) are in another repo.
- This note adds one file under `junior_osai/vault/findings/`.

## Next question

The weight-side GEMM line on this numpy layer is mapped (expand, LUT, cache, A8). Move to the KV side for long-context decode: at context 2048–8192 and B=8, does a **2-bit ternary-sign + per-group-scale K cache** (sign/zero mask planes, absmean per 64) give ≥ 1.5x less KV memory than A8 g64 K at attention-score cos ≥ 0.99, and what does the scoring step cost in numpy?

close: sha256(prev || this)[:16] = f61d25621b455b70
