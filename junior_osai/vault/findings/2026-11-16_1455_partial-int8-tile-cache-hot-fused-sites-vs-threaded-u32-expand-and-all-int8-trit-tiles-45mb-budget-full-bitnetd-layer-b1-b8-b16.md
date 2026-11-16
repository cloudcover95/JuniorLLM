---
local: true
date: 2026-11-16
wave: "2026-11-16 14:55 (ran ~14:43–15:00 MT; Chief last_run 2026-11-16T21:05:00Z ~14:05 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (batched decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, batched, b1, b8, b16, tiled-expand, u32-lut, int8-cache, partial-cache, threads, numpy, not-viable]
---

# A partial int8 tile cache (hot fused sites kept pre-expanded as exact int8 trits, the rest 2-bit u32-LUT expanded per step) does **not** reach 1.5x over the threaded u32 path inside the 45 MB resident budget. The best split, **gate|up cached** (44.0 MB resident, 1.58x smaller than all-int8 69.6 MB), is only **1.16–1.40x** faster than all-u32 (17.5 MB) across B=1/8/16 and both T/W, with one outlier 1.51x (B=16 T128 W8, run 1; 1.23x in run 2). Caching q|k|v+o (29.7 MB) gives 1.05–1.33x, down+o (35.6 MB) 1.07–1.40x. Speed scales roughly linearly with the fraction of weights cached, so every budget point sits on the line between u32 and all-int8: no split is ≥1.5x faster **and** ≥1.5x smaller at the same time. All variants are exact W1.58 (max abs 1.9e-6 vs weight-only reference).

## Question

[[2026-11-16_1055_16bit-lut-2byte-8trit-take-and-direct-f32-lut-vs-u32-lut-and-threaded-int8-tile-control-full-bitnetd-layer-b1-b8-b16]] closed the LUT-take line (16-bit LUT 0.98–1.08x over u32). Its next question: under a fixed ≤ 45 MB resident budget, does a per-worker/shared **cache of expanded int8 tiles for the hottest fused sites across decode steps** reach ≥ 1.5x over the threaded u32 path at B=8/16 while staying ≥ 1.5x smaller than full int8 tiles?

Quant Chief `last_tips.json` `last_run` 2026-11-16T21:05:00Z (~14:05 MT): pair_A and pair_B both **no gap** ("14:05 11/16: blobs unchanged; both no gap"), so this note uses the lane's own next question.

## Method

- **Unchanged:** `base.py`, `tiled.py`, `par.py` copied byte for byte from 10:55 (`/workspace/bench1055_1116`, sha256 verified): hidden 2560, GQA 20 q / 5 KV heads, FFN 6912, absmean ternary weights, seed 202611141055, site fusion q|k|v and gate|up, W worker threads, private per-worker scratch, `OPENBLAS_NUM_THREADS=1`.
- **New (`cache.py`):** at load, each chosen fused site's 2-bit pack is expanded once to int8 trit rows (`LUT_U32[PR].view(int8)`, 1 B/weight, exact) and kept resident across decode steps; those sites run `par.run_rows8` (cast int8 → f32 tile, matmul, no take). Uncached sites run `par.run_rows` (u32 LUT take). Same per-row gammas.
- **Configs:** `u32` (nothing cached, 17.5 MB), `attn` (q|k|v + o cached, 29.7 MB), `down_o` (down + o, 35.6 MB), `gu` (gate|up, 44.0 MB), `int8all` (every site cached, 69.6 MB = the int8 tile control but with exact trit values). T/W = 64/4 and 128/8.
- **Error:** layer output vs weight-only W1.58 reference (`base.wo_mm`) on 32 Gaussian tokens.
- **Timing:** 7 interleaved rounds per B (each round times all 10 configs in rotated order, 5 repeats × 5 passes, 3 warm), median of round medians with min–max; two full runs. 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench1455_1116`):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py results_run1.json
OPENBLAS_NUM_THREADS=1 python3 bench.py results_run2.json
```

<details><summary>cache.py</summary>

```python
"""Partial int8 tile cache across decode steps (14:55 11/16). Chosen fused sites keep their trits pre-expanded as
int8 rows (exact W1.58, 1 B/weight, built once from the same 2-bit pack); the rest stay 2-bit and are u32-LUT expanded
per step exactly as par.py. Same T/W tiling, threads and per-worker scratch for both kinds."""
import numpy as np
import par, base
from base import rms, silu, D, KVH, QH, HD
CT = {}
def prep():
    for k, P in par.FP.items():
        CT[k] = dict(Q=np.ascontiguousarray(par.LUT_U32[P["PR"]].view(np.int8).reshape(P["PR"].shape[0], -1)), g=P["g"], split=P["split"])
def resident_bytes(hot):
    return sum((CT[k]["Q"].nbytes if k in hot else par.FP[k]["PR"].nbytes) + par.FP[k]["g"].nbytes for k in par.FP)
def fmmc(key, x, T, W):
    P = CT[key]; Q = P["Q"]; out = Q.shape[0]
    y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
    futs = [par.pool(W).submit(par.run_rows8, Q, x, y, w * per * T, min(out, (w + 1) * per * T), T, w) for w in range(W) if w * per * T < out]
    for f in futs: f.result()
    y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
def layer(hot, x, T, W):
    mm = lambda k, h: (fmmc if k in hot else par.fmm)(k, h, T, W)
    h1 = rms(x); q, k, v = mm("qkv", h1)
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    o = mm("o", att)[0]; x2 = x + o; h2 = rms(x2)
    g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
    return x2 + mm("down", z)[0]
```

</details>

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Partial int8 tile cache (hot fused sites resident as exact int8 trits, rest 2-bit u32-LUT expand) vs all-u32 and
all-int8-trit tiles, threaded, full bitnetd layer, B=1/8/16. base/tiled/par verbatim from 10:55.
Usage: OPENBLAS_NUM_THREADS=1 python3 bench.py out.json"""
import json, os, sys, statistics
import numpy as np
import base, par, cache
base.build(); par.prep(); cache.prep()
HOT = {"u32": (), "attn": ("qkv", "o"), "down_o": ("down", "o"), "gu": ("gu",), "int8all": ("qkv", "o", "gu", "down")}
CFG = [(f"{h}_T{T}_W{W}", h, T, W) for T, W in ((64, 4), (128, 8)) for h in HOT]
res = {"seed": base.SEED, "numpy": np.__version__, "blas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS", "default"), "cpus": os.cpu_count(),
       "resident_bytes": {h: cache.resident_bytes(v) for h, v in HOT.items()}, "lut_bytes": par.LUT_U32.nbytes,
       "scratch_bytes": {f"T{T}_W{W}": par.scratch_bytes(W, T) for T, W in ((64, 4), (128, 8))}, "err": {}, "time": {}}
print(res["resident_bytes"], res["scratch_bytes"], flush=True)
xe = np.random.default_rng(base.SEED + 2).standard_normal((32, base.D)).astype(np.float32)
yw = base.layer(xe, base.wo_mm)[0]
for n, h, T, W in CFG:
    res["err"][n] = float(np.max(np.abs(cache.layer(HOT[h], xe, T, W) - yw))); print("err", n, res["err"][n], flush=True)
base.REPEATS = 5; ROUNDS = 7
for B in (1, 8, 16):
    xb = np.random.default_rng(base.SEED + 9 + B).standard_normal((B, base.D)).astype(np.float32)
    names = [c[0] for c in CFG]; fns = {n: (lambda h=h, T=T, W=W: cache.layer(HOT[h], xb, T, W)) for n, h, T, W in CFG}
    rounds = {n: [] for n in names}
    for r in range(ROUNDS):
        for n in names[r % len(names):] + names[:r % len(names)]: rounds[n].append(base.timed(fns[n])["median_ms"])
    o = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
    for n in names:
        TW = "_".join(n.split("_")[-2:]); o[n]["vs_u32_same_TW"] = o[f"u32_{TW}"]["median_ms"] / o[n]["median_ms"]
        o[n]["vs_int8all_same_TW"] = o[f"int8all_{TW}"]["median_ms"] / o[n]["median_ms"]
    res["time"][f"B{B}"] = o
    for n in names: print(B, n, {k: round(v, 3) for k, v in o[n].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
```

</details>

## Results

Resident = packed/int8 weights + gammas; total adds per-worker scratch (T64 W4 8.8 MB, T128 W8 35.4 MB) and excludes the 1 KB u32 LUT. Times are layer ms, median (min–max of 7 rounds).

| B | config | resident MB | total MB | run 1 ms | run 2 ms | vs u32 (r1 / r2) | vs int8all (r1 / r2) |
|---|---|---|---|---|---|---|---|
| 1 | `u32_T64_W4` | 17.5 | 26.3 | 17.04 (16.30–21.47) | 16.99 (16.49–20.34) | 1.00 / 1.00 | 0.44 / 0.45 |
| 1 | `attn_T64_W4` | 29.7 | 38.6 | 16.04 (14.08–17.51) | 14.72 (14.05–16.06) | 1.06 / 1.15 | 0.47 / 0.52 |
| 1 | `down_o_T64_W4` | 35.6 | 44.5 | 14.93 (13.48–16.52) | 13.70 (13.43–14.30) | 1.14 / 1.24 | 0.51 / 0.56 |
| 1 | `gu_T64_W4` | 44.0 | 52.8 | 12.41 (11.30–15.13) | 12.12 (11.59–12.55) | 1.37 / 1.40 | 0.61 / 0.63 |
| 1 | `int8all_T64_W4` | 69.6 | 78.4 | 7.58 (7.39–8.80) | 7.68 (7.40–8.31) | 2.25 / 2.21 | 1.00 / 1.00 |
| 1 | `u32_T128_W8` | 17.5 | 52.8 | 15.12 (14.87–20.88) | 15.22 (14.88–17.48) | 1.00 / 1.00 | 0.66 / 0.70 |
| 1 | `attn_T128_W8` | 29.7 | 65.1 | 14.39 (13.76–17.03) | 14.06 (13.77–15.98) | 1.05 / 1.08 | 0.70 / 0.76 |
| 1 | `down_o_T128_W8` | 35.6 | 71.0 | 13.57 (13.06–16.83) | 13.70 (13.12–15.47) | 1.11 / 1.11 | 0.74 / 0.78 |
| 1 | `gu_T128_W8` | 44.0 | 79.4 | 12.55 (12.32–16.29) | 13.06 (12.85–14.38) | 1.20 / 1.16 | 0.80 / 0.82 |
| 1 | `int8all_T128_W8` | 69.6 | 104.9 | 10.05 (9.61–12.37) | 10.70 (10.18–10.94) | 1.50 / 1.42 | 1.00 / 1.00 |
| 8 | `u32_T64_W4` | 17.5 | 26.3 | 18.32 (17.85–21.92) | 18.02 (17.60–18.30) | 1.00 / 1.00 | 0.60 / 0.59 |
| 8 | `attn_T64_W4` | 29.7 | 38.6 | 16.26 (15.97–19.87) | 16.21 (15.87–16.31) | 1.13 / 1.11 | 0.67 / 0.66 |
| 8 | `down_o_T64_W4` | 35.6 | 44.5 | 15.81 (15.62–18.91) | 15.75 (15.33–16.33) | 1.16 / 1.14 | 0.69 / 0.68 |
| 8 | `gu_T64_W4` | 44.0 | 52.8 | 14.87 (14.36–15.78) | 14.27 (14.14–15.34) | 1.23 / 1.26 | 0.74 / 0.75 |
| 8 | `int8all_T64_W4` | 69.6 | 78.4 | 10.94 (10.63–11.42) | 10.69 (10.56–11.10) | 1.67 / 1.69 | 1.00 / 1.00 |
| 8 | `u32_T128_W8` | 17.5 | 52.8 | 14.67 (14.01–17.62) | 12.59 (12.35–16.48) | 1.00 / 1.00 | 0.58 / 0.60 |
| 8 | `attn_T128_W8` | 29.7 | 65.1 | 13.22 (12.68–15.45) | 11.49 (11.30–12.11) | 1.11 / 1.10 | 0.65 / 0.66 |
| 8 | `down_o_T128_W8` | 35.6 | 71.0 | 12.83 (12.41–17.79) | 11.03 (10.78–13.84) | 1.14 / 1.14 | 0.67 / 0.69 |
| 8 | `gu_T128_W8` | 44.0 | 79.4 | 11.34 (11.15–12.40) | 10.19 (9.62–10.55) | 1.29 / 1.24 | 0.76 / 0.74 |
| 8 | `int8all_T128_W8` | 69.6 | 104.9 | 8.57 (8.20–9.11) | 7.57 (7.34–8.10) | 1.71 / 1.66 | 1.00 / 1.00 |
| 16 | `u32_T64_W4` | 17.5 | 26.3 | 19.07 (18.81–23.58) | 19.57 (18.98–20.82) | 1.00 / 1.00 | 0.65 / 0.62 |
| 16 | `attn_T64_W4` | 29.7 | 38.6 | 17.58 (17.01–19.51) | 17.74 (17.34–18.05) | 1.08 / 1.10 | 0.70 / 0.68 |
| 16 | `down_o_T64_W4` | 35.6 | 44.5 | 17.82 (16.86–22.51) | 17.17 (16.82–20.89) | 1.07 / 1.14 | 0.69 / 0.70 |
| 16 | `gu_T64_W4` | 44.0 | 52.8 | 16.23 (15.29–16.83) | 15.81 (15.48–18.02) | 1.18 / 1.24 | 0.76 / 0.76 |
| 16 | `int8all_T64_W4` | 69.6 | 78.4 | 12.32 (11.82–13.50) | 12.07 (11.96–13.07) | 1.55 / 1.62 | 1.00 / 1.00 |
| 16 | `u32_T128_W8` | 17.5 | 52.8 | 17.06 (13.16–22.11) | 13.33 (13.14–15.23) | 1.00 / 1.00 | 0.53 / 0.64 |
| 16 | `attn_T128_W8` | 29.7 | 65.1 | 12.78 (11.90–21.74) | 12.18 (11.79–14.07) | 1.33 / 1.09 | 0.70 / 0.70 |
| 16 | `down_o_T128_W8` | 35.6 | 71.0 | 12.23 (11.31–15.56) | 12.13 (11.58–14.95) | 1.40 / 1.10 | 0.73 / 0.70 |
| 16 | `gu_T128_W8` | 44.0 | 79.4 | 11.29 (10.74–14.13) | 10.87 (10.58–12.12) | 1.51 / 1.23 | 0.79 / 0.78 |
| 16 | `int8all_T128_W8` | 69.6 | 104.9 | 8.96 (8.14–11.35) | 8.49 (8.21–9.06) | 1.90 / 1.57 | 1.00 / 1.00 |

Error: max abs vs weight-only W1.58 = 1.9e-6 for all 10 configs (both runs).

## Verdict

**not-viable.** Size ratios vs all-int8 resident: u32 3.98x, attn 2.34x, down_o 1.95x, gu 1.58x. Speed vs all-u32 for those three partial splits: attn 1.05–1.33x, down_o 1.07–1.40x, gu 1.16–1.40x (one 1.51x outlier at B=16 T128 W8 that did not reproduce: 1.23x in run 2). The only split that stays ≥1.5x smaller than all-int8 (gu, 1.58x) is not reliably ≥1.5x faster than u32. A partial cache is just a point on the line between the two endpoints (u32: smallest, all-int8 trit tiles: 1.42–2.25x faster at 3.98x the bytes), so it adds a memory/speed dial for i2sd but not a free win. Keep u32-LUT expand where memory matters, all-int8 where speed matters.

**Caveat:** synthetic weights/tokens; single-process numpy on 8 cores; threaded timings shift with box load (B=16 T128 W8 u32 round spread 13.2–22.1 ms in run 1).

## Pair files this wave

At Quant Chief 14:05 MT (`last_run` 21:05Z), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was 08ecc8c (the 10:55 note); no other commit in the last 6 hours.
- No Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours; Chief's LLC skip paths (JuniorHome 612f903, 3249cbb) are in another repo.
- This note adds one file under `junior_osai/vault/findings/`.

## Next question

The expand/cache line on this layer is mapped (u32 ↔ int8 trade is linear). Move off it: does **A8 activation quantization into an int8 × int8 → int32 matmul** (numpy `int32` matmul on int8 trit tiles vs f32 tile matmul) change the speed side of the threaded path at B=8/16, or does numpy's lack of an int8 BLAS make it slower than the f32 cast-and-matmul?

close: sha256(prev || this)[:16] = 8d4e077bb997c999
