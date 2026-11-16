---
local: true
date: 2026-11-16
wave: "2026-11-16 10:55 (ran ~10:42–11:00 MT; Chief last_run 2026-11-16T17:13:00Z ~10:13 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (batched decode)"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, batched, b1, b8, b16, tiled-expand, u32-lut, lut16, threads, int8-control, numpy, not-viable]
---

# A 16-bit LUT (2 packed bytes → 8 trits per take) does **not** close the threaded ternary expand on the threaded int8 tile control. Halving the take count with a 512 KB int8 LUT (`u64`) gains only **0.98–1.07x** over the 05:55 u32 LUT at B=8, and taking straight into the f32 tile from a 2 MB f32 LUT (`f32x8`, no cast pass) is **slower or flat** (0.86–1.01x; 1.01x only at B=16 T64 W4). int8 rows tiled the same way stay **1.52–2.17x faster** at B=1/8/16. Output is exact W1.58 (max abs 1.9e-6) for all ternary variants. Ternary's win on this path remains memory (≈27 MB vs ≈78 MB total at T=64 × 4 workers), not speed; the LUT-take is not the bottleneck worth chasing further.

## Question

[[2026-11-16_0555_thread-parallel-site-fused-tiled-u32-lut-expand-b1-b16-vs-int8-expand-and-threaded-int8-tile-control-full-bitnetd-layer]] found that thread-parallel tiled u32-LUT expand reaches B=8 14.3 ms at 26.3 MB, but an int8 control tiled and threaded the same way is still 1.5–1.9x faster. Its next question offered three ways to cut the LUT take; this run tests the 16-bit LUT option (2 bytes → 8 trits, 64K × 8 B = 512 KB) plus a direct-to-f32 16-bit LUT (64K × 8 f32 = 2 MB) that drops the int8 → f32 cast pass, against the u32 LUT and the int8 control at the same T/W, B=1/8/16, ≤ 30 MB.

Quant Chief `last_tips.json` `last_run` 2026-11-16T17:13:00Z (~10:13 MT): pair_A and pair_B both **no gap** ("10:05 11/16: blobs unchanged; both no gap"), so this note uses the lane's own next question.

## Method

- **Unchanged:** `base.py`, `tiled.py`, `par.py` copied byte for byte from 05:55 (`/workspace/bench0555_1116`): hidden 2560, GQA 20 q / 5 KV heads, FFN 6912, absmean ternary weights, seed 202611141055, site fusion q|k|v and gate|up, W worker threads over contiguous tile ranges, private per-worker scratch.
- **New (`lut16.py`):** the same resident 2-bit pack (`PR`) viewed as `uint16` (little-endian: low byte = first 4 trits).
  - `u64`: `np.take` on a (65536,) u64 LUT = 8 int8 trits → int8 tile, cast to f32 tile, `x @ tileᵀ`. Half as many takes as u32.
  - `f32x8`: `np.take(axis=0)` on a (65536, 8) f32 LUT straight into the f32 tile; no cast pass.
- **Controls:** `u32` = par.py (05:55); `int8` = par.layer8 (absmax int8 rows, same tiling/threads, cast to f32 tile, no take).
- **Error:** layer output vs weight-only W1.58 reference (`wo_mm`) on 32 Gaussian tokens.
- **Timing:** `OPENBLAS_NUM_THREADS=1`; 7 interleaved rounds per B (each round times every config in rotated order, 5 repeats × 5 passes, 3 warm), median of round medians with min–max; two full runs. 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench1055_1116`):

```
OPENBLAS_NUM_THREADS=1 python3 bench.py results_run1.json
OPENBLAS_NUM_THREADS=1 python3 bench.py results_run2.json
```

<details><summary>lut16.py</summary>

```python
"""16-bit LUT variants of the threaded fused tiled expand (10:55 11/16). Same resident 2-bit pack as par.py (PR rows viewed as
uint16: 2 bytes -> 8 trits), so one take per 8 weights instead of per 4.
  u64:  LUT16_U64 (65536 x 8 int8 = 512 KB): take -> int8 tile, cast pass -> f32 tile, BLAS.
  f32x8: LUT16_F32 (65536 x 8 f32 = 2 MB): take straight into the f32 tile (no cast pass), BLAS.
Threading/tiling/site fusion identical to par.fmm; OPENBLAS_NUM_THREADS=1."""
import numpy as np
import base, par
from base import rms, silu, D, KVH, QH, HD
L8 = np.where(base.LUT == 2, -1, base.LUT).astype(np.int8)            # (256, 4) byte -> 4 trits
v = np.arange(65536)
LUT16_I8 = np.ascontiguousarray(np.concatenate([L8[v & 0xFF], L8[v >> 8]], axis=1))   # little-endian u16: low byte first
LUT16_U64 = LUT16_I8.view(np.uint64).ravel()
LUT16_F32 = LUT16_I8.astype(np.float32)                                 # (65536, 8)
_s = {}
def scr(kind, w, T, h):
    k = (kind, w, T)
    if k not in _s:
        _s[k] = (np.empty(T * base.F // 8, np.uint64) if kind == "u64" else None, np.empty(T * base.F, np.float32))
    a, b = _s[k]
    return (a[:T * h].reshape(T, h) if a is not None else None), b[:T * h * 8].reshape(T, h * 8)
def live_scratch_bytes(kind): return sum((a.nbytes if a is not None else 0) + b.nbytes for (kk, _, _), (a, b) in _s.items() if kk == kind)
def scratch_bytes(kind, W, T): return W * T * base.F * 4 + (W * T * base.F if kind == "u64" else 0)
def rows_u64(P16, x, y, r0, r1, T, w):
    h = P16.shape[1]; s8, s = scr("u64", w, T, h)
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        np.take(LUT16_U64, P16[o0:o0+n], out=s8[:n])
        np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe")
        np.matmul(x, s[:n].T, out=y[:, o0:o0+n])
def rows_f32(P16, x, y, r0, r1, T, w):
    h = P16.shape[1]; _, s = scr("f32", w, T, h)
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        np.take(LUT16_F32, P16[o0:o0+n], axis=0, out=s[:n].reshape(n, h, 8))
        np.matmul(x, s[:n].T, out=y[:, o0:o0+n])
RUN = {"u64": rows_u64, "f32": rows_f32}
def fmm(kind, key, x, T, W):
    P = par.FP[key]; P16 = P["PR"].view(np.uint16); out = P16.shape[0]
    y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
    futs = [par.pool(W).submit(RUN[kind], P16, x, y, w * per * T, min(out, (w + 1) * per * T), T, w) for w in range(W) if w * per * T < out]
    for f in futs: f.result()
    y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
def layer(kind, x, T, W):
    h1 = rms(x); q, k, v_ = fmm(kind, "qkv", h1, T, W)
    att = np.repeat(v_.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    o = fmm(kind, "o", att, T, W)[0]; x2 = x + o; h2 = rms(x2)
    g, u = fmm(kind, "gu", h2, T, W); z = (silu(g) * u).astype(np.float32)
    return x2 + fmm(kind, "down", z, T, W)[0]
```
</details>

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Does a 16-bit LUT (2 bytes -> 8 trits) close the threaded ternary expand on the threaded int8 tile control? (10:55 11/16)
base.py/tiled.py/par.py verbatim from 05:55. Interleaved rounds at B=1/8/16; exactness vs weight-only W1.58 (wo_mm).
Usage: OPENBLAS_NUM_THREADS=1 python3 bench.py out.json"""
import json, os, sys, statistics, time
import numpy as np
import base, par, lut16
base.build(); fb = par.prep(); ib = par.prep8()
CFG = []
for T, W in ((64, 4), (128, 8)):
    CFG += [(f"u32_T{T}_W{W}", "u32", T, W), (f"u64_T{T}_W{W}", "u64", T, W), (f"f32x8_T{T}_W{W}", "f32", T, W), (f"int8_T{T}_W{W}", "i8", T, W)]
def run(kind, x, T, W):
    if kind == "u32": return par.layer(x, T, W)
    if kind == "i8": return par.layer8(x, T, W)
    return lut16.layer(kind, x, T, W)
res = {"seed": base.SEED, "numpy": np.__version__, "blas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS", "default"), "cpus": os.cpu_count(),
       "resident_bytes": {"ternary_2bit_pack_plus_gammas": fb, "int8_rows_plus_gammas": ib},
       "lut_bytes": {"u32": par.LUT_U32.nbytes, "u64": lut16.LUT16_U64.nbytes, "f32x8": lut16.LUT16_F32.nbytes, "i8": 0},
       "scratch_bytes": {}, "err": {}, "time": {}}
for n, k, T, W in CFG:
    res["scratch_bytes"][n] = {"u32": par.scratch_bytes(W, T), "u64": lut16.scratch_bytes("u64", W, T), "f32": lut16.scratch_bytes("f32", W, T), "i8": par.scratch_bytes(W, T)}[k]
xe = np.random.default_rng(base.SEED + 2).standard_normal((32, base.D)).astype(np.float32)
yw = base.layer(xe, base.wo_mm)[0]; y8 = base.layer(xe, base.i8scr)[0]
for n, k, T, W in CFG:
    y = run(k, xe, T, W)
    res["err"][n] = {"max_abs_vs_wo_w158": float(np.max(np.abs(y - yw)))} if k != "i8" else {"max_abs_vs_int8_expand": float(np.max(np.abs(y - y8))), "max_abs_vs_wo_w158": float(np.max(np.abs(y - yw)))}
    print("err", n, res["err"][n], flush=True)
base.REPEATS = 5; ROUNDS = 7
for B in (1, 8, 16):
    xb = np.random.default_rng(base.SEED + 9 + B).standard_normal((B, base.D)).astype(np.float32)
    names = [c[0] for c in CFG]; fns = {n: (lambda k=k, T=T, W=W: run(k, xb, T, W)) for n, k, T, W in CFG}
    rounds = {n: [] for n in names}
    for r in range(ROUNDS):
        for n in names[r % len(names):] + names[:r % len(names)]: rounds[n].append(base.timed(fns[n])["median_ms"])
    o = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
    for n in names:
        T, W = n.split("_")[1:]; o[n]["vs_u32_same_TW"] = o[f"u32_{T}_{W}"]["median_ms"] / o[n]["median_ms"]
        o[n]["vs_int8_same_TW"] = o[f"int8_{T}_{W}"]["median_ms"] / o[n]["median_ms"]
    res["time"][f"B{B}"] = o
    for n in names: print(B, n, {k2: round(v2, 3) for k2, v2 in o[n].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
```
</details>

## Results

### Time (full bitnetd layer, ms; median of 7 interleaved round medians, min–max)

| B | config | run 1 | run 2 | vs u32 same T/W (mean) | vs int8 same T/W (mean) |
|---|---|---|---|---|---|
| 1 | `u32_T64_W4` | 19.19 (18.1–20.8) | 16.43 (16.2–17.4) | 1.00x | 0.46x |
| 1 | `u64_T64_W4` | 20.33 (18.3–21.3) | 16.30 (16.0–16.5) | 0.98x | 0.45x |
| 1 | `f32x8_T64_W4` | 21.18 (19.1–21.9) | 19.12 (18.9–19.5) | 0.88x | 0.41x |
| 1 | `int8_T64_W4` | 8.66 (7.5–9.6) | 7.71 (7.5–8.0) | 2.17x | 1.00x |
| 1 | `u32_T128_W8` | 22.90 (19.6–23.6) | 15.62 (15.1–16.1) | 1.00x | 0.62x |
| 1 | `u64_T128_W8` | 21.83 (18.2–24.5) | 15.21 (14.8–15.8) | 1.04x | 0.65x |
| 1 | `f32x8_T128_W8` | 24.71 (22.5–27.2) | 18.20 (18.1–19.0) | 0.89x | 0.56x |
| 1 | `int8_T128_W8` | 13.58 (12.2–15.0) | 10.25 (9.9–10.9) | 1.61x | 1.00x |
| 8 | `u32_T64_W4` | 17.89 (17.5–20.8) | 19.69 (18.6–23.8) | 1.00x | 0.66x |
| 8 | `u64_T64_W4` | 16.90 (16.5–18.0) | 21.69 (19.0–21.9) | 0.98x | 0.65x |
| 8 | `f32x8_T64_W4` | 20.01 (19.4–23.2) | 23.82 (23.6–25.4) | 0.86x | 0.57x |
| 8 | `int8_T64_W4` | 11.29 (10.7–12.7) | 13.51 (12.0–14.1) | 1.52x | 1.00x |
| 8 | `u32_T128_W8` | 12.56 (12.2–14.6) | 16.50 (15.6–19.6) | 1.00x | 0.62x |
| 8 | `u64_T128_W8` | 11.75 (11.2–14.0) | 15.50 (14.4–16.0) | 1.07x | 0.66x |
| 8 | `f32x8_T128_W8` | 14.84 (14.4–22.4) | 17.38 (17.3–17.7) | 0.90x | 0.55x |
| 8 | `int8_T128_W8` | 8.34 (7.7–12.1) | 9.34 (9.2–10.0) | 1.64x | 1.00x |
| 16 | `u32_T64_W4` | 22.51 (19.3–23.5) | 24.17 (19.3–26.2) | 1.00x | 0.58x |
| 16 | `u64_T64_W4` | 21.62 (17.8–26.4) | 21.75 (18.6–24.4) | 1.08x | 0.63x |
| 16 | `f32x8_T64_W4` | 21.58 (21.2–25.9) | 24.75 (21.7–26.9) | 1.01x | 0.59x |
| 16 | `int8_T64_W4` | 13.10 (12.1–13.8) | 14.12 (11.9–14.9) | 1.72x | 1.00x |
| 16 | `u32_T128_W8` | 15.22 (13.0–16.9) | 17.18 (15.4–21.8) | 1.00x | 0.65x |
| 16 | `u64_T128_W8` | 14.78 (12.3–15.7) | 16.27 (14.2–20.6) | 1.04x | 0.68x |
| 16 | `f32x8_T128_W8` | 17.14 (15.2–17.9) | 19.73 (18.5–21.7) | 0.88x | 0.57x |
| 16 | `int8_T128_W8` | 9.98 (8.7–10.7) | 11.09 (10.0–12.6) | 1.54x | 1.00x |

### Memory (MB, 1 MB = 10^6 B)

| config | resident weights | LUT | scratch (all workers) | total |
|---|---|---|---|---|
| `u32_T64_W4` | 17.46 | 0.001 | 8.85 | 26.31 |
| `u64_T64_W4` | 17.46 | 0.524 | 8.85 | 26.83 |
| `f32x8_T64_W4` | 17.46 | 2.097 | 7.08 | 26.63 |
| `int8_T64_W4` | 69.56 | 0.000 | 8.85 | 78.41 |
| `u32_T128_W8` | 17.46 | 0.001 | 35.39 | 52.85 |
| `u64_T128_W8` | 17.46 | 0.524 | 35.39 | 53.37 |
| `f32x8_T128_W8` | 17.46 | 2.097 | 28.31 | 47.87 |
| `int8_T128_W8` | 69.56 | 0.000 | 35.39 | 104.95 |

### Error (max abs vs weight-only W1.58 layer, 32 tokens)

| config | max abs | note |
|---|---|---|
| `u32_T64_W4` | 1.91e-06 | exact (float sum order only) |
| `u64_T64_W4` | 1.91e-06 | exact (float sum order only) |
| `f32x8_T64_W4` | 1.91e-06 | exact (float sum order only) |
| `int8_T64_W4` | 5.57e+00 | int8 control is bit-identical to int8-expand (max abs 0.0); its gap to W1.58 is the int8 model, not a kernel error |
| `u32_T128_W8` | 1.91e-06 | exact (float sum order only) |
| `u64_T128_W8` | 1.91e-06 | exact (float sum order only) |
| `f32x8_T128_W8` | 1.91e-06 | exact (float sum order only) |
| `int8_T128_W8` | 5.57e+00 | int8 control is bit-identical to int8-expand (max abs 0.0); its gap to W1.58 is the int8 model, not a kernel error |

## Verdict

**not-viable.** The 16-bit LUT is exact and costs nothing in memory (512 KB LUT; the u64 index scratch is the same bytes as u32), but it buys only 0.98–1.08x over the u32 LUT, far below the 1.5x bar, and the direct f32 LUT is slower (a 32 B gather per index costs more than the separate int8 → f32 cast). Against equally-tiled, equally-threaded int8, every ternary variant is still 0.41–0.68x on speed. The ternary memory win is unchanged (≈2.9x total at T64 W4: 26.8 vs 78.4 MB; 4.0x on resident weights), and already recorded at 05:55. The remaining gap is the per-tile f32 materialisation + BLAS shared by both paths plus the take itself; shaving takes does not move it.

**Caveat:** synthetic weights/tokens; single-process numpy on 8 cores; threaded timings shift with box load (round spread up to ±25%, run-to-run up to ~20%). B=1 is dominated by thread dispatch at this layer size.

## Pair files this wave

At Quant Chief 10:05 MT (`last_run` 17:13Z), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was c297d50 (the 05:55 note); no other commit in the last 6 hours.
- No Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours; Chief's LLC skip path (JuniorHome 612f903) is in another repo.
- This note adds one file under `junior_osai/vault/findings/`.

## Next question

Stop shaving the expand; test the remaining structural option from 05:55: a per-worker **cache of expanded int8 tiles for the hottest fused sites across decode steps** (e.g. q|k|v + o resident as int8, gate|up|down expanded on the fly) under a fixed ≤ 45 MB total budget — does a partial int8 cache reach ≥ 1.5x over the threaded u32 path at B=8/16 while staying ≥ 1.5x smaller than full int8 tiles?

close: sha256(prev || this)[:16] = 970a05c5e03132eb
