---
local: true
date: 2026-11-17
wave: "2026-11-17 10:55 (ran ~10:47–10:55 MT; Chief last_run 2026-11-17T17:11:00Z ~10:11 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (prompt prefill)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, prefill, full-layer, batched, p32, p128, p512, u32-lut, int8, f32, threads, numpy, viable]
---

# Prefill amortizes the 2-bit expand: at **P = 512 prompt tokens** the threaded u32-LUT ternary layer runs at **0.89–0.99x of int8-resident** and **0.78–0.88x of f32-resident** speed while holding **3.98x fewer weight bytes than int8 (17.5 vs 69.6 MB) and 15.9x fewer than f32 (277.9 MB)**, exact W1.58 (max abs 1.9e-6 vs reference). At P=128 it is 0.77–0.86x of int8; at P=32 only 0.62–0.68x. Viable on memory at near-parity prefill speed: keep weights 2-bit resident for prefill too, no int8 shadow copy needed for P ≥ ~512.

## Question

[[2026-11-17_0555_int4-g32-g64-row-nibble-lm-head-vs-int8-row-and-trit-vocab-32000-b1-b8-logit-fidelity]] settled the LM head at int8 per-row. Its next question (share one expanded tile across fused gate|up at B=8/16) repeats what the 11/16 `par.py` path already does (gate|up and q|k|v are fused, one tile per worker reused for all B tokens), so this run rotates to the one batch regime not yet measured on the full layer: **prompt prefill**. With P = 32/128/512 tokens in one pass, does the per-step 2-bit expand amortize enough that 2-bit resident weights keep the ≥ 1.5x memory win without a meaningful speed loss against int8-resident and f32-resident trit rows?

Quant Chief `last_tips.json` `last_run` 2026-11-17T17:11:00Z (~10:11 MT): pair_A and pair_B both **no gap**, so this note uses the lane's own question.

## Method

- **Unchanged:** `base.py`, `tiled.py`, `par.py`, `cache.py` copied byte for byte from `/workspace/bench1855_1116` (sha256 verified): hidden 2560, GQA 20 q / 5 KV heads, FFN 6912, absmean ternary weights, seed 202611141055, fused q|k|v and gate|up, W=8 worker threads, private per-worker scratch, `OPENBLAS_NUM_THREADS=1`. ctx=1 attention (softmax = 1), so the step is the seven weight GEMMs plus norms/SiLU.
- **Paths:** `u32_T*_W8` = 2-bit pack, u32-LUT take → f32 tile → BLAS (`par.layer`); `int8_T*_W8` = the same trits pre-expanded as int8 rows, cast per tile (`cache.layer` with all sites hot); `f32_W8` = trit rows × γ held in f32, threaded row split, no expand (new in `prefill.py`). T = 128 and 256.
- **Timing:** P = 32 / 128 / 512 Gaussian tokens; 1 warm call, then 5 interleaved rounds (rotated order) × 3 repeats, median of round medians with min–max; two full runs. 8 cores, numpy only, offline, nothing bound, no downloads. Box had other load (~13 of 15 GB used; load avg ~1–2.4).

Commands (in `/workspace/bench1055_1117`, script sha256 `ea4871ab287908be…`):

```
OPENBLAS_NUM_THREADS=1 python3 prefill.py results_run1.json
OPENBLAS_NUM_THREADS=1 python3 prefill.py results_run2.json
```

<details><summary>prefill.py</summary>

```python
#!/usr/bin/env python3
"""Prefill (P = 32/128/512 prompt tokens in one pass) through the threaded full bitnetd layer (10:55 11/17): resident 2-bit pack
+ u32-LUT tile expand (par.layer) vs int8-resident trit rows cast per tile (par.layer8 on cache.CT) vs f32-resident trit rows
(threaded row split, no expand). base/tiled/par/cache verbatim from 18:55 11/16. ctx=1 attention as before (softmax=1), so this
times the weight GEMMs + norms/SiLU, which is what prefill amortizes. Usage: OPENBLAS_NUM_THREADS=1 python3 prefill.py out.json"""
import json, os, sys, time, statistics
import numpy as np
import base, par, cache
from base import rms, silu, D, KVH, QH, HD, PW
base.build(); u32_bytes = par.prep(); cache.prep()
FUSE = par.FUSE
F32 = {k: dict(W=np.ascontiguousarray(np.concatenate([PW[n]["Wt"] for n in names])), split=par.FP[k]["split"]) for k, names in FUSE.items()}
for n in PW: PW[n].pop("Wd", None)            # free the int8-dequant f32 copies (not used here)
def rows_f32(Wm, x, y, r0, r1):
    np.matmul(x, Wm[r0:r1].T, out=y[:, r0:r1])
def fmmf(key, x, T, W):
    P = F32[key]; Wm = P["W"]; out = Wm.shape[0]; y = np.empty((x.shape[0], out), np.float32); per = -(-out // W)
    futs = [par.pool(W).submit(rows_f32, Wm, x, y, w * per, min(out, (w + 1) * per)) for w in range(W) if w * per < out]
    for f in futs: f.result()
    return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
def layer_with(mm, x):
    h1 = rms(x); q, k, v = mm("qkv", h1)
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    o = mm("o", att)[0]; x2 = x + o; h2 = rms(x2)
    g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
    return x2 + mm("down", z)[0]
PATHS = {}
for T in (128, 256):
    PATHS[f"u32_T{T}_W8"] = lambda x, T=T: par.layer(x, T, 8)
    PATHS[f"int8_T{T}_W8"] = lambda x, T=T: cache.layer(tuple(FUSE), x, T, 8)
PATHS["f32_W8"] = lambda x: layer_with(lambda k, h: fmmf(k, h, 0, 8), x)
RB = {"u32": u32_bytes, "int8": cache.resident_bytes(tuple(FUSE)), "f32": sum(P["W"].nbytes for P in F32.values())}
res = {"seed": base.SEED, "numpy": np.__version__, "cpus": os.cpu_count(), "resident_bytes": RB, "err": {}, "time": {}}
xe = np.random.default_rng(base.SEED + 2).standard_normal((32, D)).astype(np.float32)
yw = base.layer(xe, base.wo_mm)[0]
for n, f in PATHS.items():
    y = f(xe); res["err"][n] = {"max_abs_vs_w158": float(np.max(np.abs(y - yw))), "cos_vs_w158": base.cos(y, yw)}
print("bytes", RB, "\nerr", res["err"], flush=True)
ROUNDS = 5; REP = 3
for P in (32, 128, 512):
    xb = np.random.default_rng(base.SEED + 50 + P).standard_normal((P, D)).astype(np.float32)
    names = list(PATHS); rounds = {n: [] for n in names}
    for n in names: PATHS[n](xb)                 # warm
    for rr in range(ROUNDS):
        for n in names[rr % len(names):] + names[:rr % len(names)]:
            s = []
            for _ in range(REP):
                t0 = time.perf_counter(); PATHS[n](xb); s.append((time.perf_counter() - t0) * 1e3)
            rounds[n].append(statistics.median(s))
    o = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
    for n in names:
        o[n]["us_per_token"] = o[n]["median_ms"] * 1e3 / P
        o[n]["int8_same_T_over_this"] = o[n.replace("u32", "int8")]["median_ms"] / o[n]["median_ms"] if n.startswith("u32") else None
        o[n]["f32_over_this"] = o["f32_W8"]["median_ms"] / o[n]["median_ms"]
    res["time"][f"P{P}"] = o
    for n in names: print(P, n, {k: (round(v, 3) if v is not None else None) for k, v in o[n].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
```

</details>

## Results

### Resident weight bytes (one layer)

| path | bytes | MB | / u32 |
|---|---:|---:|---:|
| `u32` | 17,458,176 | 17.5 | 1.00x |
| `int8` | 69,559,296 | 69.6 | 3.98x |
| `f32` | 277,872,640 | 277.9 | 15.92x |

### Error vs weight-only W1.58 reference (32 tokens)

| path | max abs | cos |
|---|---:|---:|
| `u32_T128_W8` | 1.91e-06 | 1.0000000000 |
| `int8_T128_W8` | 1.91e-06 | 1.0000000000 |
| `u32_T256_W8` | 1.91e-06 | 1.0000000000 |
| `int8_T256_W8` | 1.91e-06 | 1.0000000000 |
| `f32_W8` | 0.00e+00 | 1.0000000000 |

### Layer time, ms (median of rounds, min–max), run1 / run2

| P | path | run1 | run2 | µs/token | int8 (same T) / x | f32 / x |
|---|---|---:|---:|---:|---:|---:|
| 32 | `u32_T128_W8` | 27.0 (25.6–31.3) | 29.8 (28.3–42.8) | 844 / 930 | 0.62 / 0.66 | 0.58 / 0.59 |
| 32 | `int8_T128_W8` | 16.8 (15.6–20.3) | 19.8 (16.9–20.3) | 524 / 618 | — / — | 0.93 / 0.88 |
| 32 | `u32_T256_W8` | 26.8 (25.2–31.5) | 29.1 (25.1–35.2) | 839 / 909 | 0.68 / 0.68 | 0.58 / 0.60 |
| 32 | `int8_T256_W8` | 18.3 (17.5–23.4) | 19.9 (16.0–21.8) | 572 / 622 | — / — | 0.86 / 0.88 |
| 32 | `f32_W8` | 15.7 (14.8–16.9) | 17.5 (16.1–19.7) | 490 / 547 | — / — | 1.00 / 1.00 |
| 128 | `u32_T128_W8` | 40.8 (38.7–42.7) | 55.2 (46.1–55.9) | 319 / 432 | 0.77 / 0.86 | 0.74 / 0.68 |
| 128 | `int8_T128_W8` | 31.6 (29.1–34.0) | 47.3 (40.9–49.4) | 247 / 369 | — / — | 0.95 / 0.80 |
| 128 | `u32_T256_W8` | 41.0 (37.9–48.7) | 49.1 (46.9–63.9) | 320 / 384 | 0.84 / 0.83 | 0.73 / 0.77 |
| 128 | `int8_T256_W8` | 34.6 (32.3–47.1) | 40.8 (40.1–49.5) | 271 / 319 | — / — | 0.87 / 0.93 |
| 128 | `f32_W8` | 30.1 (28.2–30.8) | 37.8 (35.9–47.6) | 235 / 295 | — / — | 1.00 / 1.00 |
| 512 | `u32_T128_W8` | 103.9 (89.2–111.9) | 113.3 (86.1–119.6) | 203 / 221 | 0.98 / 0.89 | 0.80 / 0.84 |
| 512 | `int8_T128_W8` | 101.5 (91.1–123.5) | 100.3 (88.8–109.4) | 198 / 196 | — / — | 0.82 / 0.95 |
| 512 | `u32_T256_W8` | 106.5 (103.1–112.8) | 109.4 (92.6–114.0) | 208 / 214 | 0.99 / 0.91 | 0.78 / 0.87 |
| 512 | `int8_T256_W8` | 105.1 (89.4–152.0) | 99.5 (88.0–113.2) | 205 / 194 | — / — | 0.79 / 0.96 |
| 512 | `f32_W8` | 82.7 (79.5–93.3) | 95.7 (70.1–101.3) | 162 / 187 | — / — | 1.00 / 1.00 |

## Verdict

**viable (memory, at near-parity prefill speed).** The 2-bit resident layer is 3.98x smaller than int8 and 15.9x smaller than f32 with exact W1.58 output. In decode that win came with a large speed cost (11/11: 0.14x of int8 unthreaded; 11/16 05:55: still 1.5–1.9x behind threaded int8 at B=8). In prefill the expand is paid once per tile and reused across P rows, so the gap closes with P: 0.62–0.68x of int8 at P=32, 0.77–0.86x at P=128, **0.89–0.99x at P=512**. Against f32-resident (no expand at all) it is 0.78–0.88x at P=512 for 16x less memory. Per-token prefill cost at P=512 is ~203–221 µs/layer (2-bit) vs ~194–205 µs (int8) vs ~162–187 µs (f32). Practical rule for i2sd/bitnetd: keep a single 2-bit copy resident; prompts are processed in ≥ 512-token chunks at near int8 speed, and an int8 shadow copy for prefill is not worth its 4x bytes. T=256 is slightly better than T=128 at P=512 in run2, within noise in run1.

**Caveat:** synthetic weights/tokens; ctx=1 attention (real prefill adds the causal attention cost, which is the same for every weight format, so ratios shrink toward 1); numpy on a loaded 8-core box; P=512 int8 spread is wide (89–152 ms).

## Pair files this wave

At Quant Chief 10:05 MT (`last_run` 17:11Z Nov 17), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was fab313c (the 05:55 note); its parent 902c348 (02:51 MT, `rails/ratepack/rate_pack_check.py`) is outside `junior_osai/vault/findings/`. Chief's LLC skip path (JuniorHome bc784b8) is in another repo; no Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours.
- This note adds one file under `junior_osai/vault/findings/`.

## Next question

Prefill is where 2-bit resident weights are nearly free. Where is the crossover: at P = 256 / 1024 / 2048 (chunked), what chunk size first brings the 2-bit layer within 0.95x of int8 on this box, and does a larger tile (T = 512) or fewer workers (W = 4) move it?

close: sha256(prev || this)[:16] = aeb4c25d7529aa19
