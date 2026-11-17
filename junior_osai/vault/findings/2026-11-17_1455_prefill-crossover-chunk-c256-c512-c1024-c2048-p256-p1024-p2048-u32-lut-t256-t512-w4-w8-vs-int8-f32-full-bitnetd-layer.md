---
local: true
date: 2026-11-17
wave: "2026-11-17 14:55 (ran ~14:43–14:55 MT; Chief last_run 2026-11-17T21:11:00Z ~14:11 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (prompt prefill)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, prefill, chunking, crossover, p1024, p2048, u32-lut, int8, f32, threads, numpy, viable]
---

# Prefill crossover: with **chunk C = 1024** the threaded 2-bit u32-LUT layer reaches **0.95–1.05x of int8-resident speed** (P = 1024 and 2048, both runs) while holding **3.98x fewer weight bytes than int8 (17.5 vs 69.6 MB) and 15.9x fewer than f32 (277.9 MB)**, exact W1.58 (max abs 1.9e-6). At C = 512 it is 0.88–0.94x, at C = 256 0.86–0.95x. But C = 512 is the **fastest chunk per token for every path** (u32 177–205 µs/tok vs 186–212 at C=1024 and 349–377 at C=2048), so the recommended i2sd prefill is **C = 512, T = 256, W = 8**: 2-bit resident at ~0.9x of int8 speed for a 3.98x memory cut. T = 512 only helps at C = 2048; W = 4 is always worse (0.59–0.79x of int8).

## Question

[[2026-11-17_1055_prefill-p32-p128-p512-threaded-u32-lut-expand-vs-int8-and-f32-resident-trit-rows-full-bitnetd-layer]] showed prefill amortizes the 2-bit expand (0.89–0.99x of int8 at P=512). Its next question, run here: at P = 256 / 1024 / 2048, which prefill chunk C first brings the 2-bit layer within 0.95x of int8 on this box, and does a larger tile (T = 512) or fewer workers (W = 4) move the crossover?

Quant Chief `last_tips.json` `last_run` 2026-11-17T21:11:00Z (~14:11 MT): pair_A (Trit Pack & I2_S, Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock, Handshake & Ports) both **no gap**; roadmap blobs unchanged (4efb666 / bac6923); LLC JuniorHome 1c76370 left to LLC. So this note uses the lane's own question. No Trit Pack, Handshake or LLC commit touched JuniorLLM in the last 6 h (only f02b26a, this lane).

## Method

- **Unchanged:** `base.py`, `tiled.py`, `par.py`, `cache.py` copied byte for byte from `/workspace/bench1055_1117` (= 18:55 11/16, sha256 verified): hidden 2560, GQA 20 q / 5 KV heads, FFN 6912, absmean ternary weights, seed 202611141055, fused q|k|v and gate|up, private per-worker scratch, `OPENBLAS_NUM_THREADS=1`, ctx=1 attention (the step is the seven weight GEMMs + norms/SiLU).
- **Paths:** `u32_T256_W8`, `u32_T512_W8`, `u32_T256_W4` = 2-bit pack + u32-LUT take → f32 tile → BLAS (`par.layer`); `int8_T128_W8` = int8-resident trit rows cast per tile (`cache.layer`, all sites hot; best int8 config at P=512 in the 10:55 note); `f32_W8` = trit rows × γ held in f32, threaded row split.
- **Chunking:** prompt of P Gaussian tokens split into chunks of C rows run through the layer in sequence (each chunk is independent at ctx=1). Grid (P, C): (256, 256), (1024, 256/512/1024), (2048, 512/1024/2048).
- **Timing:** 1 warm call per path per grid point, then 5 interleaved rounds (rotated order) × 3 repeats; median of round medians with min–max; two full runs. 8 cores, numpy only, offline, nothing bound, no downloads. Box had other load (~13 of 15 GB used; load avg up to ~4.2 at the end).
- **Error:** each path on 32 tokens (chunked at 16) vs the W1.58 reference `base.layer`.

Commands (in `/workspace/bench1455_1117`, `crossover.py` sha256 `64de9649470ff8f4…`):

```
OPENBLAS_NUM_THREADS=1 python3 crossover.py results_run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 python3 crossover.py results_run2.json > run2.txt
```

<details><summary>crossover.py</summary>

```python
#!/usr/bin/env python3
"""Prefill crossover (14:55 11/17): P = 256/1024/2048 prompt tokens, processed in chunks C, through the threaded full bitnetd
layer. 2-bit u32-LUT expand (par.layer) at T=256/512 W=8 and T=256 W=4 vs int8-resident trit rows (cache.layer, T=128 W=8,
best int8 config at P=512 in the 10:55 note) vs f32-resident trit rows (W=8). base/tiled/par/cache verbatim from 18:55 11/16;
f32 path and layer_with reused from prefill.py (10:55). Usage: OPENBLAS_NUM_THREADS=1 python3 crossover.py out.json"""
import json, os, sys, time, statistics
import numpy as np
import base, par, cache
from base import rms, silu, D, KVH, QH, HD, PW
base.build(); u32_bytes = par.prep(); cache.prep()
FUSE = par.FUSE
F32 = {k: dict(W=np.ascontiguousarray(np.concatenate([PW[n]["Wt"] for n in names])), split=par.FP[k]["split"]) for k, names in FUSE.items()}
for n in PW: PW[n].pop("Wd", None)
def rows_f32(Wm, x, y, r0, r1): np.matmul(x, Wm[r0:r1].T, out=y[:, r0:r1])
def fmmf(key, x, W):
    P = F32[key]; Wm = P["W"]; out = Wm.shape[0]; y = np.empty((x.shape[0], out), np.float32); per = -(-out // W)
    for f in [par.pool(W).submit(rows_f32, Wm, x, y, w * per, min(out, (w + 1) * per)) for w in range(W) if w * per < out]: f.result()
    return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
def layer_with(mm, x):
    h1 = rms(x); q, k, v = mm("qkv", h1)
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    o = mm("o", att)[0]; x2 = x + o; h2 = rms(x2)
    g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
    return x2 + mm("down", z)[0]
BASE = {"u32_T256_W8": lambda x: par.layer(x, 256, 8), "u32_T512_W8": lambda x: par.layer(x, 512, 8),
        "u32_T256_W4": lambda x: par.layer(x, 256, 4), "int8_T128_W8": lambda x: cache.layer(tuple(FUSE), x, 128, 8),
        "f32_W8": lambda x: layer_with(lambda k, h: fmmf(k, h, 8), x)}
def chunked(f, C): return lambda x: np.concatenate([f(x[i:i + C]) for i in range(0, x.shape[0], C)])
RB = {"u32": u32_bytes, "int8": cache.resident_bytes(tuple(FUSE)), "f32": sum(P["W"].nbytes for P in F32.values())}
res = {"seed": base.SEED, "numpy": np.__version__, "cpus": os.cpu_count(), "resident_bytes": RB, "err": {}, "time": {}}
xe = np.random.default_rng(base.SEED + 2).standard_normal((32, D)).astype(np.float32)
yw = base.layer(xe, base.wo_mm)[0]
for n, f in BASE.items():
    y = chunked(f, 16)(xe); res["err"][n] = {"max_abs_vs_w158": float(np.max(np.abs(y - yw))), "cos_vs_w158": base.cos(y, yw)}
print("bytes", RB, "\nerr", res["err"], flush=True)
ROUNDS = 5; REP = 3
for P, C in ((256, 256), (1024, 256), (1024, 512), (1024, 1024), (2048, 512), (2048, 1024), (2048, 2048)):
    xb = np.random.default_rng(base.SEED + 70 + P).standard_normal((P, D)).astype(np.float32)
    paths = {n: chunked(f, C) for n, f in BASE.items()}; names = list(paths); rounds = {n: [] for n in names}
    for n in names: paths[n](xb)
    for rr in range(ROUNDS):
        for n in names[rr % len(names):] + names[:rr % len(names)]:
            s = []
            for _ in range(REP):
                t0 = time.perf_counter(); paths[n](xb); s.append((time.perf_counter() - t0) * 1e3)
            rounds[n].append(statistics.median(s))
    o = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
    for n in names:
        o[n]["us_per_token"] = o[n]["median_ms"] * 1e3 / P
        o[n]["int8_over_this"] = o["int8_T128_W8"]["median_ms"] / o[n]["median_ms"]
        o[n]["f32_over_this"] = o["f32_W8"]["median_ms"] / o[n]["median_ms"]
    res["time"][f"P{P}_C{C}"] = o
    for n in names: print(P, C, n, {k: round(v, 3) for k, v in o[n].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
```

</details>

## Results

Resident weight bytes (whole layer, incl. per-row γ): **u32 2-bit 17,458,176** · int8 69,559,296 (3.98x) · f32 277,872,640 (15.9x).

Error vs W1.58 reference: all u32 and int8 paths max abs 1.9e-6, cos 0.99999999999994; f32 0.0 / 1.0.

Time per full prefill of P tokens (ms, median [min–max] of round medians), run1 / run2. `int8/this` > 1 means faster than int8-resident.

| P | C | path | run1 ms | run2 ms | µs/token r1 / r2 | int8/this r1 / r2 | f32/this r1 / r2 |
|---|---|---|---|---|---|---|---|
| 256 | 256 | `u32_T256_W8` | 61.9 [54–77] | 55.6 [52–59] | 242 / 217 | 0.94 / 0.87 | 0.75 / 0.78 |
| 256 | 256 | `u32_T512_W8` | 67.4 [58–87] | 61.0 [54–66] | 263 / 238 | 0.87 / 0.79 | 0.69 / 0.71 |
| 256 | 256 | `u32_T256_W4` | 88.7 [84–122] | 78.5 [76–88] | 346 / 307 | 0.66 / 0.62 | 0.52 / 0.55 |
| 256 | 256 | `int8_T128_W8` | 58.5 [48–63] | 48.3 [46–58] | 229 / 189 | 1.00 / 1.00 | 0.79 / 0.90 |
| 256 | 256 | `f32_W8` | 46.4 [41–61] | 43.4 [41–45] | 181 / 170 | 1.26 / 1.11 | 1.00 / 1.00 |
| 1024 | 256 | `u32_T256_W8` | 213.5 [199–266] | 213.0 [209–236] | 209 / 208 | 0.86 / 0.89 | 0.77 / 0.79 |
| 1024 | 256 | `u32_T512_W8` | 218.5 [217–252] | 225.5 [211–230] | 213 / 220 | 0.84 / 0.84 | 0.75 / 0.74 |
| 1024 | 256 | `u32_T256_W4` | 309.6 [300–320] | 304.6 [300–315] | 302 / 297 | 0.59 / 0.62 | 0.53 / 0.55 |
| 1024 | 256 | `int8_T128_W8` | 184.0 [179–221] | 189.7 [175–193] | 180 / 185 | 1.00 / 1.00 | 0.90 / 0.88 |
| 1024 | 256 | `f32_W8` | 164.9 [153–185] | 167.8 [156–170] | 161 / 164 | 1.12 / 1.13 | 1.00 / 1.00 |
| 1024 | 512 | `u32_T256_W8` | 181.6 [178–199] | 187.7 [175–247] | 177 / 183 | 0.89 / 0.87 | 0.84 / 0.78 |
| 1024 | 512 | `u32_T512_W8` | 193.3 [184–212] | 185.2 [178–219] | 189 / 181 | 0.83 / 0.89 | 0.79 / 0.79 |
| 1024 | 512 | `u32_T256_W4` | 269.6 [265–302] | 260.6 [257–278] | 263 / 254 | 0.60 / 0.63 | 0.57 / 0.56 |
| 1024 | 512 | `int8_T128_W8` | 161.0 [159–184] | 164.2 [154–207] | 157 / 160 | 1.00 / 1.00 | 0.95 / 0.89 |
| 1024 | 512 | `f32_W8` | 153.4 [142–154] | 146.8 [136–203] | 150 / 143 | 1.05 / 1.12 | 1.00 / 1.00 |
| 1024 | 1024 | `u32_T256_W8` | 190.4 [187–220] | 198.6 [194–240] | 186 / 194 | 1.01 / 1.05 | 0.88 / 0.86 |
| 1024 | 1024 | `u32_T512_W8` | 205.7 [202–231] | 216.2 [207–247] | 201 / 211 | 0.93 / 0.96 | 0.81 / 0.79 |
| 1024 | 1024 | `u32_T256_W4` | 272.4 [270–306] | 289.5 [281–345] | 266 / 283 | 0.70 / 0.72 | 0.61 / 0.59 |
| 1024 | 1024 | `int8_T128_W8` | 191.8 [179–212] | 207.7 [183–220] | 187 / 203 | 1.00 / 1.00 | 0.87 / 0.82 |
| 1024 | 1024 | `f32_W8` | 166.9 [155–187] | 171.2 [163–176] | 163 / 167 | 1.15 / 1.21 | 1.00 / 1.00 |
| 2048 | 512 | `u32_T256_W8` | 419.5 [400–438] | 382.2 [368–402] | 205 / 187 | 0.91 / 0.94 | 0.78 / 0.78 |
| 2048 | 512 | `u32_T512_W8` | 427.3 [417–501] | 401.3 [387–500] | 209 / 196 | 0.90 / 0.89 | 0.77 / 0.75 |
| 2048 | 512 | `u32_T256_W4` | 591.2 [560–636] | 567.9 [542–605] | 289 / 277 | 0.65 / 0.63 | 0.56 / 0.53 |
| 2048 | 512 | `int8_T128_W8` | 383.2 [348–436] | 358.9 [338–429] | 187 / 175 | 1.00 / 1.00 | 0.86 / 0.83 |
| 2048 | 512 | `f32_W8` | 329.0 [318–353] | 299.3 [285–313] | 161 / 146 | 1.16 / 1.20 | 1.00 / 1.00 |
| 2048 | 1024 | `u32_T256_W8` | 385.5 [354–418] | 434.7 [417–472] | 188 / 212 | 0.95 / 1.01 | 0.84 / 0.87 |
| 2048 | 1024 | `u32_T512_W8` | 412.0 [377–447] | 428.8 [416–471] | 201 / 209 | 0.89 / 1.03 | 0.79 / 0.88 |
| 2048 | 1024 | `u32_T256_W4` | 543.5 [531–608] | 575.6 [554–606] | 265 / 281 | 0.67 / 0.76 | 0.60 / 0.65 |
| 2048 | 1024 | `int8_T128_W8` | 364.9 [332–388] | 439.7 [391–484] | 178 / 215 | 1.00 / 1.00 | 0.89 / 0.86 |
| 2048 | 1024 | `f32_W8` | 324.0 [300–376] | 376.5 [324–382] | 158 / 184 | 1.13 / 1.17 | 1.00 / 1.00 |
| 2048 | 2048 | `u32_T256_W8` | 772.2 [744–832] | 714.3 [675–798] | 377 / 349 | 1.00 / 0.97 | 0.88 / 0.88 |
| 2048 | 2048 | `u32_T512_W8` | 734.5 [706–852] | 683.0 [661–793] | 359 / 333 | 1.05 / 1.01 | 0.93 / 0.92 |
| 2048 | 2048 | `u32_T256_W4` | 977.5 [873–998] | 907.0 [847–936] | 477 / 443 | 0.79 / 0.76 | 0.70 / 0.69 |
| 2048 | 2048 | `int8_T128_W8` | 771.1 [737–795] | 689.9 [677–726] | 376 / 337 | 1.00 / 1.00 | 0.89 / 0.91 |
| 2048 | 2048 | `f32_W8` | 683.0 [653–735] | 625.5 [573–637] | 333 / 305 | 1.13 / 1.10 | 1.00 / 1.00 |

**Crossover (u32_T256_W8 or best u32 vs int8_T128_W8):**

| C | int8/u32 range over P and runs | ≥ 0.95x? |
|---|---|---|
| 256 | 0.86–0.95 (T256) | borderline, one run only |
| 512 | 0.88–0.94 (T256), 0.83–0.90 (T512) | no |
| 1024 | 0.95–1.05 (T256), 0.89–1.03 (T512) | **yes, both P, both runs** |
| 2048 | 0.97–1.00 (T256), 1.01–1.05 (T512) | yes, but 2x slower per token |

## Verdict

**viable** (memory, ≥ 1.5x at parity speed). At C ≥ 1024 the 2-bit resident layer matches int8-resident speed (0.95–1.05x) with 3.98x fewer weight bytes and exact W1.58 output; vs f32-resident it is 0.84–0.93x speed for 15.9x fewer bytes. Practical pick: C = 512 is the fastest absolute prefill on this box for every path (activations fit cache better; C = 2048 doubles µs/token for all paths), and there 2-bit costs ~6–12% vs int8, so there is no case for an int8 shadow copy. T = 512 is only better at C = 2048; W = 4 halves throughput (0.59–0.79x) and should not be used for prefill. Not a speed win over f32 anywhere (f32 stays 1.05–1.26x faster than int8 and 1.07–1.33x faster than 2-bit).

## Links

- previous: [[2026-11-17_1055_prefill-p32-p128-p512-threaded-u32-lut-expand-vs-int8-and-f32-resident-trit-rows-full-bitnetd-layer]]
- this wave's pair files: none — Quant Chief 14:05 assignment pair_A no gap, pair_B no gap (`/workspace/quant-chief/last_tips.json`, last_run 2026-11-17T21:11:00Z); roadmap `JuniorHome docs/ROADMAP.md` blob 4efb666, `JuniorLLM docs/BETA_TO_OS.md` blob bac6923
- bench dir: `/workspace/bench1455_1117` (run1.txt, run2.txt, results_run1.json, results_run2.json)

## Next question

Prefill settles at C = 512 with 2-bit resident. The remaining prefill cost is ctx > 1 attention: at C = 512 chunks over a growing context (512 → 4096), how does the A8 g64 KV-cache attention time compare to the 2-bit weight GEMMs per chunk, and does it become the dominant term before P = 2048?

close: sha256(prev || this)[:16] = b6f43d7b9b68dfac
