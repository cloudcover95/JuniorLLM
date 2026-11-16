---
local: true
date: 2026-11-16
wave: "2026-11-16 02:55 (ran ~02:48–03:05 MT; Chief last_run 2026-11-16T09:10:00Z ~02:10 MT)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (batched decode)"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, batched, b2, b4, b8, b16, tiled-expand, u32-lut, crossover, int8-expand, popcount, tmac-g8, numpy, viable]
---

# A batched packed kernel works. Expanding the resident 2-bit pack to f32 one 16-row tile at a time (a u32 LUT gives 4 int8 trits per byte) and reusing each tile across the batch through BLAS runs the full layer at B=8 in **37.4–39.4 ms**. The 18:55 packed popcount default takes 106.0–106.6 ms, so this is **2.70–2.83x faster**, and it is exact W1.58 (max abs 2.0e-06). It keeps the same 17.55 MB resident plus 0.88 MB scratch, against 140.25 MB for int8-expand (**7.6x less**). It is still 0.55–0.87x of int8-expand's speed (0.73x interleaved), so it does not beat dense int8 on time.

## Question

The prior note [[2026-11-15_1855_drop-static-lists-pure-dynamic-split-quantile-step-b1-b8-batch-invariance-full-bitnetd-layer-vs-tmac-g8]] found that at B=8 the packed popcount path is 5.5x slower than int8-expand + BLAS, so B≥8 went to int8. Its next question: can a batched packed kernel close that gap without ever holding a full int8 matrix? Measure B = 1/2/4/8/16 against int8-expand + BLAS at the same 17.55 MB resident, and find the crossover B.

Quant Chief `last_tips.json` `last_run` 2026-11-16T09:10:00Z (~02:10 MT) set pair_A and pair_B to **no gap** ("02:05 11/16: blobs unchanged; both no gap"), so this note uses the lane's own next question.

## Method

- **Unchanged:** the layer, weights and draws are those of 14:55 11/14 through 18:55 11/15. `base.py` is copied byte for byte (sha256[:16] `a38c8245cd13d800`). `prev18.py`, `prev0255.py`, `prev0555.py`, `prev1055.py` and `prev1455.py` are copied as before, and the 18:55 `bench.py` is copied as `prev1855.py` (`0e81c1dbcaaa6017`).
  - The layer is one full bitnetd decoder layer: hidden 2560, GQA 20 q / 5 KV heads, FFN 6912, absmean ternary weights.
- **New kernels (`tiled.py`)** expand the weights to an f32 tile of T output rows × in, run `x @ tileᵀ` through BLAS for all B tokens, then move to the next tile. A full int8 or f32 weight matrix is never built. Three expand variants:
  - `u32`: a row-major 2-bit pack (out × in/4, the same bytes as `tmac_g8`'s `PT`, transposed), a 256-entry uint32 LUT holding 4 int8 trits per entry, and an int8 → f32 cast into the scratch tile.
  - `lut4`: a 256×4 f32 LUT gather.
  - `bits`: `np.unpackbits` of the mask/sign planes, then m − 2(m&s).
  - A pre-run sweep of T ∈ {8, 16, 32, 64, 128, 256, 512, 1024} picked T = 16–32: the tile stays in L2, about 0.44–0.88 MB of f32 at in = 6912. Larger T was 1.3–1.8x slower.
- **Baselines:**
  - `tmac_g8`: a per-token loop, because it is a B=1 kernel.
  - int8-expand + BLAS: int8 resident, with a full f32 scratch of the largest matrix.
  - `f32_resident`.
  - The 18:55 packed A4/A5 popcount default (`D8_nostatic`, calibrated on t4 as before) at B=1 and B=8.
- **Error:** layer output vs the weight-only W1.58 reference (`wo_mm`) on 32 t3hot48 tokens. The tiled kernels and `tmac_g8` compute the same W1.58 layer. int8 and f32 use absmax-int8 of the original dense W, which is a different model, so their error vs W1.58 is not comparable.
- **Timing and setup:** 3 runs at B = 1/2/4/8/16 (9 repeats × 5 passes, 3 warm each), plus a 7-round interleaved check at B=1 and B=8. 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench0255_1116`, next to the copied files):

```
python3 bench.py results_run1.json          # error check + B=1/2/4/8/16 timing (+ D8 popcount at B=1/8)
python3 bench.py results_run2.json noeval
python3 bench.py results_run3.json noeval
python3 tinter.py                           # interleaved B=1 / B=8
```

<details><summary>tiled.py</summary>

```python
"""Batched packed kernels: expand resident 2-bit weights to f32 one output tile at a time, reuse across the B tokens with BLAS."""
import numpy as np
import base
LUT4 = base.LUTF_T.T.copy()            # (256, 4) f32: byte -> 4 trits (codes 0,1,2=-1)
SCRT = {}
def prep():
    for P in base.PW.values():
        P["PR"] = np.ascontiguousarray(P["PT"].T)      # (out, in/4) row-major 2-bit pack, same bytes as PT
def lut_tiled(T):
    def mm(P, x):
        PR = P["PR"]; out, q = PR.shape; i = q * 4
        s = SCRT.setdefault(("l", T, i), np.empty((T, q, 4), np.float32))
        y = np.empty((x.shape[0], out), np.float32)
        for o0 in range(0, out, T):
            n = min(T, out - o0)
            np.take(LUT4, PR[o0:o0+n], axis=0, out=s[:n])
            np.matmul(x, s[:n].reshape(n, i).T, out=y[:, o0:o0+n])
        y *= np.float32(P["gw"]); return y
    return mm
def bits_tiled(T):
    def mm(P, x):
        wm8, ws8 = P["wm8"], P["ws8"]; out = wm8.shape[0]; i = wm8.shape[1] * 8
        s = SCRT.setdefault(("b", T, i), np.empty((T, i), np.float32))
        y = np.empty((x.shape[0], out), np.float32)
        for o0 in range(0, out, T):
            n = min(T, out - o0)
            m = np.unpackbits(wm8[o0:o0+n], axis=1); sg = np.unpackbits(ws8[o0:o0+n], axis=1)
            m = m.view(np.int8); m -= (m & sg) << 1          # +1 / -1 / 0
            np.copyto(s[:n], m, casting="unsafe")
            np.matmul(x, s[:n].T, out=y[:, o0:o0+n])
        y *= np.float32(P["gw"]); return y
    return mm
LUT_U32 = np.ascontiguousarray(np.where(base.LUT == 2, -1, base.LUT).astype(np.int8)).view(np.uint32).ravel()  # byte -> 4 int8 trits as one u32
def u32_tiled(T):
    def mm(P, x):
        PR = P["PR"]; out, q = PR.shape; i = q * 4
        s8 = SCRT.setdefault(("u8", T, i), np.empty((T, q), np.uint32))
        s = SCRT.setdefault(("uf", T, i), np.empty((T, i), np.float32))
        y = np.empty((x.shape[0], out), np.float32)
        for o0 in range(0, out, T):
            n = min(T, out - o0)
            np.take(LUT_U32, PR[o0:o0+n], out=s8[:n])
            np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe")
            np.matmul(x, s[:n].T, out=y[:, o0:o0+n])
        y *= np.float32(P["gw"]); return y
    return mm
```

</details>

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Can a batched packed kernel close the B>=8 gap? (02:55 11/16). Same layer/weights/kernels as 14:55..18:55 11/15 (base.py
verbatim). New: tiled expand of the resident 2-bit pack to f32 one output tile at a time (T rows), reused across B tokens via
BLAS; never holds a full int8/f32 matrix. Variants: u32 LUT (byte -> 4 int8 trits as one u32, then cast), f32 LUT4 gather,
mask/sign unpackbits. Weight-only W1.58 (exact vs wo_mm). Baselines: tmac_g8 per-token loop, int8_expand (int8 resident +
full f32 scratch + BLAS), f32_resident, and the 18:55 packed A4/A5 popcount default (D8_nostatic) at B=1/8.
numpy-only, offline. Usage: python3 bench.py results_runN.json [noeval]"""
import json, sys
import numpy as np
import base, tiled, prev1855 as p18
from base import layer, timed, cos, wo_mm, tmac_g8, i8scr, f32res, D, F, SEED
BS = (1, 2, 4, 8, 16)
def per_tok(mm): return lambda P, x: np.concatenate([mm(P, x[i:i+1]) for i in range(x.shape[0])])
def main():
    _, by = base.build(); tiled.prep()
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "err": {}, "time": {}, "ratios": {}}
    K = {"u32_T16": tiled.u32_tiled(16), "u32_T32": tiled.u32_tiled(32), "lut4_T32": tiled.lut_tiled(32), "bits_T32": tiled.bits_tiled(32)}
    res["scratch_bytes"] = {"int8_expand": base.SCR.nbytes, "u32_T16": 16 * F * 8, "u32_T32": 32 * F * 8, "lut4_T32": 32 * F * 4, "bits_T32": 32 * F * 4 + 2 * 32 * F}
    D8 = p18.variants(p18.calibrate())["D8_nostatic"]
    tok = p18.tokf(*p18.DRAWS["t3hot48"])
    xe = tok(np.random.default_rng(SEED + 2), 32); yw = layer(xe, wo_mm)[0]
    if "noeval" not in sys.argv:
        for n, mm in list(K.items()) + [("tmac_g8", per_tok(tmac_g8))]:
            y = layer(xe, mm)[0]; d1, d0 = y - xe, yw - xe
            res["err"][n] = {"max_abs": float(np.max(np.abs(y - yw))), "delta_cos": cos(d1, d0),
                             "rel_l2": float(np.linalg.norm((d1 - d0).astype(np.float64)) / np.linalg.norm(d0.astype(np.float64)))}
            print("err", n, res["err"][n], flush=True)
    for B in BS:
        xb = p18.tokf(4, 0)(np.random.default_rng(SEED + 9), B)
        fns = {"tmac_g8": lambda: layer(xb, per_tok(tmac_g8)), "int8_expand": lambda: layer(xb, i8scr), "f32_resident": lambda: layer(xb, f32res)}
        for n, mm in K.items(): fns[n] = (lambda mm=mm: layer(xb, mm))
        if B in (1, 8): fns["packed_D8_popcount"] = lambda: p18.layer(xb, None, D8)
        T = {}
        for n, fn in fns.items(): T[n] = timed(fn); print(B, n, T[n], flush=True)
        i8 = T["int8_expand"]["median_ms"]; tm = T["tmac_g8"]["median_ms"]
        res["time"][f"B{B}"] = T
        res["ratios"][f"B{B}"] = {n: {"vs_int8_expand": i8 / T[n]["median_ms"], "vs_tmac": tm / T[n]["median_ms"]} for n in T}
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
if __name__ == "__main__": main()
```

</details>

<details><summary>tinter.py</summary>

```python
"""Interleaved timing at B=1 and B=8: 7 rounds, each times every config (REPEATS=5 x PASSES=5) in rotated order; median of round medians."""
import json, statistics
import numpy as np
import base, tiled, prev1855 as p18
base.build(); tiled.prep(); base.REPEATS = 5; out = {}
for B in (1, 8):
    xb = p18.tokf(4, 0)(np.random.default_rng(base.SEED + 9), B)
    fns = {"int8_expand": lambda: base.layer(xb, base.i8scr), "u32_T16": lambda: base.layer(xb, tiled.u32_tiled(16)),
           "u32_T32": lambda: base.layer(xb, tiled.u32_tiled(32)), "f32_resident": lambda: base.layer(xb, base.f32res)}
    names = list(fns); rounds = {n: [] for n in names}
    for r in range(7):
        for n in names[r % len(names):] + names[:r % len(names)]: rounds[n].append(base.timed(fns[n])["median_ms"])
    o = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
    for n in names: o[n]["vs_int8_expand"] = o["int8_expand"]["median_ms"] / o[n]["median_ms"]
    out[f"B{B}"] = o
print(json.dumps(out, indent=1)); json.dump(out, open("tinter.json", "w"), indent=1)
```

</details>

## Results

Error vs the weight-only W1.58 layer (32 t3hot48 tokens):

| kernel | max abs vs W1.58 layer | delta cos | rel-L2 |
|---|---|---|---|
| u32_T16 | 1.97e-06 | 1.000000000000 | 8.1e-07 |
| u32_T32 | 1.97e-06 | 1.000000000000 | 8.1e-07 |
| lut4_T32 | 1.97e-06 | 1.000000000000 | 8.1e-07 |
| bits_T32 | 1.97e-06 | 1.000000000000 | 8.1e-07 |
| tmac_g8 | 1.67e-06 | 1.000000000000 | 6.7e-07 |

Full layer at **B=1**:

| config | ms median (min–max), runs 1 / 2 / 3 | vs int8_expand | vs `tmac_g8` |
|---|---|---|---|
| `tmac_g8` (per-token loop) | 20.84 (20.77–21.03) / 20.94 (20.75–21.50) / 20.86 (20.71–21.27) | 0.78x / 0.94x / 0.88x | 1.00x / 1.00x / 1.00x |
| 18:55 packed A4/A5 popcount default (D8, no static) | 12.62 (12.59–14.24) / 12.55 (12.47–12.66) / 12.44 (12.37–12.66) | 1.28x / 1.57x / 1.47x | 1.65x / 1.67x / 1.68x |
| **tiled u32-LUT expand, T=16** | 28.44 (28.33–29.12) / 28.41 (28.39–28.79) / 28.24 (28.03–28.81) | 0.57x / 0.69x / 0.65x | 0.73x / 0.74x / 0.74x |
| tiled u32-LUT expand, T=32 | 27.14 (27.06–27.87) / 26.58 (26.35–27.59) / 26.50 (26.36–37.87) | 0.60x / 0.74x / 0.69x | 0.77x / 0.79x / 0.79x |
| tiled f32 LUT4 gather, T=32 | 35.73 (34.07–35.96) / 34.38 (34.07–35.20) / 33.48 (32.45–35.12) | 0.45x / 0.57x / 0.55x | 0.58x / 0.61x / 0.62x |
| tiled mask/sign unpackbits, T=32 | 41.77 (41.70–42.76) / 41.78 (40.68–44.00) / 40.54 (40.33–41.25) | 0.39x / 0.47x / 0.45x | 0.50x / 0.50x / 0.51x |
| int8 expand + BLAS (int8 resident + full f32 scratch) | 16.19 (15.28–19.98) / 19.66 (18.53–21.60) / 18.26 (17.05–21.57) | 1.00x / 1.00x / 1.00x | 1.29x / 1.07x / 1.14x |
| `f32_resident` | 1.37 (1.34–2.06) / 2.60 (1.50–2.71) / 1.44 (1.40–1.63) | 11.80x / 7.55x / 12.69x | 15.19x / 8.05x / 14.49x |

**B=2**:

| config | ms median (min–max), runs 1 / 2 / 3 | vs int8_expand | vs `tmac_g8` |
|---|---|---|---|
| `tmac_g8` (per-token loop) | 41.88 (41.69–45.29) / 42.64 (42.22–49.30) / 42.79 (41.93–44.80) | 0.40x / 0.77x / 0.46x | 1.00x / 1.00x / 1.00x |
| **tiled u32-LUT expand, T=16** | 29.67 (29.63–30.14) / 30.47 (30.19–31.79) / 31.52 (31.27–33.73) | 0.56x / 1.08x / 0.63x | 1.41x / 1.40x / 1.36x |
| tiled u32-LUT expand, T=32 | 29.01 (28.80–29.75) / 28.78 (28.63–29.57) / 29.16 (28.71–29.61) | 0.58x / 1.14x / 0.68x | 1.44x / 1.48x / 1.47x |
| tiled f32 LUT4 gather, T=32 | 35.92 (35.41–36.36) / 37.46 (36.99–39.44) / 37.57 (36.61–38.60) | 0.47x / 0.87x / 0.53x | 1.17x / 1.14x / 1.14x |
| tiled mask/sign unpackbits, T=32 | 43.71 (43.61–44.02) / 43.90 (43.26–44.53) / 44.35 (43.71–45.21) | 0.38x / 0.75x / 0.45x | 0.96x / 0.97x / 0.96x |
| int8 expand + BLAS (int8 resident + full f32 scratch) | 16.74 (15.89–18.19) / 32.78 (29.62–40.38) / 19.81 (18.99–20.68) | 1.00x / 1.00x / 1.00x | 2.50x / 1.30x / 2.16x |
| `f32_resident` | 3.77 (3.54–5.02) / 8.83 (7.72–18.75) / 4.08 (3.96–5.79) | 4.45x / 3.71x / 4.86x | 11.12x / 4.83x / 10.49x |

**B=4**:

| config | ms median (min–max), runs 1 / 2 / 3 | vs int8_expand | vs `tmac_g8` |
|---|---|---|---|
| `tmac_g8` (per-token loop) | 82.81 (82.49–83.82) / 86.01 (85.04–88.06) / 92.96 (90.25–94.71) | 0.22x / 0.23x / 0.30x | 1.00x / 1.00x / 1.00x |
| **tiled u32-LUT expand, T=16** | 32.08 (31.78–32.80) / 32.72 (32.03–33.59) / 31.58 (31.23–32.07) | 0.56x / 0.60x / 0.88x | 2.58x / 2.63x / 2.94x |
| tiled u32-LUT expand, T=32 | 30.60 (30.34–30.77) / 31.15 (30.30–33.94) / 30.64 (30.02–31.77) | 0.58x / 0.63x / 0.91x | 2.71x / 2.76x / 3.03x |
| tiled f32 LUT4 gather, T=32 | 37.06 (36.18–37.58) / 38.90 (38.41–40.56) / 42.81 (41.04–44.35) | 0.48x / 0.50x / 0.65x | 2.23x / 2.21x / 2.17x |
| tiled mask/sign unpackbits, T=32 | 45.57 (45.51–47.43) / 44.73 (44.23–46.22) / 47.28 (46.11–48.25) | 0.39x / 0.44x / 0.59x | 1.82x / 1.92x / 1.97x |
| int8 expand + BLAS (int8 resident + full f32 scratch) | 17.84 (16.83–20.37) / 19.51 (18.69–20.01) / 27.78 (26.65–31.06) | 1.00x / 1.00x / 1.00x | 4.64x / 4.41x / 3.35x |
| `f32_resident` | 3.85 (3.72–5.72) / 4.03 (3.83–6.26) / 4.92 (4.53–5.85) | 4.64x / 4.84x / 5.65x | 21.52x / 21.34x / 18.90x |

**B=8**:

| config | ms median (min–max), runs 1 / 2 / 3 | vs int8_expand | vs `tmac_g8` |
|---|---|---|---|
| `tmac_g8` (per-token loop) | 169.17 (166.88–177.05) / 169.41 (167.32–169.95) / 185.82 (176.92–195.39) | 0.15x / 0.12x / 0.18x | 1.00x / 1.00x / 1.00x |
| 18:55 packed A4/A5 popcount default (D8, no static) | 106.63 (104.89–116.97) / 106.00 (105.47–106.27) / 106.58 (105.58–109.02) | 0.23x / 0.19x / 0.32x | 1.59x / 1.60x / 1.74x |
| **tiled u32-LUT expand, T=16** | 37.67 (37.30–38.39) / 37.40 (37.11–37.67) / 39.44 (37.68–42.43) | 0.65x / 0.55x / 0.87x | 4.49x / 4.53x / 4.71x |
| tiled u32-LUT expand, T=32 | 40.24 (39.81–40.52) / 39.85 (39.64–40.72) / 42.11 (41.25–43.78) | 0.61x / 0.51x / 0.81x | 4.20x / 4.25x / 4.41x |
| tiled f32 LUT4 gather, T=32 | 46.49 (45.42–47.27) / 49.16 (48.95–50.83) / 52.43 (49.92–53.58) | 0.53x / 0.42x / 0.65x | 3.64x / 3.45x / 3.54x |
| tiled mask/sign unpackbits, T=32 | 55.75 (55.54–56.61) / 53.49 (53.08–54.74) / 56.24 (54.78–58.22) | 0.44x / 0.38x / 0.61x | 3.03x / 3.17x / 3.30x |
| int8 expand + BLAS (int8 resident + full f32 scratch) | 24.65 (19.87–28.80) / 20.45 (18.96–28.12) / 34.15 (32.55–36.48) | 1.00x / 1.00x / 1.00x | 6.86x / 8.28x / 5.44x |
| `f32_resident` | 4.92 (4.57–9.42) / 4.35 (4.14–5.38) / 7.86 (6.49–8.61) | 5.01x / 4.70x / 4.34x | 34.36x / 38.90x / 23.64x |

**B=16**:

| config | ms median (min–max), runs 1 / 2 / 3 | vs int8_expand | vs `tmac_g8` |
|---|---|---|---|
| `tmac_g8` (per-token loop) | 339.16 (336.16–350.95) / 340.64 (338.12–356.04) / 348.70 (345.68–352.51) | 0.07x / 0.07x / 0.07x | 1.00x / 1.00x / 1.00x |
| **tiled u32-LUT expand, T=16** | 54.26 (53.81–61.56) / 52.66 (52.38–52.87) / 53.64 (53.23–54.75) | 0.41x / 0.45x / 0.44x | 6.25x / 6.47x / 6.50x |
| tiled u32-LUT expand, T=32 | 69.45 (68.88–71.11) / 70.83 (70.45–71.17) / 74.02 (71.38–75.85) | 0.32x / 0.34x / 0.32x | 4.88x / 4.81x / 4.71x |
| tiled f32 LUT4 gather, T=32 | 85.24 (84.19–85.62) / 86.23 (85.99–87.82) / 87.84 (86.73–89.49) | 0.26x / 0.28x / 0.27x | 3.98x / 3.95x / 3.97x |
| tiled mask/sign unpackbits, T=32 | 87.05 (86.31–88.66) / 86.02 (85.01–88.24) / 91.54 (88.57–94.05) | 0.26x / 0.28x / 0.26x | 3.90x / 3.96x / 3.81x |
| int8 expand + BLAS (int8 resident + full f32 scratch) | 22.34 (20.77–23.59) / 23.87 (22.08–28.33) / 23.57 (22.89–25.97) | 1.00x / 1.00x / 1.00x | 15.18x / 14.27x / 14.79x |
| `f32_resident` | 5.58 (5.37–6.55) / 6.77 (5.93–9.90) / 6.27 (5.71–8.49) | 4.00x / 3.53x / 3.76x | 60.76x / 50.33x / 55.65x |

Interleaved check (7 rounds, rotated order):

| B | config | ms median of 7 round medians (min–max) | vs int8_expand |
|---|---|---|---|
| 1 | int8_expand | 24.52 (22.65–28.22) | 1.00x |
| 1 | u32_T16 | 31.36 (29.97–33.01) | 0.78x |
| 1 | u32_T32 | 30.10 (28.65–30.21) | 0.81x |
| 1 | f32_resident | 1.84 (1.67–2.14) | 13.34x |
| 8 | int8_expand | 27.40 (23.65–29.97) | 1.00x |
| 8 | u32_T16 | 37.47 (36.57–41.98) | 0.73x |
| 8 | u32_T32 | 42.77 (41.10–44.86) | 0.64x |
| 8 | f32_resident | 5.30 (4.77–8.33) | 5.17x |

Bytes:

| path | resident weights | scratch | total |
|---|---|---|---|
| tiled u32 T=16 (2-bit pack + row sums) | 17.55 MB | 0.88 MB (16×6912 u32 + f32) | **18.43 MB** |
| tiled u32 T=32 | 17.55 MB | 1.77 MB | 19.32 MB |
| packed popcount (18:55) | 17.55 MB | ~0 | 17.55 MB |
| int8 expand + BLAS | 69.47 MB | 70.78 MB (full f32 6912×2560) | 140.25 MB |
| `f32_resident` | 277.87 MB | 0 | 277.87 MB |

## Verdict

**Viable** as the batched (B ≥ 3) packed path. It is a large speed gain over the packed path that existed before, at exact W1.58 accuracy and the same 2-bit resident bytes. It is **not** a speed win over dense int8.

1. **B=8 closes most of the gap.** Tiled u32 T=16 runs at 37.40–39.44 ms, against 106.00–106.63 ms for the packed popcount default (**2.70–2.83x**) and 167–186 ms for the per-token `tmac_g8` loop (4.5–4.7x). It is exact W1.58 (max abs 2.0e-06, rel-L2 8e-07), so it needs no A4/A5 activation quantisation or outlier split.
2. **Crossover.** Popcount scales about linearly, at ~12.5 ms per token. Tiled expand is nearly flat: 28.2–28.4 ms at B=1, 29.7–31.5 at B=2, 31.6–32.7 at B=4, 37.4–39.4 at B=8, 52.7–54.3 at B=16 (T=16).
   - Tiled expand beats popcount from **B≈3** (B=2: about 25 ms popcount vs 29–31 ms tiled; B=4: about 50 vs 31–32).
   - It beats `tmac_g8` from B=2 (1.36–1.48x).
   - At B=1 the 18:55 popcount default stays best on the packed path (12.4–12.6 ms, 1.65–1.68x vs `tmac_g8`).
3. **vs int8-expand:** 0.55–0.87x at B=8 (0.73x interleaved) and 0.41–0.45x at B=16. int8-expand's single full-matrix cast is cheaper in numpy than the LUT take plus tile cast, and its relative cost shrinks as B grows. In exchange, tiled expand uses **7.6x less memory** (18.43 vs 140.25 MB including scratch, 3.96x on resident weights alone). On a memory-bound or RSS-capped i2sd box, it is the B≥3 path that keeps the 2-bit residency.
4. **T=16 vs T=32:** the two are tied up to B=4 (T=32 is ~4% faster). From B=8, T=16 wins (B=16: 52.7–54.3 vs 69.5–74.0 ms), so T=16 is the default. Of the expand variants, the u32 LUT beats the f32 LUT4 gather by 1.2–1.6x and unpackbits by 1.4–1.7x.

Recommended dispatch for i2sd/bitnetd decode at 17.55 MB resident:
- **B=1:** the 18:55 popcount default (one p = 0.993, top-32 / top-8 splits).
- **B≥3:** tiled u32-LUT expand T=16 (exact W1.58).
- **int8-expand + BLAS** only where an extra ~123 MB of RSS per layer is acceptable.

**Caveat:** weights and tokens are synthetic, and everything is single-process numpy with 8-thread BLAS. Run-to-run spread for int8-expand is large (B=8 medians 20.4–34.1 ms) because its 70.8 MB scratch competes for cache and memory bandwidth. `f32_resident` is still 4–15x faster than every packed path, at 15.8x the memory.

## Pair files this wave

At Quant Chief 02:05 MT (`last_run` 09:10Z), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs are unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was 3eaf332 (02:48 MT, an unattributed cloudcover95 commit adding only `rails/scratchdraft/scratch_outside_repo_check.py`) on top of 38ca9dd (the 18:55 note).
- No Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours. Chief's LLC skip path (JuniorHome 0226f9b) is in another repo.
- This note adds one file under `junior_osai/vault/findings/`.

## Next question

Can the tiled expand get cheaper than int8-expand at B=8–16? Options:
- Expand each tile once into an int8 tile and use an int8 → f32 matmul on that tile, i.e. drop the u32 take.
- Fuse q/k/v and gate/up into one tiled pass over a concatenated pack, which shares the activation reads and halves the Python loop count.
- Keep a small, L2-resident f32 cache of the hottest tiles across decode steps.

Measure each at B=8/16 against int8-expand at ≤ 20 MB total.

close: sha256(prev || this)[:16] = 7994a1c904f96382
