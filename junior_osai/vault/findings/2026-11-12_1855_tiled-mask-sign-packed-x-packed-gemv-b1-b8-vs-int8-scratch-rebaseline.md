---
local: true
date: 2026-11-12
wave: "2026-11-12 18:55 (ran ~18:56–19:05 MT; Chief last_run 2026-11-13T01:12:00Z ~18:12 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, packed-activation, bitplane, popcount, int8-baseline, rebaseline, numpy, not-viable]
---

# Packed×packed mask/sign GEMV still beats a scratch-reusing int8 GEMV at B=1–2 (5.9–6.7x, 2.8–2.9x calm), but ternary activations cost cos ≈ 0.80 vs the weight-only path and an f32-resident GEMV is still 3x faster, so not viable

## Question

The prior note [[2026-11-12_1455_n-layer-cold-weight-stack-b1-tmac-g8-vs-int8-scratch-vs-int8-expand-past-llc]] showed the old fresh-temp int8-expand baseline is allocator-bound, and asked: re-run the tiled mask/sign packed-activation × packed-weight GEMV of [[2026-11-08_0555_output-tiled-mask-sign-packed-gemv-b1-b8-crossover-vs-int8-expand]] (8.70x/4.20x/2.30x/1.20x at B=1/2/4/8 vs int8-expand) against `int8_scratch`. Does any batch size still clear 1.5x on speed, and at what error? Quant Chief `last_tips.json` `last_run` 2026-11-13T01:12:00Z (~18:12 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Out=In=2560 single GEMV, B=1/2/4/8, one process, numpy 2.5.4, 8-core virtualised Xeon (box load avg ~0.9–2.3 during runs). Weights absmean-ternarised and stored as two packed uint64 planes (mask = t≠0, sign = t<0). Activations absmean-ternarised and packed per row **inside the timed path**; dot = popcount(Xm&Wm) − 2·popcount(Xm&Wm&(Xs^Ws)), output tiles T ∈ {256, 1024} (best per batch reported).

Baselines:
- `int8_scratch` (new): int8-resident weights, one preallocated 2560×2560 f32 scratch reused every call (`np.copyto` dequant then `np.matmul` into a reused output). No per-call allocation.
- `int8_expand` (old): `x @ (q8.astype(f32) * m).T`, fresh temp each call.
- `f32_resident`: weights dequantised once and kept as f32 (26.2 MB), reference ceiling.

9 timed repeats × 8–20 passes after 3 warmups; three full runs. Integer exactness checked against an int32 trit×trit reference. Error is reported vs the f64 float GEMV and vs the weight-only ternary path (W1.58 × f32 activations, i.e. the `tmac_g8` arithmetic).

Command (offline, no network, nothing bound):

```
cd /workspace/bench1855_1112 && for i in 1 2 3; do python3 bench.py results_run$i.json | tee run$i.txt; done && python3 err.py
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Re-run of 2026-11-08_0555 output-tiled mask/sign packed-activation x packed-weight GEMV
(2 popcounts) at B=1/2/4/8, Out=In=2560, now against int8_scratch (int8-resident, one reused
f32 scratch, no per-call allocation) as well as the old fresh-temp int8-expand.
numpy-only, offline, single process. Usage: python3 bench.py results_runN.json"""
import json, statistics, sys, time
import numpy as np
assert hasattr(np, "bitwise_count")
OUT = IN = 2560
BATCHES = (1, 2, 4, 8)
TILES = (256, 1024)
REPEATS, WARM = 9, 3
SEED = 202611121855
PASSES = {1: 20, 2: 15, 4: 10, 8: 8}
def absmean_trits(w):
    g = float(np.mean(np.abs(w))) + 1e-7
    return np.clip(np.rint(w / g), -1, 1).astype(np.int8), g
def absmean_rows(x):
    g = np.mean(np.abs(x), axis=1) + 1e-7
    return np.clip(np.rint(x / g[:, None]), -1, 1).astype(np.int8), g
def absmax_int8(w):
    m = float(np.max(np.abs(w))) / 127.0
    return np.clip(np.rint(w / m), -127, 127).astype(np.int8), m
u64 = lambda p: np.ascontiguousarray(p).view(np.uint64)
def pack_ms(t):
    return u64(np.packbits(t != 0, axis=-1)), u64(np.packbits(t < 0, axis=-1))
def tiled2(x, wm, ws, gw, T):
    tx, gx = absmean_rows(x); xm, xs = pack_ms(tx)
    B = x.shape[0]; out = wm.shape[0]
    d = np.empty((B, out), np.int32)
    xm3 = xm[:, None, :]; xs3 = xs[:, None, :]
    for o0 in range(0, out, T):
        a = xm3 & wm[None, o0:o0+T]
        nz = np.bitwise_count(a).sum(-1, dtype=np.int32)
        a &= xs3 ^ ws[None, o0:o0+T]
        ng = np.bitwise_count(a).sum(-1, dtype=np.int32)
        d[:, o0:o0+T] = nz - 2 * ng
    return d.astype(np.float32) * (np.float32(gw) * gx.astype(np.float32)[:, None])
def i8exp(x, q8, m):  # old baseline: fresh f32 temp every call
    return x @ (q8.astype(np.float32) * np.float32(m)).T
SCR = np.empty((OUT, IN), np.float32)
def i8scr(x, q8, m, y):  # int8-resident, reused scratch, no allocation
    np.copyto(SCR, q8, casting='unsafe'); np.matmul(x, SCR.T, out=y); y *= np.float32(m); return y
def timed(fn, passes):
    for _ in range(WARM): fn()
    s = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(passes): fn()
        s.append((time.perf_counter() - t0) / passes * 1e3)
    return {"median_ms": statistics.median(s), "min_ms": min(s), "max_ms": max(s)}
def main():
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((OUT, IN)) * 0.02
    tw, gw = absmean_trits(W); wm, ws = pack_ms(tw)
    q8, m8 = absmax_int8(W); Wd = q8.astype(np.float32) * np.float32(m8)
    res = {"seed": SEED, "numpy": np.__version__, "bytes": {"ms_planes": int(wm.nbytes + ws.nbytes), "int8": int(q8.nbytes),
           "int8_scratch_extra_f32": int(SCR.nbytes), "f32": int(Wd.nbytes)}, "batches": {}}
    for B in BATCHES:
        x = (rng.standard_t(4, (B, IN)) * 0.05).astype(np.float32)
        tx, gx = absmean_rows(x)
        ref_int = tx.astype(np.int32) @ tw.astype(np.int32).T
        ref = ref_int.astype(np.float64) * gw * gx[:, None]
        yf = x.astype(np.float64) @ W.T
        y = np.empty((B, OUT), np.float32); r = {}
        for T in TILES:
            yt = tiled2(x, wm, ws, gw, T)
            di = np.rint(yt / (np.float32(gw) * gx.astype(np.float32)[:, None])).astype(np.int64)
            r[f"tiled{T}_int_exact"] = bool(np.array_equal(di, ref_int))
            r[f"tiled{T}"] = timed(lambda: tiled2(x, wm, ws, gw, T), PASSES[B])
        yi = i8scr(x, q8, m8, y).copy()
        r["i8scr"] = timed(lambda: i8scr(x, q8, m8, y), PASSES[B])
        r["i8exp"] = timed(lambda: i8exp(x, q8, m8), PASSES[B])
        r["f32res"] = timed(lambda: x @ Wd.T, PASSES[B])
        r["mae_tern_vs_f32"] = float(np.mean(np.abs(ref - yf)))
        r["mae_int8_vs_f32"] = float(np.mean(np.abs(yi - yf)))
        best = min(TILES, key=lambda T: r[f"tiled{T}"]["median_ms"]); r["best_tile"] = best
        tb = r[f"tiled{best}"]
        r["x_vs_i8scr"] = r["i8scr"]["median_ms"] / tb["median_ms"]
        r["x_vs_i8scr_worst"] = r["i8scr"]["min_ms"] / tb["max_ms"]
        r["x_vs_i8exp"] = r["i8exp"]["median_ms"] / tb["median_ms"]
        r["x_vs_f32res"] = r["f32res"]["median_ms"] / tb["median_ms"]
        res["batches"][str(B)] = r
        print(f"B={B} T={best} packed {tb['median_ms']:.3f} i8scr {r['i8scr']['median_ms']:.3f} i8exp {r['i8exp']['median_ms']:.3f} "
              f"f32res {r['f32res']['median_ms']:.3f} ms | vs i8scr {r['x_vs_i8scr']:.2f}x (worst {r['x_vs_i8scr_worst']:.2f}x) "
              f"vs i8exp {r['x_vs_i8exp']:.2f}x vs f32res {r['x_vs_f32res']:.2f}x exact {r[f'tiled{best}_int_exact']}", flush=True)
    print("bytes", res["bytes"])
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
if __name__ == "__main__":
    main()
```
</details>

<details><summary>err.py</summary>

```python
import numpy as np
from bench import absmean_trits, absmean_rows, SEED
rng=np.random.default_rng(SEED); W=rng.standard_normal((2560,2560))*0.02
tw,gw=absmean_trits(W)
for B in (1,2,4,8):
    x=(rng.standard_t(4,(B,2560))*0.05).astype(np.float32); tx,gx=absmean_rows(x)
    wo=(x.astype(np.float64)@tw.T.astype(np.float64))*gw            # weight-only ternary (A=f32), the tmac_g8 path
    bb=(tx.astype(np.float64)@tw.T.astype(np.float64))*gw*gx[:,None] # both-sides ternary (packed x packed)
    c=lambda a,b: float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
    print(f"B={B} rel_mae_vs_weight_only {np.mean(np.abs(bb-wo))/np.mean(np.abs(wo)):.3f} cos {c(bb,wo):.4f}")
```
</details>

## Results

Bytes: packed mask/sign planes 1,638,400 B; int8 6,553,600 B (4.00x more); int8_scratch also holds a 26,214,400 B f32 scratch (shared across layers); f32-resident 26,214,400 B (16.0x more than packed).

ms per GEMV, median (min–max) over 9 repeats. Ratios are baseline / packed; "worst" is int8_scratch min / packed max.

| B | run | tile | packed×packed | int8_scratch | int8_expand | f32_resident | vs int8_scratch (worst) | vs int8_expand | vs f32_resident | int-exact |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 256 | 0.380 (0.370–0.452) | 4.600 (4.199–9.131) | 5.590 | 0.108 | **12.11x** (9.29x) | 14.72x | 0.28x | True |
| 1 | 2 | 1024 | 0.339 (0.324–0.347) | 2.270 (1.879–2.630) | 3.279 | 0.264 | **6.71x** (5.42x) | 9.69x | 0.78x | True |
| 1 | 3 | 1024 | 0.338 (0.328–0.370) | 1.985 (1.816–2.298) | 2.719 | 0.104 | **5.86x** (4.91x) | 8.03x | 0.31x | True |
| 2 | 1 | 1024 | 0.687 (0.684–0.714) | 4.159 (3.548–4.892) | 5.854 | 0.606 | **6.05x** (4.97x) | 8.52x | 0.88x | True |
| 2 | 2 | 256 | 0.654 (0.639–2.165) | 1.909 (1.790–2.126) | 2.838 | 0.375 | **2.92x** (0.83x) | 4.34x | 0.57x | True |
| 2 | 3 | 256 | 0.672 (0.648–0.722) | 1.867 (1.732–2.200) | 2.728 | 0.332 | **2.78x** (2.40x) | 4.06x | 0.49x | True |
| 4 | 1 | 1024 | 1.342 (1.313–2.114) | 3.638 (2.579–8.123) | 4.528 | 0.860 | **2.71x** (1.22x) | 3.37x | 0.64x | True |
| 4 | 2 | 1024 | 1.356 (1.337–1.442) | 1.870 (1.755–2.497) | 2.741 | 0.369 | **1.38x** (1.22x) | 2.02x | 0.27x | True |
| 4 | 3 | 256 | 1.231 (1.186–1.610) | 1.834 (1.781–2.320) | 2.738 | 0.331 | **1.49x** (1.11x) | 2.23x | 0.27x | True |
| 8 | 1 | 256 | 2.457 (2.403–2.626) | 9.237 (2.028–24.000) | 4.428 | 0.767 | **3.76x** (0.77x) | 1.80x | 0.31x | True |
| 8 | 2 | 256 | 2.540 (2.355–2.612) | 1.841 (1.806–2.308) | 2.792 | 0.377 | **0.72x** (0.69x) | 1.10x | 0.15x | True |
| 8 | 3 | 256 | 2.456 (2.344–3.529) | 1.768 (1.666–3.052) | 2.861 | 0.367 | **0.72x** (0.47x) | 1.16x | 0.15x | True |

Run 1 was noisy (int8_scratch max 9–24 ms, load rising); runs 2–3 are the calm pair.

Error (run 1; identical inputs in every run):

| B | mae packed×packed vs f32 | mae int8 vs f32 |
|---|---|---|
| 1 | 0.0459 | 0.000708 |
| 2 | 0.0447 | 0.000716 |
| 4 | 0.0456 | 0.000720 |
| 8 | 0.0445 | 0.000710 |

Mean |y| of the float GEMV is ~0.06, so the packed×packed mae is ~75% of signal while int8 is ~1.2%.

Activation-ternarisation cost alone (both-sides ternary vs weight-only ternary, same trit weights):

```
B=1 rel_mae_vs_weight_only 0.654 cos 0.7953
B=2 rel_mae_vs_weight_only 0.636 cos 0.8032
B=4 rel_mae_vs_weight_only 0.654 cos 0.7925
B=8 rel_mae_vs_weight_only 0.642 cos 0.8030
```

## Verdict

**not-viable.** The speed win survives the rebaseline at small batch: on calm runs packed×packed is 5.86–6.71x faster than `int8_scratch` at B=1 and 2.78–2.92x at B=2, falls to 1.38–1.49x at B=4 and loses at B=8 (0.72x). Results are integer-exact and memory is exactly 4.00x smaller than int8. But:

1. Ternarising activations collapses accuracy: cos ≈ 0.79–0.80 and relative mae ≈ 0.64–0.65 against the weight-only W1.58 path the stack actually runs (BitNet-style models keep 8-bit activations). That is not acceptable error.
2. The comparison is only a win because numpy has no int8 GEMV: both int8 baselines pay a 26 MB f32 dequant per call. An f32-resident GEMV (dequant once) is 3.2x faster than packed×packed at B=1 (0.10 vs 0.34 ms calm) at 16x the memory, so the "speedup" mostly measures numpy's missing int8 kernel, not ternary compute.

Keep weight-only `tmac_g8` (f32 activations) for i2sd/bitnetd decode; do not adopt ternary activations. The 11-08 packed-activation speed wins should be read with this error caveat.

## Pair files this wave

Quant Chief 18:12 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) both **no gap**, so no pair files were produced this wave. Roadmap blobs unchanged (JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923). Chief skip path: JuniorHome 72f4e11 (LLC-bot, T13+ skill-pin, not eligible). This note touches only `junior_osai/vault/findings/`.

## Next question

Does a 2-bit (4-level, A2) activation stored as two extra bitplanes recover accuracy (cos ≥ 0.98 vs the weight-only path) while a 3–4-popcount packed GEMV still beats `tmac_g8` (f32 activations) by ≥ 1.5x at B=1?

close: 188fac8f96312534 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-12_1455)
