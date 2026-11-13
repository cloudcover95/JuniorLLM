---
local: true
date: 2026-11-13
wave: "2026-11-13 05:55 (ran ~05:52–05:54 MT; Chief last_run 2026-11-13T12:13:40Z ~05:13 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-bitplane-activation, a4, a5, a6, per-group-scale, popcount, tmac-g8, numpy, viable]
---

# A5 per-row bitplane activations × packed ternary weights reach cos 0.988 vs the weight-only path and run 1.61–1.81x faster than `tmac_g8` at B=1 (same 1.64 MB packed bytes), so viable for the packed numpy fallback at B=1–2; per-group g=128 scales add only +0.001 cos and are ~5x slower

## Question

The prior note [[2026-11-13_0255_a2-a3-a4-bitplane-activation-x-packed-ternary-gemv-b1-b4-vs-tmac-g8]] found A4 stuck at cos ≈ 0.97 vs the weight-only W1.58 path and asked: does A4 with per-group (g=128) scales, or A5 (5 planes, 10 popcounts), reach cos ≥ 0.98 on multi-row heavy-tailed activations while staying ≥ 1.5x faster than `tmac_g8` at B=1? A6 was added to bracket the bar. Quant Chief `last_tips.json` `last_run` 2026-11-13T12:13:40Z (~05:13 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Out=In=2560 single GEMV, B=1/2/4, one process, numpy 2.5.4, 8-core virtualised Xeon (load avg 0.50 → 1.45 across the runs). Weights absmean-ternarised, stored as packed uint64 mask/sign planes (1,638,400 B) plus per-row Σw (10 KB) or per-(row, group) Σw (409,600 B int32 for g=128). Activations quantised to k bits, uniform symmetric x ≈ h·(2q − (2^k−1)) with h = c·absmean over the row or over each 128-wide group; c is the pooled-cos-optimal grid point from `err.py` on a 16-row calibration draw (seed+1). Each bit of q is one packed plane, 2 popcounts per plane, output tile T=1024; per-group variants keep popcount sums per group (2 words), subtract L·Σw_g and combine with the group scales via einsum. Quantisation and packing are **inside** the timed path.

Accuracy is measured on a separate 32-row t(4)·0.05 eval draw (seed+2): pooled cos, per-row median and min cos, rel mae, all vs the weight-only path (what `tmac_g8` computes, its own error ~3e-7). Each variant is also checked exact against the f64 product of its dequantised activations and the trit weights. Baselines: `tmac_g8` (f32-activation LUT over the transposed packed layout, the current packed fallback), `int8_scratch`, `f32_resident`. 9 timed repeats × 10–20 passes after 3 warmups; three full runs.

Command (offline, no network, nothing bound):

```
cd /workspace/bench0555_1113 && python3 err.py | tee err.txt && for i in 1 2 3; do python3 bench.py results_run$i.json | tee run$i.txt; done
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""A4/A5/A6 multi-bitplane activations with per-row vs per-group (g=128) absmean scales x ternary
mask/sign packed weights (2 popcounts per plane, tiled), vs weight-only tmac_g8 (f32 activations,
same 4.00x packed bytes), int8_scratch and f32_resident. Out=In=2560, B=1/2/4. Accuracy = cos vs
weight-only W1.58 path on a 32-row heavy-tailed eval set (separate draw) plus bench rows.
numpy-only, offline, single process. Usage: python3 bench.py results_runN.json"""
import json, statistics, sys, time
import numpy as np
assert hasattr(np, "bitwise_count")
OUT = IN = 2560
BATCHES = (1, 2, 4)
GS = 128; NG = IN // GS; WPG = GS // 64          # 20 groups, 2 uint64 words per group
VARIANTS = [(4, "row"), (4, "g128"), (5, "row"), (5, "g128"), (6, "row"), (6, "g128")]
CSTEP = {}                                       # filled from err.py output (cstep.json)
T = 1024
REPEATS, WARM = 9, 3
PASSES = {1: 20, 2: 15, 4: 10}
SEED = 202611130555
u64 = lambda p: np.ascontiguousarray(p).view(np.uint64)
def absmean_trits(w):
    g = float(np.mean(np.abs(w))) + 1e-7
    return np.clip(np.rint(w / g), -1, 1).astype(np.int8), g
def absmax_int8(w):
    m = float(np.max(np.abs(w))) / 127.0
    return np.clip(np.rint(w / m), -127, 127).astype(np.int8), m
def quant(x, k, mode, c):  # x ~ h*(2q-(2^k-1)); h per row (B,1) or per group (B,NG)
    L = (1 << k) - 1; B = x.shape[0]
    if mode == "row":
        h = c * np.mean(np.abs(x), axis=1, keepdims=True) + 1e-7
        hx = np.repeat(h, IN, axis=1)
    else:
        h = c * np.mean(np.abs(x.reshape(B, NG, GS)), axis=2) + 1e-7
        hx = np.repeat(h, GS, axis=1)
    q = np.clip(np.rint((x / hx + L) / 2), 0, L).astype(np.uint8)
    return q, h.astype(np.float32)
def deq(q, h, k, mode):
    hx = np.repeat(h.astype(np.float64), IN if mode == "row" else GS, axis=1)
    return (2.0 * q.astype(np.float64) - ((1 << k) - 1)) * hx
def pack_ms(t): return u64(np.packbits(t != 0, axis=-1)), u64(np.packbits(t < 0, axis=-1))
def akgemv(x, wm, ws, sumw, sumwg, gw, k, mode, c):
    q, h = quant(x, k, mode, c)
    planes = [u64(np.packbits((q >> j) & 1, axis=-1)) for j in range(k)]  # inside timed path
    B = x.shape[0]; L = (1 << k) - 1
    if mode == "row":
        acc = np.zeros((B, OUT), np.int32)
        for o0 in range(0, OUT, T):
            wmt = wm[None, o0:o0+T]; wst = ws[None, o0:o0+T]
            for j, p in enumerate(planes):
                a = p[:, None, :] & wmt
                s = np.bitwise_count(a).sum(-1, dtype=np.int32)
                a &= wst
                s -= 2 * np.bitwise_count(a).sum(-1, dtype=np.int32)
                acc[:, o0:o0+T] += s << (j + 1)
        acc -= L * sumw[None, :]
        return acc.astype(np.float32) * (np.float32(gw) * h)
    y = np.empty((B, OUT), np.float32)
    for o0 in range(0, OUT, T):
        wmt = wm[None, o0:o0+T]; wst = ws[None, o0:o0+T]
        n = wmt.shape[1]
        acc = np.zeros((B, n, NG), np.int32)
        for j, p in enumerate(planes):
            a = p[:, None, :] & wmt
            s = np.bitwise_count(a).reshape(B, n, NG, WPG).sum(-1, dtype=np.int32)
            a &= wst
            s -= 2 * np.bitwise_count(a).reshape(B, n, NG, WPG).sum(-1, dtype=np.int32)
            acc += s << (j + 1)
        acc -= L * sumwg[None, o0:o0+T]
        y[:, o0:o0+T] = np.einsum("bng,bg->bn", acc.astype(np.float32), h)
    y *= np.float32(gw)
    return y
LUT = np.array([[((c >> (2 * j)) & 3) for j in range(4)] for c in range(256)], np.int64)
LUTF_T = np.where(LUT == 2, -1, LUT).astype(np.float32).T
def pack2(t):
    c = np.where(t < 0, 2, t).astype(np.uint8).reshape(t.shape[0], -1, 4)
    p = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return np.ascontiguousarray(p.T)
G = 8
TS_IDX = np.empty((G, OUT), np.uint16); TS_F = np.empty((G, OUT), np.float32); TS_ACC = np.empty(OUT, np.float32)
OFFS = (np.arange(G, dtype=np.uint16) * 256)[:, None]
def tmac_g8(PT, gw, x1, y):
    q = PT.shape[0]
    Tf = (x1.reshape(q, 4) @ LUTF_T).ravel()
    y[:] = 0
    for c in range(0, q, G):
        np.add(PT[c:c+G], OFFS, out=TS_IDX, dtype=np.uint16)
        np.take(Tf[c*256:(c+G)*256], TS_IDX, out=TS_F)
        np.add.reduce(TS_F, axis=0, out=TS_ACC); y += TS_ACC
    y *= np.float32(gw)
def tmac_b(PT, gw, x, Y):
    for b in range(x.shape[0]): tmac_g8(PT, gw, x[b], Y[b])
SCR = np.empty((OUT, IN), np.float32)
def i8scr(x, q8, m, y):
    np.copyto(SCR, q8, casting='unsafe'); np.matmul(x, SCR.T, out=y); y *= np.float32(m)
def timed(fn, passes):
    for _ in range(WARM): fn()
    s = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(passes): fn()
        s.append((time.perf_counter() - t0) / passes * 1e3)
    return {"median_ms": statistics.median(s), "min_ms": min(s), "max_ms": max(s)}
def cos(a, b): return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
def rowcos(a, b): return [cos(a[i], b[i]) for i in range(a.shape[0])]
def weights():
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((OUT, IN)) * 0.02
    return rng, W
def main():
    CSTEP.update({tuple(json.loads(k)): v for k, v in json.load(open("cstep.json")).items()})
    rng, W = weights()
    tw, gw = absmean_trits(W); wm, ws = pack_ms(tw)
    sumw = tw.astype(np.int32).sum(1)
    sumwg = np.ascontiguousarray(tw.astype(np.int32).reshape(OUT, NG, GS).sum(2))
    PT = pack2(tw)
    q8, m8 = absmax_int8(W); Wd = q8.astype(np.float32) * np.float32(m8)
    twT = tw.T.astype(np.float64)
    # 32-row eval set, separate draw from err.py calibration (SEED+1) and bench rows
    xe = (np.random.default_rng(SEED + 2).standard_t(4, (32, IN)) * 0.05).astype(np.float32)
    woe = (xe.astype(np.float64) @ twT) * gw
    res = {"seed": SEED, "numpy": np.__version__, "cstep": {f"A{k}_{m}": c for (k, m), c in CSTEP.items()}, "tile": T,
           "group": GS, "bytes": {"ms_planes": int(wm.nbytes + ws.nbytes), "sumw_groups": int(sumwg.nbytes),
                                  "tmac_PT": int(PT.nbytes), "int8": int(q8.nbytes), "f32": int(Wd.nbytes)},
           "eval32": {}, "batches": {}}
    for (k, m) in VARIANTS:
        c = CSTEP[(k, m)]
        ye = akgemv(xe, wm, ws, sumw, sumwg, gw, k, m, c)
        qa, ha = quant(xe, k, m, c); ref = (deq(qa, ha, k, m) @ twT) * gw
        rc = rowcos(ye.astype(np.float64), woe)
        res["eval32"][f"A{k}_{m}"] = {"exact": bool(np.allclose(ye, ref, rtol=1e-4, atol=1e-5 * np.max(np.abs(ref)))),
            "cos_pooled": cos(ye.astype(np.float64), woe), "cos_row_median": statistics.median(rc), "cos_row_min": min(rc),
            "relmae": float(np.mean(np.abs(ye - woe)) / np.mean(np.abs(woe)))}
        e = res["eval32"][f"A{k}_{m}"]
        print(f"eval32 A{k}_{m} c {c} cos pooled {e['cos_pooled']:.4f} row med {e['cos_row_median']:.4f} min {e['cos_row_min']:.4f} relmae {e['relmae']:.3f} exact {e['exact']}", flush=True)
    for B in BATCHES:
        x = (rng.standard_t(4, (B, IN)) * 0.05).astype(np.float32)
        wo = (x.astype(np.float64) @ twT) * gw
        r = {}
        Y = np.empty((B, OUT), np.float32); tmac_b(PT, gw, x, Y)
        r["tmac_relerr_vs_wo"] = float(np.max(np.abs(Y - wo)) / np.max(np.abs(wo)))
        r["tmac_g8"] = timed(lambda: tmac_b(PT, gw, x, Y), PASSES[B])
        y = np.empty((B, OUT), np.float32)
        r["i8scr"] = timed(lambda: i8scr(x, q8, m8, y), PASSES[B])
        r["f32res"] = timed(lambda: np.matmul(x, Wd.T, out=y), PASSES[B])
        line = [f"B={B} tmac {r['tmac_g8']['median_ms']:.3f} i8scr {r['i8scr']['median_ms']:.3f} f32res {r['f32res']['median_ms']:.3f}"]
        for (k, m) in VARIANTS:
            c = CSTEP[(k, m)]; n = f"A{k}_{m}"
            ya = akgemv(x, wm, ws, sumw, sumwg, gw, k, m, c)
            r[f"{n}_cos_vs_wo"] = cos(ya.astype(np.float64), wo)
            r[n] = timed(lambda: akgemv(x, wm, ws, sumw, sumwg, gw, k, m, c), PASSES[B])
            md = r[n]["median_ms"]
            r[f"{n}_vs_tmac"] = r["tmac_g8"]["median_ms"] / md
            r[f"{n}_vs_tmac_worst"] = r["tmac_g8"]["min_ms"] / r[n]["max_ms"]
            r[f"{n}_vs_i8scr"] = r["i8scr"]["median_ms"] / md
            r[f"{n}_vs_f32res"] = r["f32res"]["median_ms"] / md
            line.append(f"{n} {md:.3f} ({r[n]['min_ms']:.3f}-{r[n]['max_ms']:.3f}) vs tmac {r[n+'_vs_tmac']:.2f}x (worst {r[n+'_vs_tmac_worst']:.2f}x) vs i8scr {r[n+'_vs_i8scr']:.2f}x vs f32res {r[n+'_vs_f32res']:.2f}x cos {r[n+'_cos_vs_wo']:.4f}")
        res["batches"][str(B)] = r
        print(" | ".join(line), flush=True)
    print("bytes", res["bytes"])
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
if __name__ == "__main__":
    main()
```
</details>

<details><summary>err.py</summary>

```python
# offline grid for the half-step constant c per (bits, scale mode); pooled cos vs weight-only W1.58 path
import json
import numpy as np
import bench
from bench import absmean_trits, SEED, IN, OUT, VARIANTS
rng, W = bench.weights()
tw, gw = absmean_trits(W); twf = tw.T.astype(np.float64)
x = (np.random.default_rng(SEED + 1).standard_t(4, (16, IN)) * 0.05)  # calibration draw, not used for eval/bench
wo = (x @ twf) * gw
out = {}
for k, m in VARIANTS:
    best = None
    for c in np.arange(0.06, 1.01, 0.02):
        q, h = bench.quant(x, k, m, c); y = (bench.deq(q, h, k, m) @ twf) * gw
        cs = bench.cos(y, wo)
        if best is None or cs > best[1]: best = (round(float(c), 2), cs)
    out[json.dumps([k, m])] = best[0]
    print(f"A{k}_{m} best c {best[0]} cos {best[1]:.4f}")
json.dump(out, open("cstep.json", "w"))
```
</details>

## Results

Bytes: packed mask/sign planes 1,638,400 B (same as `tmac_g8`'s `PT`); per-group Σw adds 409,600 B (as int32; int16 would halve it); int8 6,553,600 B (4.00x more); f32-resident 26,214,400 B (16.0x more).

Accuracy on the 32-row eval set (identical every run):

| variant | c | cos pooled | cos row median | cos row min | rel mae | exact |
|---|---|---|---|---|---|---|
| A4_row | 0.38 | 0.9735 | 0.9777 | 0.9318 | 0.225 | True |
| A4_g128 | 0.38 | 0.9753 | 0.9789 | 0.9348 | 0.217 | True |
| A5_row | 0.26 | 0.9879 | 0.9910 | 0.9570 | 0.150 | True |
| A5_g128 | 0.26 | 0.9892 | 0.9921 | 0.9608 | 0.141 | True |
| A6_row | 0.18 | 0.9949 | 0.9970 | 0.9746 | 0.095 | True |
| A6_g128 | 0.18 | 0.9956 | 0.9971 | 0.9783 | 0.089 | True |

Rows below cos 0.98 out of 32: A4_row 23, A5_row 3, A6_row 1.

Calibration grid (16 rows, seed+1):

```
A4_row best c 0.38 cos 0.9761
A4_g128 best c 0.38 cos 0.9771
A5_row best c 0.26 cos 0.9891
A5_g128 best c 0.26 cos 0.9898
A6_row best c 0.18 cos 0.9951
A6_g128 best c 0.18 cos 0.9955
```

ms per GEMV, median (min–max) over 9 repeats. Ratios are baseline / variant; "worst" is `tmac_g8` min / variant max. cos is on that batch's bench rows.

| variant | B | run | packed A_k | tmac_g8 | int8_scratch | f32_resident | vs tmac_g8 (worst) | vs int8_scratch | vs f32_resident | cos vs weight-only |
|---|---|---|---|---|---|---|---|---|---|---|
| A4_row | 1 | 1 | 1.048 (1.031–1.411) | 2.276 | 1.915 | 0.102 | **2.17x** (1.54x) | 1.83x | 0.10x | 0.9830 |
| A4_row | 1 | 2 | 1.074 (1.052–1.229) | 2.044 | 1.866 | 0.109 | **1.90x** (1.65x) | 1.74x | 0.10x | 0.9830 |
| A4_row | 1 | 3 | 1.033 (1.026–1.038) | 2.039 | 1.814 | 0.099 | **1.97x** (1.96x) | 1.76x | 0.10x | 0.9830 |
| A4_row | 2 | 1 | 1.962 (1.951–2.010) | 4.126 | 1.907 | 0.327 | **2.10x** (2.02x) | 0.97x | 0.17x | 0.9807 |
| A4_row | 2 | 2 | 2.038 (2.012–2.145) | 4.194 | 1.912 | 0.346 | **2.06x** (1.93x) | 0.94x | 0.17x | 0.9807 |
| A4_row | 2 | 3 | 1.976 (1.957–2.891) | 4.423 | 1.731 | 0.323 | **2.24x** (1.44x) | 0.88x | 0.16x | 0.9807 |
| A4_row | 4 | 1 | 4.110 (4.090–4.641) | 8.148 | 1.875 | 0.330 | **1.98x** (1.75x) | 0.46x | 0.08x | 0.9720 |
| A4_row | 4 | 2 | 4.235 (4.196–4.351) | 8.216 | 1.793 | 0.328 | **1.94x** (1.87x) | 0.42x | 0.08x | 0.9720 |
| A4_row | 4 | 3 | 4.065 (4.055–4.079) | 8.172 | 1.743 | 0.331 | **2.01x** (1.99x) | 0.43x | 0.08x | 0.9720 |
| A4_g128 | 1 | 1 | 5.206 (5.174–5.441) | 2.276 | 1.915 | 0.102 | **0.44x** (0.40x) | 0.37x | 0.02x | 0.9826 |
| A4_g128 | 1 | 2 | 5.191 (5.170–5.651) | 2.044 | 1.866 | 0.109 | **0.39x** (0.36x) | 0.36x | 0.02x | 0.9826 |
| A4_g128 | 1 | 3 | 5.230 (5.168–5.372) | 2.039 | 1.814 | 0.099 | **0.39x** (0.38x) | 0.35x | 0.02x | 0.9826 |
| A4_g128 | 2 | 1 | 10.307 (10.280–10.738) | 4.126 | 1.907 | 0.327 | **0.40x** (0.38x) | 0.19x | 0.03x | 0.9810 |
| A4_g128 | 2 | 2 | 10.474 (10.356–10.789) | 4.194 | 1.912 | 0.346 | **0.40x** (0.38x) | 0.18x | 0.03x | 0.9810 |
| A4_g128 | 2 | 3 | 10.293 (10.210–10.657) | 4.423 | 1.731 | 0.323 | **0.43x** (0.39x) | 0.17x | 0.03x | 0.9810 |
| A4_g128 | 4 | 1 | 20.833 (20.708–21.216) | 8.148 | 1.875 | 0.330 | **0.39x** (0.38x) | 0.09x | 0.02x | 0.9746 |
| A4_g128 | 4 | 2 | 20.973 (20.868–24.670) | 8.216 | 1.793 | 0.328 | **0.39x** (0.33x) | 0.09x | 0.02x | 0.9746 |
| A4_g128 | 4 | 3 | 20.935 (20.658–22.184) | 8.172 | 1.743 | 0.331 | **0.39x** (0.37x) | 0.08x | 0.02x | 0.9746 |
| A5_row | 1 | 1 | 1.254 (1.241–1.261) | 2.276 | 1.915 | 0.102 | **1.81x** (1.72x) | 1.53x | 0.08x | 0.9940 |
| A5_row | 1 | 2 | 1.270 (1.253–1.281) | 2.044 | 1.866 | 0.109 | **1.61x** (1.59x) | 1.47x | 0.09x | 0.9940 |
| A5_row | 1 | 3 | 1.265 (1.249–1.335) | 2.039 | 1.814 | 0.099 | **1.61x** (1.52x) | 1.43x | 0.08x | 0.9940 |
| A5_row | 2 | 1 | 2.422 (2.408–2.458) | 4.126 | 1.907 | 0.327 | **1.70x** (1.65x) | 0.79x | 0.14x | 0.9923 |
| A5_row | 2 | 2 | 2.474 (2.452–2.514) | 4.194 | 1.912 | 0.346 | **1.70x** (1.64x) | 0.77x | 0.14x | 0.9923 |
| A5_row | 2 | 3 | 2.413 (2.395–2.432) | 4.423 | 1.731 | 0.323 | **1.83x** (1.71x) | 0.72x | 0.13x | 0.9923 |
| A5_row | 4 | 1 | 5.029 (4.995–5.238) | 8.148 | 1.875 | 0.330 | **1.62x** (1.55x) | 0.37x | 0.07x | 0.9871 |
| A5_row | 4 | 2 | 5.527 (5.390–5.651) | 8.216 | 1.793 | 0.328 | **1.49x** (1.44x) | 0.32x | 0.06x | 0.9871 |
| A5_row | 4 | 3 | 5.084 (5.049–6.365) | 8.172 | 1.743 | 0.331 | **1.61x** (1.27x) | 0.34x | 0.07x | 0.9871 |
| A5_g128 | 1 | 1 | 6.483 (6.425–6.536) | 2.276 | 1.915 | 0.102 | **0.35x** (0.33x) | 0.30x | 0.02x | 0.9943 |
| A5_g128 | 1 | 2 | 6.670 (6.518–7.244) | 2.044 | 1.866 | 0.109 | **0.31x** (0.28x) | 0.28x | 0.02x | 0.9943 |
| A5_g128 | 1 | 3 | 6.449 (6.412–6.545) | 2.039 | 1.814 | 0.099 | **0.32x** (0.31x) | 0.28x | 0.02x | 0.9943 |
| A5_g128 | 2 | 1 | 12.886 (12.823–13.741) | 4.126 | 1.907 | 0.327 | **0.32x** (0.30x) | 0.15x | 0.03x | 0.9921 |
| A5_g128 | 2 | 2 | 13.123 (12.949–13.848) | 4.194 | 1.912 | 0.346 | **0.32x** (0.30x) | 0.15x | 0.03x | 0.9921 |
| A5_g128 | 2 | 3 | 12.783 (12.738–13.047) | 4.423 | 1.731 | 0.323 | **0.35x** (0.32x) | 0.14x | 0.03x | 0.9921 |
| A5_g128 | 4 | 1 | 25.782 (25.713–26.304) | 8.148 | 1.875 | 0.330 | **0.32x** (0.31x) | 0.07x | 0.01x | 0.9890 |
| A5_g128 | 4 | 2 | 26.486 (26.311–27.680) | 8.216 | 1.793 | 0.328 | **0.31x** (0.29x) | 0.07x | 0.01x | 0.9890 |
| A5_g128 | 4 | 3 | 28.106 (26.722–29.596) | 8.172 | 1.743 | 0.331 | **0.29x** (0.27x) | 0.06x | 0.01x | 0.9890 |
| A6_row | 1 | 1 | 1.548 (1.529–1.792) | 2.276 | 1.915 | 0.102 | **1.47x** (1.21x) | 1.24x | 0.07x | 0.9973 |
| A6_row | 1 | 2 | 1.627 (1.581–1.715) | 2.044 | 1.866 | 0.109 | **1.26x** (1.18x) | 1.15x | 0.07x | 0.9973 |
| A6_row | 1 | 3 | 1.521 (1.508–1.535) | 2.039 | 1.814 | 0.099 | **1.34x** (1.33x) | 1.19x | 0.06x | 0.9973 |
| A6_row | 2 | 1 | 2.876 (2.861–2.975) | 4.126 | 1.907 | 0.327 | **1.43x** (1.37x) | 0.66x | 0.11x | 0.9972 |
| A6_row | 2 | 2 | 2.967 (2.932–3.202) | 4.194 | 1.912 | 0.346 | **1.41x** (1.29x) | 0.64x | 0.12x | 0.9972 |
| A6_row | 2 | 3 | 2.873 (2.857–2.891) | 4.423 | 1.731 | 0.323 | **1.54x** (1.44x) | 0.60x | 0.11x | 0.9972 |
| A6_row | 4 | 1 | 5.822 (5.806–5.864) | 8.148 | 1.875 | 0.330 | **1.40x** (1.38x) | 0.32x | 0.06x | 0.9945 |
| A6_row | 4 | 2 | 6.011 (5.937–6.237) | 8.216 | 1.793 | 0.328 | **1.37x** (1.31x) | 0.30x | 0.05x | 0.9945 |
| A6_row | 4 | 3 | 5.896 (5.827–7.305) | 8.172 | 1.743 | 0.331 | **1.39x** (1.11x) | 0.30x | 0.06x | 0.9945 |
| A6_g128 | 1 | 1 | 7.717 (7.694–7.760) | 2.276 | 1.915 | 0.102 | **0.29x** (0.28x) | 0.25x | 0.01x | 0.9972 |
| A6_g128 | 1 | 2 | 8.056 (7.847–8.338) | 2.044 | 1.866 | 0.109 | **0.25x** (0.24x) | 0.23x | 0.01x | 0.9972 |
| A6_g128 | 1 | 3 | 7.697 (7.654–7.869) | 2.039 | 1.814 | 0.099 | **0.26x** (0.26x) | 0.24x | 0.01x | 0.9972 |
| A6_g128 | 2 | 1 | 15.291 (15.242–15.697) | 4.126 | 1.907 | 0.327 | **0.27x** (0.26x) | 0.12x | 0.02x | 0.9971 |
| A6_g128 | 2 | 2 | 15.588 (15.449–15.840) | 4.194 | 1.912 | 0.346 | **0.27x** (0.26x) | 0.12x | 0.02x | 0.9971 |
| A6_g128 | 2 | 3 | 15.504 (15.269–16.153) | 4.423 | 1.731 | 0.323 | **0.29x** (0.26x) | 0.11x | 0.02x | 0.9971 |
| A6_g128 | 4 | 1 | 31.125 (30.870–35.640) | 8.148 | 1.875 | 0.330 | **0.26x** (0.23x) | 0.06x | 0.01x | 0.9959 |
| A6_g128 | 4 | 2 | 31.752 (31.077–32.967) | 8.216 | 1.793 | 0.328 | **0.26x** (0.25x) | 0.06x | 0.01x | 0.9959 |
| A6_g128 | 4 | 3 | 30.891 (30.728–30.983) | 8.172 | 1.743 | 0.331 | **0.26x** (0.26x) | 0.06x | 0.01x | 0.9959 |

## Verdict

**viable, narrowly, for the packed numpy fallback at B=1–2.**

1. A5 per-row clears the accuracy bar on multi-row heavy-tailed inputs (pooled cos 0.988, row median 0.991, 29 of 32 rows ≥ 0.98, rel mae 0.15) and is 1.61–1.81x faster than `tmac_g8` at B=1 (worst-case 1.52–1.72x) and 1.70–1.83x at B=2, at the same 1.64 MB resident weights (4.00x under int8). At B=4 it drops to 1.49–1.62x, right at the bar.
2. A4 does not clear it with either scale (pooled 0.974/0.975; 23 of 32 rows under 0.98). Per-group g=128 scales buy only +0.001–0.002 cos at every bit width, cost 409,600 B of extra Σw, and run ~5x slower in numpy (the per-group reduce and einsum dominate), so per-row scales win outright.
3. A6 per-row is the accuracy-safe option (pooled 0.995, 31 of 32 rows ≥ 0.98) but only 1.26–1.47x vs `tmac_g8` at B=1, under the speed bar.
4. Same caveat as the last three notes: an f32-resident GEMV is still ~12x faster than A5 at B=1 (0.10 vs 1.27 ms) at 16x the memory, and A5 is only 1.43–1.53x faster than `int8_scratch` at B=1. The gain is real only against the current packed (memory-bound) fallback, not against float compute.

Recommendation: if the i2sd/bitnetd packed fallback must hold weights at 2 bits, A5 per-row bitplane activations with c=0.26 are the faster path at B=1–2; keep `tmac_g8` for B≥4 and anywhere a single bad row (min cos 0.957) matters.

## Pair files this wave

Quant Chief 05:13 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) both **no gap**, so no pair files were produced this wave. Roadmap blobs unchanged (JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923). Chief skip path: JuniorHome 72f4e11 (LLC advance of overnight-owned T13+, not eligible). JuniorLLM head at clone was afe7b2e (the previous findings note); this note touches only `junior_osai/vault/findings/`.

## Next question

Does A5 per-row hold cos ≥ 0.98 and ≥ 1.5x over `tmac_g8` on the full bitnetd layer shapes (q/k/v/o + gate/up/down, GQA, B=1), i.e. does the single-GEMV win survive the down-projection's 6912-wide input and the per-layer activation re-quantisation cost?

close: 6bcad8baf4b70283 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-13_0255)
