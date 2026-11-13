---
local: true
date: 2026-11-13
wave: "2026-11-13 02:55 (ran ~02:55–03:05 MT; Chief last_run 2026-11-13T09:12:00Z ~02:12 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-bitplane-activation, a2, a3, a4, popcount, tmac-g8, numpy, not-viable]
---

# A2/A3/A4 bitplane activations × packed ternary weights beat weight-only `tmac_g8` by 2.0–4.3x at B=1–4, but none reaches cos ≥ 0.98 vs the weight-only path reliably (A4 ≈ 0.97), so not viable

## Question

The prior note [[2026-11-12_1855_tiled-mask-sign-packed-x-packed-gemv-b1-b8-vs-int8-scratch-rebaseline]] found ternary (A1.58) activations cost cos ≈ 0.80 vs the weight-only W1.58 path, and asked: does a 2-bit (A2) activation held as extra bitplanes recover accuracy (cos ≥ 0.98 vs weight-only) while the packed popcount GEMV still beats `tmac_g8` (f32 activations, same 4.00x packed bytes) by ≥ 1.5x at B=1? A3 and A4 were added to find where the accuracy bar is actually met. Quant Chief `last_tips.json` `last_run` 2026-11-13T09:12:00Z (~02:12 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Out=In=2560 single GEMV, B=1/2/4, one process, numpy 2.5.4, 8-core virtualised Xeon (load avg 0.49 → 1.24 across the runs). Weights absmean-ternarised, stored as packed uint64 mask/sign planes (1,638,400 B), with per-row Σw precomputed (10 KB). Activations quantised per row to k bits, uniform symmetric levels x ≈ h·(2q − (2^k−1)), h = c·absmean(row), with c picked offline by `err.py` as the cos-optimal grid point on a separate input draw (A2 0.80, A3 0.56, A4 0.44). Each bit of q is one packed plane; dot = Σ_j 2^(j+1)·[popcount(P_j&Wm) − 2·popcount(P_j&Wm&Ws)] − (2^k−1)·Σw, i.e. 2 popcounts per plane, output tile T=1024. Activation quantisation and packing are **inside** the timed path.

Baselines: `tmac_g8` (the 2026-11-11_1855 activation-LUT form over the transposed packed layout, f32 activations, table built every call, B>1 loops rows), `int8_scratch` (int8-resident, one reused f32 scratch), `f32_resident` (26.2 MB, ceiling). 9 timed repeats × 10–20 passes after 3 warmups; three full runs. Exactness checked against the f64 product of the dequantised A_k activations and the trit weights. Accuracy is cos and relative mae vs the weight-only ternary path (W1.58 × f32 activations), which is what `tmac_g8` computes (its own error 2–4e-7).

Command (offline, no network, nothing bound):

```
cd /workspace/bench0255_1113 && python3 err.py > err.txt && for i in 1 2 3; do python3 bench.py results_run$i.json | tee run$i.txt; done
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""A2/A3 multi-bitplane activations x ternary mask/sign weights packed GEMV (2 popcounts per
activation plane, tiled) vs weight-only tmac_g8 (f32 activations, same 4.00x packed bytes),
int8_scratch and f32_resident. Out=In=2560, B=1/2/4. numpy-only, offline, single process.
Usage: python3 bench.py results_runN.json"""
import json, statistics, sys, time
import numpy as np
assert hasattr(np, "bitwise_count")
OUT = IN = 2560
BATCHES = (1, 2, 4)
KS = (2, 3, 4)       # activation bitplanes: A2=4, A3=8, A4=16 levels (uniform, symmetric)
CSTEP = {2: 0.80, 3: 0.56, 4: 0.44}  # half-step h = CSTEP[k] * absmean(row); picked offline by err.py grid (cos-optimal)
T = 1024
REPEATS, WARM = 9, 3
PASSES = {1: 20, 2: 15, 4: 10}
SEED = 202611130255
u64 = lambda p: np.ascontiguousarray(p).view(np.uint64)
def absmean_trits(w):
    g = float(np.mean(np.abs(w))) + 1e-7
    return np.clip(np.rint(w / g), -1, 1).astype(np.int8), g
def absmax_int8(w):
    m = float(np.max(np.abs(w))) / 127.0
    return np.clip(np.rint(w / m), -127, 127).astype(np.int8), m
def quant_ak(x, k):  # x ~ h*(2q-(2^k-1)), q in [0, 2^k-1]
    L = (1 << k) - 1
    h = CSTEP[k] * np.mean(np.abs(x), axis=1) + 1e-7
    q = np.clip(np.rint((x / h[:, None] + L) / 2), 0, L).astype(np.uint8)
    return q, h
def deq_ak(q, h, k): return (2.0 * q.astype(np.float64) - ((1 << k) - 1)) * h[:, None]
def pack_ms(t): return u64(np.packbits(t != 0, axis=-1)), u64(np.packbits(t < 0, axis=-1))
def akgemv(x, wm, ws, sumw, gw, k):
    q, h = quant_ak(x, k)
    planes = [u64(np.packbits((q >> j) & 1, axis=-1)) for j in range(k)]  # packing inside timed path
    B = x.shape[0]
    acc = np.zeros((B, OUT), np.int32)
    for o0 in range(0, OUT, T):
        wmt = wm[None, o0:o0+T]; wst = ws[None, o0:o0+T]
        for j, p in enumerate(planes):
            a = p[:, None, :] & wmt
            s = np.bitwise_count(a).sum(-1, dtype=np.int32)
            a &= wst
            s -= 2 * np.bitwise_count(a).sum(-1, dtype=np.int32)
            acc[:, o0:o0+T] += s << (j + 1)
    acc -= ((1 << k) - 1) * sumw[None, :]
    return acc.astype(np.float32) * (np.float32(gw) * h.astype(np.float32)[:, None])
LUT = np.array([[((c >> (2 * j)) & 3) for j in range(4)] for c in range(256)], np.int64)
LUTF_T = np.where(LUT == 2, -1, LUT).astype(np.float32).T  # code 0->0, 1->+1, 2->-1
def pack2(t):  # (Out, In) trits -> (In/4, Out) uint8, transposed layout for tmac
    c = np.where(t < 0, 2, t).astype(np.uint8).reshape(t.shape[0], -1, 4)
    p = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return np.ascontiguousarray(p.T)
G = 8
TS_IDX = np.empty((G, OUT), np.uint16); TS_F = np.empty((G, OUT), np.float32); TS_ACC = np.empty(OUT, np.float32)
OFFS = (np.arange(G, dtype=np.uint16) * 256)[:, None]
def tmac_g8(PT, gw, x1, y):  # single vector (B=1); B>1 loops rows
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
def main():
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((OUT, IN)) * 0.02
    tw, gw = absmean_trits(W); wm, ws = pack_ms(tw)
    sumw = tw.astype(np.int32).sum(1)
    PT = pack2(tw)
    q8, m8 = absmax_int8(W); Wd = q8.astype(np.float32) * np.float32(m8)
    res = {"seed": SEED, "numpy": np.__version__, "cstep": CSTEP, "tile": T,
           "bytes": {"ms_planes": int(wm.nbytes + ws.nbytes), "tmac_PT": int(PT.nbytes), "int8": int(q8.nbytes),
                     "f32": int(Wd.nbytes)}, "batches": {}}
    for B in BATCHES:
        x = (rng.standard_t(4, (B, IN)) * 0.05).astype(np.float32)
        wo = (x.astype(np.float64) @ tw.T.astype(np.float64)) * gw   # weight-only W1.58 reference
        r = {}
        Y = np.empty((B, OUT), np.float32); tmac_b(PT, gw, x, Y)
        r["tmac_relerr_vs_wo"] = float(np.max(np.abs(Y - wo)) / np.max(np.abs(wo)))
        r["tmac_g8"] = timed(lambda: tmac_b(PT, gw, x, Y), PASSES[B])
        y = np.empty((B, OUT), np.float32)
        r["i8scr"] = timed(lambda: i8scr(x, q8, m8, y), PASSES[B])
        r["f32res"] = timed(lambda: np.matmul(x, Wd.T, out=y), PASSES[B])
        for k in KS:
            ya = akgemv(x, wm, ws, sumw, gw, k)
            qa, ha = quant_ak(x, k)
            ref = (deq_ak(qa, ha, k) @ tw.T.astype(np.float64)) * gw
            r[f"A{k}_exact"] = bool(np.allclose(ya, ref, rtol=1e-5, atol=1e-6 * np.max(np.abs(ref))))
            r[f"A{k}_cos_vs_wo"] = cos(ya.astype(np.float64), wo)
            r[f"A{k}_relmae_vs_wo"] = float(np.mean(np.abs(ya - wo)) / np.mean(np.abs(wo)))
            r[f"A{k}"] = timed(lambda: akgemv(x, wm, ws, sumw, gw, k), PASSES[B])
            m = r[f"A{k}"]["median_ms"]
            r[f"A{k}_vs_tmac"] = r["tmac_g8"]["median_ms"] / m
            r[f"A{k}_vs_tmac_worst"] = r["tmac_g8"]["min_ms"] / r[f"A{k}"]["max_ms"]
            r[f"A{k}_vs_i8scr"] = r["i8scr"]["median_ms"] / m
            r[f"A{k}_vs_f32res"] = r["f32res"]["median_ms"] / m
        res["batches"][str(B)] = r
        print(f"B={B} tmac {r['tmac_g8']['median_ms']:.3f} i8scr {r['i8scr']['median_ms']:.3f} f32res {r['f32res']['median_ms']:.3f} | "
              + " | ".join(f"A{k} {r[f'A{k}']['median_ms']:.3f} ({r[f'A{k}']['min_ms']:.3f}-{r[f'A{k}']['max_ms']:.3f}) vs tmac {r[f'A{k}_vs_tmac']:.2f}x "
                           f"(worst {r[f'A{k}_vs_tmac_worst']:.2f}x) vs i8scr {r[f'A{k}_vs_i8scr']:.2f}x vs f32res {r[f'A{k}_vs_f32res']:.2f}x "
                           f"cos {r[f'A{k}_cos_vs_wo']:.4f} relmae {r[f'A{k}_relmae_vs_wo']:.3f} exact {r[f'A{k}_exact']}" for k in KS)
              + f" | tmac err {r['tmac_relerr_vs_wo']:.1e}", flush=True)
    print("bytes", res["bytes"])
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
if __name__ == "__main__":
    main()
```
</details>

<details><summary>err.py</summary>

```python
# offline grid for the A_k half-step constant c (h = c*absmean(row)); cos vs weight-only W1.58 path
import numpy as np
import bench
from bench import absmean_trits, SEED, IN, OUT
rng = np.random.default_rng(SEED); W = rng.standard_normal((OUT, IN)) * 0.02
tw, gw = absmean_trits(W); twf = tw.T.astype(np.float64)
x = (np.random.default_rng(SEED + 1).standard_t(4, (8, IN)) * 0.05)  # different draw than bench inputs
wo = (x @ twf) * gw
for k in (2, 3, 4):
    best = None
    for c in np.arange(0.10, 1.21, 0.02):
        bench.CSTEP[k] = c; q, h = bench.quant_ak(x, k); y = (bench.deq_ak(q, h, k) @ twf) * gw
        cs = bench.cos(y, wo)
        if best is None or cs > best[1]: best = (round(float(c), 2), cs)
    print(f"A{k} best c {best[0]} cos {best[1]:.4f}")
```
</details>

## Results

Bytes: packed mask/sign planes 1,638,400 B (same as `tmac_g8`'s transposed `PT`); int8 6,553,600 B (4.00x more); f32-resident 26,214,400 B (16.0x more). Per-call activation planes are k × 320 B per row (negligible).

ms per GEMV, median (min–max) over 9 repeats. Ratios are baseline / A_k; "worst" is `tmac_g8` min / A_k max. cos and rel mae are vs the weight-only W1.58 path (identical inputs each run).

| act | B | run | A_k packed | tmac_g8 | int8_scratch | f32_resident | vs tmac_g8 (worst) | vs int8_scratch | vs f32_resident | cos vs weight-only | rel mae | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 | 1 | 1 | 0.652 (0.579–0.864) | 2.471 | 3.421 | 0.117 | **3.79x** (2.78x) | 5.24x | 0.18x | 0.8915 | 0.453 | True |
| A2 | 1 | 2 | 0.590 (0.566–0.611) | 2.560 | 3.370 | 0.112 | **4.34x** (4.00x) | 5.71x | 0.19x | 0.8915 | 0.453 | True |
| A2 | 1 | 3 | 0.602 (0.564–0.620) | 2.296 | 3.832 | 0.108 | **3.82x** (3.47x) | 6.37x | 0.18x | 0.8915 | 0.453 | True |
| A2 | 2 | 1 | 1.067 (1.009–1.085) | 4.372 | 3.314 | 0.412 | **4.10x** (3.91x) | 3.11x | 0.39x | 0.8620 | 0.506 | True |
| A2 | 2 | 2 | 1.099 (1.032–1.115) | 4.459 | 3.517 | 0.413 | **4.06x** (3.94x) | 3.20x | 0.38x | 0.8620 | 0.506 | True |
| A2 | 2 | 3 | 1.081 (1.043–1.096) | 4.414 | 3.435 | 0.402 | **4.08x** (3.86x) | 3.18x | 0.37x | 0.8620 | 0.506 | True |
| A2 | 4 | 1 | 2.244 (2.182–2.352) | 8.769 | 3.281 | 0.397 | **3.91x** (3.64x) | 1.46x | 0.18x | 0.8641 | 0.503 | True |
| A2 | 4 | 2 | 2.289 (2.217–2.680) | 8.695 | 2.461 | 0.431 | **3.80x** (3.10x) | 1.07x | 0.19x | 0.8641 | 0.503 | True |
| A2 | 4 | 3 | 2.243 (2.185–2.401) | 9.007 | 3.662 | 0.402 | **4.02x** (3.70x) | 1.63x | 0.18x | 0.8641 | 0.503 | True |
| A3 | 1 | 1 | 0.905 (0.844–1.089) | 2.471 | 3.421 | 0.117 | **2.73x** (2.20x) | 3.78x | 0.13x | 0.9578 | 0.286 | True |
| A3 | 1 | 2 | 0.886 (0.861–1.112) | 2.560 | 3.370 | 0.112 | **2.89x** (2.20x) | 3.80x | 0.13x | 0.9578 | 0.286 | True |
| A3 | 1 | 3 | 0.853 (0.839–1.062) | 2.296 | 3.832 | 0.108 | **2.69x** (2.02x) | 4.49x | 0.13x | 0.9578 | 0.286 | True |
| A3 | 2 | 1 | 1.550 (1.527–1.587) | 4.372 | 3.314 | 0.412 | **2.82x** (2.68x) | 2.14x | 0.27x | 0.9404 | 0.337 | True |
| A3 | 2 | 2 | 1.611 (1.558–1.686) | 4.459 | 3.517 | 0.413 | **2.77x** (2.61x) | 2.18x | 0.26x | 0.9404 | 0.337 | True |
| A3 | 2 | 3 | 1.629 (1.578–1.660) | 4.414 | 3.435 | 0.402 | **2.71x** (2.55x) | 2.11x | 0.25x | 0.9404 | 0.337 | True |
| A3 | 4 | 1 | 3.286 (3.148–3.478) | 8.769 | 3.281 | 0.397 | **2.67x** (2.46x) | 1.00x | 0.12x | 0.9277 | 0.369 | True |
| A3 | 4 | 2 | 3.390 (3.247–3.559) | 8.695 | 2.461 | 0.431 | **2.56x** (2.33x) | 0.73x | 0.13x | 0.9277 | 0.369 | True |
| A3 | 4 | 3 | 3.226 (3.144–3.320) | 9.007 | 3.662 | 0.402 | **2.79x** (2.67x) | 1.14x | 0.12x | 0.9277 | 0.369 | True |
| A4 | 1 | 1 | 1.120 (1.082–1.152) | 2.471 | 3.421 | 0.117 | **2.21x** (2.08x) | 3.05x | 0.10x | 0.9811 | 0.193 | True |
| A4 | 1 | 2 | 1.147 (1.095–1.168) | 2.560 | 3.370 | 0.112 | **2.23x** (2.09x) | 2.94x | 0.10x | 0.9811 | 0.193 | True |
| A4 | 1 | 3 | 1.120 (1.114–1.182) | 2.296 | 3.832 | 0.108 | **2.05x** (1.82x) | 3.42x | 0.10x | 0.9811 | 0.193 | True |
| A4 | 2 | 1 | 2.033 (2.003–2.117) | 4.372 | 3.314 | 0.412 | **2.15x** (2.01x) | 1.63x | 0.20x | 0.9738 | 0.224 | True |
| A4 | 2 | 2 | 1.977 (1.949–2.226) | 4.459 | 3.517 | 0.413 | **2.26x** (1.97x) | 1.78x | 0.21x | 0.9738 | 0.224 | True |
| A4 | 2 | 3 | 2.122 (2.035–2.405) | 4.414 | 3.435 | 0.402 | **2.08x** (1.76x) | 1.62x | 0.19x | 0.9738 | 0.224 | True |
| A4 | 4 | 1 | 4.488 (4.308–4.766) | 8.769 | 3.281 | 0.397 | **1.95x** (1.80x) | 0.73x | 0.09x | 0.9593 | 0.273 | True |
| A4 | 4 | 2 | 4.399 (4.163–4.630) | 8.695 | 2.461 | 0.431 | **1.98x** (1.79x) | 0.56x | 0.10x | 0.9593 | 0.273 | True |
| A4 | 4 | 3 | 4.308 (4.172–5.148) | 9.007 | 3.662 | 0.402 | **2.09x** (1.72x) | 0.85x | 0.09x | 0.9593 | 0.273 | True |

Offline scale grid (8 rows, separate draw, t(4)·0.05 activations), best cos per bit-width:

```
A2 best c 0.8 cos 0.8772
A3 best c 0.56 cos 0.9431
A4 best c 0.44 cos 0.9732
```

## Verdict

**not-viable.** Speed is there: every form beats `tmac_g8` by ≥ 1.5x at every batch (A2 3.8–4.3x, A3 2.6–2.9x, A4 1.95–2.26x; worst-case A4 still 1.72–2.09x), at the same 4.00x-smaller-than-int8 resident bytes, and results are exact. But accuracy does not clear the bar:

1. A2 (the question) recovers only to cos 0.86–0.89 (rel mae ≈ 0.45–0.51), better than A1.58's 0.80 but far from 0.98.
2. A3 reaches 0.93–0.96; A4 reaches 0.959–0.981 on the bench rows and 0.973 on the 8-row offline grid. The single B=1 row at 0.981 is one 2560-output sample; the multi-row figure is ~0.97, under 0.98. Heavy-tailed (t(4)) activations are what hurt: a per-row absmean scale with few levels clips or wastes range.
3. As in the prior note, an f32-resident GEMV is still ~10x faster than A4 at B=1 (0.11 vs 1.12 ms) at 16x the memory, so the win over `tmac_g8` is a numpy kernel-shape effect (BLAS vs Python-level gathers), not ternary compute beating float.

Keep weight-only `tmac_g8` (f32 activations) for the packed numpy fallback; do not quantise activations below 8 bits on this path.

## Pair files this wave

Quant Chief 02:12 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) both **no gap**, so no pair files were produced this wave. Roadmap blobs unchanged (JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923). Chief skip path: JuniorHome 237c196 (LLC advance of overnight T13+, not eligible). JuniorLLM head at clone was 79dfc30 (02:51 MT, `rails/window/` LLC write-window guard); this note touches only `junior_osai/vault/findings/`.

## Next question

Does A4 with per-group (g=128) scales, or A5 (5 planes, 10 popcounts), reach cos ≥ 0.98 vs the weight-only path on multi-row heavy-tailed activations while staying ≥ 1.5x faster than `tmac_g8` at B=1?

close: 83006a457ab467de (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-12_1855)
