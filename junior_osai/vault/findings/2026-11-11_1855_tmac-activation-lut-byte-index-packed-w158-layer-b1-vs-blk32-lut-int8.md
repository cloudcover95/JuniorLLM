---
local: true
date: 2026-11-11
wave: "2026-11-11 18:55 (ran ~18:55–19:05 MT; Chief last_run 2026-11-12T01:05:00Z ~18:05 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, w158, i2s, t-mac, activation-lut, numpy, not-viable]
---

# A T-MAC-style activation LUT indexed by the packed weight byte is 1.22–1.33x faster than the blk32 weight-LUT expand at B=1, a real gain but under the 1.5x bar

## Question

The prior note [[2026-11-11_1455_gather-free-shift-mask-expand-blk32-packed-w158-layer-b1-vs-lut-take-int8]] pinned the weight-side expand at the 256-entry int8 LUT `take` and asked: at B=1, does flipping the table to the activation side (for each 4-input chunk of x, precompute the 256 signed partial sums `x4·trits(byte)`, then gather by the packed weight byte and sum) beat `blk32_lut` by at least 1.5x on the full layer while keeping the 4.00x weight-memory win? Quant Chief `last_tips.json` `last_run` 2026-11-12T01:05:00Z (~18:05 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same layer as the prior three notes: B=1, hidden 2560, FFN 6912 (gate, up, down), 20 query heads × 128, KV heads 20 (MHA) and 5 (GQA), assumed shapes. Absmean ternary weights, one f32 scale per matrix, 2-bit codes (trit+1) four per byte along the input dim; the layer time is the sum of the seven GEMVs. Single-threaded Python loops (OpenBLAS default threads inside `matmul`).

- `blk32_lut`: prior winner; 32-row blocks, `take` from a 256×4 int8 LUT, cast to f32 scratch, `matmul`.
- `tmac_gG`: activation table `T = x.reshape(In/4, 4) @ LUT.T` (In/4 × 256 f32, built inside every call and counted); weights stored transposed as `PT` (In/4 × Out uint8, the same bytes as the packed matrix). For each group of G byte-columns: `idx = PT[c:c+G] + 256·row` into a uint16 scratch, `take` from the flat table into a G×Out f32 scratch, `add.reduce` over the group, accumulate into y. G = 8, 16, 32, 64, 256.
- `tmac_col`: G=1, one `take` of a single table row by one byte-column per step, no index add.
- `tmac_build_only`: table build alone. `int8`: int8-resident cast + GEMV. `f32`: pre-scaled f32-resident GEMV (memory-heavy ceiling).

numpy 2.5.4, 8-core shared box (load 0.9–2.1 during the runs), seed `202611111855`, 2 warm-ups, 7 repeats × 3 passes, median and min–max, three back-to-back runs. Error is max |y − ref| / max |ref| against an f64 GEMV on the same trits.

Commands, from `/workspace/bench1855_1111`: `python3 bench.py results_run1.json`, `python3 bench.py results_run2.json`, `python3 bench.py results_run3.json`.

```python
# Full bitnetd-shaped layer at B=1 (hidden 2560, FFN 6912, 20 q heads x 128, KV heads 20 and 5), single-threaded:
# T-MAC-style activation LUT (per 4-input chunk, 256-entry table of signed partial sums indexed directly by the packed
# weight byte) vs blk32 row-blocked weight-LUT expand (prior winner) vs int8-resident (cast+GEMV) and f32-resident GEMV.
import numpy as np, time, json, sys
SEED = 202611111855
Hd, F, D, BS = 2560, 6912, 128, 32
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
LUT = np.array([[((b >> (2 * k)) & 3) for k in range(4)] for b in range(256)], dtype=np.int8) - 1
LUT[LUT == 2] = 0  # code 3 never appears; map to 0 for the activation table
LUTF_T = LUT.astype(np.float32).T.copy()  # (4,256)
def tern(o, i):
    W = rng.standard_normal((o, i)).astype(np.float32); s = np.abs(W).mean()
    return np.clip(np.rint(W / s), -1, 1).astype(np.int8), np.float32(s)
def pack(T):  # code = trit+1, 4 codes per byte, little-endian 2-bit fields along the input dim
    c = (T + 1).astype(np.uint8).reshape(T.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
SI8 = np.empty((BS, F), np.int8); SF = np.empty((BS, F), np.float32)
def blk_lut(P, s, x, y):  # prior winner: take(LUT int8) -> cast f32 -> matmul, 32-row blocks
    q = P.shape[1]; i = q * 4
    for r in range(0, P.shape[0], BS):
        n = min(BS, P.shape[0] - r)
        b8 = SI8[:n, :i].reshape(n, q, 4); np.take(LUT, P[r:r + n], axis=0, out=b8)
        bf = SF[:n, :i]; np.copyto(bf, SI8[:n, :i], casting='unsafe')
        np.matmul(bf, x, out=y[r:r + n])
    y *= s
GMAX = 256
TS_IDX = np.empty((GMAX, F), np.uint16); TS_F = np.empty((GMAX, F), np.float32); TS_ACC = np.empty(F, np.float32)
OFFS = (np.arange(GMAX, dtype=np.uint16) * 256)[:, None]
def make_tmac(G):
    def f(PT, s, x, y):  # PT: (In/4, Out) uint8, same bytes as P, transposed layout
        q, o = PT.shape
        T = (x.reshape(q, 4) @ LUTF_T)  # (q,256) activation table, built per call (counted)
        Tf = T.ravel()
        y[:] = 0
        for c in range(0, q, G):
            g = min(G, q - c)
            idx = TS_IDX[:g, :o]; np.add(PT[c:c + g], OFFS[:g], out=idx, dtype=np.uint16)
            v = TS_F[:g, :o]; np.take(Tf[c * 256:(c + g) * 256], idx, out=v)
            if g == 1: y += v[0]
            else: np.add.reduce(v, axis=0, out=TS_ACC[:o]); y += TS_ACC[:o]
        y *= s
    return f
def tmac_col(PT, s, x, y):  # G=1 per-column take, no index add
    q, o = PT.shape
    T = (x.reshape(q, 4) @ LUTF_T); y[:] = 0; v = TS_F[0, :o]
    for c in range(q):
        np.take(T[c], PT[c], out=v); y += v
    y *= s
def tbuild(PT, s, x, y):  # table build alone
    q = PT.shape[0]; T = (x.reshape(q, 4) @ LUTF_T)
def i8(W8, s, x, y): np.matmul(W8.astype(np.float32), x, out=y); y *= s
def f32(Wf, s, x, y): np.matmul(Wf, x, out=y)
def bench(f, *a):
    for _ in range(WARM): f(*a)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for _ in range(PASS): f(*a)
        ts.append((time.perf_counter() - t0) / PASS * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))
forms = {"f32": f32, "int8": i8, "blk32_lut": blk_lut, "tmac_col": tmac_col, "tmac_g8": make_tmac(8), "tmac_g16": make_tmac(16), "tmac_g32": make_tmac(32),
         "tmac_g64": make_tmac(64), "tmac_g256": make_tmac(256), "tmac_build_only": tbuild}
def scr(k, i, o):
    if k == "blk32_lut": return BS * F * 5
    if k.startswith("tmac"):
        G = {"tmac_col": 1, "tmac_g8": 8, "tmac_g16": 16, "tmac_g32": 32, "tmac_g64": 64, "tmac_g256": 256}.get(k, 0)
        return (i // 4) * 256 * 4 + G * o * 6 + o * 4
    return 0
out = {"seed": SEED, "numpy": np.__version__, "layers": {}}
x = rng.standard_normal(Hd).astype(np.float32); xf = rng.standard_normal(F).astype(np.float32)
for kvh in (20, 5):
    shapes = {"q": (Hd, Hd), "k": (kvh * D, Hd), "v": (kvh * D, Hd), "o": (Hd, Hd), "gate": (F, Hd), "up": (F, Hd), "down": (Hd, F)}
    acc = {k: [0.0, 0.0, 0.0] for k in forms}; by = {k: 0 for k in forms}; err = {k: 0.0 for k in forms}; sc = {k: 0 for k in forms}
    for nm, (o, i) in shapes.items():
        T8, s = tern(o, i); P = pack(T8); PT = np.ascontiguousarray(P.T); Wf = T8.astype(np.float32) * s; v = xf if i == F else x
        ref = (T8.astype(np.float64) @ v.astype(np.float64)) * float(s); y = np.empty(o, np.float32)
        for k, f in forms.items():
            W = T8 if k == "int8" else (Wf if k == "f32" else (P if k == "blk32_lut" else PT))
            m, lo, hi = bench(f, W, s, v, y)
            acc[k] = [acc[k][0] + m, acc[k][1] + lo, acc[k][2] + hi]; by[k] += W.nbytes; sc[k] = max(sc[k], scr(k, i, o))
            if k != "tmac_build_only":
                f(W, s, v, y); err[k] = max(err[k], float(np.max(np.abs(y - ref)) / np.max(np.abs(ref))))
        del Wf, T8, P, PT
    out["layers"][str(kvh)] = dict(ms=acc, bytes=by, scratch=sc, max_rel_err=err)
    print("kvh", kvh, flush=True)
    for k in forms:
        print(f"  {k:16s} {acc[k][0]:9.3f} ms [{acc[k][1]:.3f}-{acc[k][2]:.3f}] {by[k]/2**20:7.2f} MiB scratch {sc[k]/2**20:5.2f} MiB err {err[k]:.1e} vs_int8 {acc['int8'][0]/acc[k][0]:.2f}x vs_lut {acc['blk32_lut'][0]/acc[k][0]:.2f}x", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Seven-matrix layer weight stage, ms median run1 / run2 / run3 (run3 min–max). The `int8` row was noisy (run1 medians 58.3 and 48.6 ms with spreads to 82 ms under box contention), so the int8 ratios use run2 / run3 only; the ratios against `blk32_lut` are stable across all three runs.

| KV heads | form | resident MiB | scratch MiB | ms run1 / run2 / run3 (run3 min–max) | vs blk32_lut (r1 / r2 / r3) | vs int8 (r2 / r3) | max rel err |
|---|---|---|---|---|---|---|---|
| 20 | f32 | 302.50 | 0 | 1.68 / 1.41 / 1.62 (1.50–8.91) | n/a | 32.4x / 22.9x | 4.2e-07 |
| 20 | int8 | 75.62 | 0 | 58.28 / 45.75 / 37.11 (33.76–54.32) | 0.61x / 0.78x / 0.99x | 1.00x | 4.7e-07 |
| 20 | blk32_lut | 18.91 | 1.05 | 35.59 / 35.82 / 36.79 (35.10–46.05) | 1.00x | 1.28x / 1.01x | 4.7e-07 |
| 20 | tmac_col | 18.91 | 1.71 | 32.96 / 32.15 / 33.42 (32.15–36.32) | 1.08x / 1.11x / 1.10x | 1.42x / 1.11x | 1.3e-06 |
| 20 | **tmac_g8** | 18.91 | 1.81 | **27.90 / 27.75 / 28.27** (27.67–30.63) | **1.28x / 1.29x / 1.30x** | 1.65x / 1.31x | 5.5e-07 |
| 20 | tmac_g16 | 18.91 | 1.93 | 29.21 / 28.31 / 28.18 (27.43–30.01) | 1.22x / 1.27x / 1.31x | 1.62x / 1.32x | 3.0e-07 |
| 20 | tmac_g32 | 18.91 | 2.17 | 32.61 / 30.26 / 31.09 (30.17–34.91) | 1.09x / 1.18x / 1.18x | 1.51x / 1.19x | 3.9e-07 |
| 20 | tmac_g64 | 18.91 | 3.18 | 39.67 / 34.98 / 35.89 (34.73–40.22) | 0.90x / 1.02x / 1.03x | 1.31x / 1.03x | 2.2e-07 |
| 20 | tmac_g256 | 18.91 | 10.78 | 38.94 / 36.86 / 38.30 (37.16–39.69) | 0.91x / 0.97x / 0.96x | 1.24x / 0.97x | 5.5e-07 |
| 20 | tmac_build_only | n/a | 1.70 | 0.144 / 0.137 / 0.139 | n/a | n/a | n/a |
| 5 | f32 | 265.00 | 0 | 2.16 / 1.27 / 1.41 (1.34–1.77) | n/a | 29.1x / 24.1x | 4.3e-07 |
| 5 | int8 | 66.25 | 0 | 48.63 / 36.80 / 34.05 (31.01–40.99) | 0.69x / 0.86x / 0.96x | 1.00x | 4.6e-07 |
| 5 | blk32_lut | 16.56 | 1.05 | 33.39 / 31.62 / 32.81 (31.34–41.59) | 1.00x | 1.16x / 1.04x | 4.6e-07 |
| 5 | tmac_col | 16.56 | 1.71 | 30.68 / 30.11 / 30.73 (29.72–42.31) | 1.09x / 1.05x / 1.07x | 1.22x / 1.11x | 1.4e-06 |
| 5 | **tmac_g8** | 16.56 | 1.81 | **25.10 / 25.30 / 26.01** (25.12–30.64) | **1.33x / 1.25x / 1.26x** | 1.45x / 1.31x | 4.7e-07 |
| 5 | tmac_g16 | 16.56 | 1.93 | 26.05 / 25.15 / 26.55 (24.91–30.89) | 1.28x / 1.26x / 1.24x | 1.46x / 1.28x | 4.1e-07 |
| 5 | tmac_g32 | 16.56 | 2.17 | 28.03 / 29.53 / 28.22 (26.98–29.54) | 1.19x / 1.07x / 1.16x | 1.25x / 1.21x | 2.9e-07 |
| 5 | tmac_g64 | 16.56 | 3.18 | 32.72 / 32.03 / 31.81 (30.32–34.03) | 1.02x / 0.99x / 1.03x | 1.15x / 1.07x | 2.4e-07 |
| 5 | tmac_g256 | 16.56 | 10.78 | 33.77 / 34.23 / 33.40 (31.92–43.54) | 0.99x / 0.92x / 0.98x | 1.08x / 1.02x | 4.9e-07 |
| 5 | tmac_build_only | n/a | 1.70 | 0.158 / 0.148 / 0.148 | n/a | n/a | n/a |

Resident packed weights stay exactly 4.00x smaller than int8 for every packed form (the transposed `PT` layout is the same bytes); with the ~1.8 MiB table + scratch of `tmac_g8` the resident total is 3.66x (MHA) / 3.63x (GQA) smaller than int8. The activation-table build is ~0.14–0.16 ms per layer (under 0.6% of the layer), and in a real layer q/k/v and gate/up share one input, so it could be built 4 times per layer instead of 7. Small groups win because the uint16 index add, the gather and the reduce all stay in L1/L2 (G=8 moves ~55 KiB of f32 per step for the 6912-row matrices); at G≥64 the G×Out scratch spills L2 and the gain disappears. The activation LUT does one gather and one add per packed byte instead of four expanded trits, one cast and four multiply-adds, which is why it is the first numpy form to beat the weight-LUT expand; numpy still pays a separate pass for the index add, the gather and the reduce, so it stops at ~1.3x. Error stays at float32-noise level (≤5.5e-07; `tmac_col` 1.4e-06 from 640–1728 sequential f32 adds).

## Verdict

**Not viable at the 1.5x bar, but the new best packed B=1 path.** `tmac_g8` (and `tmac_g16`, within noise) is 1.25–1.33x faster than `blk32_lut` in every run on both layer shapes, and 1.31x of int8-resident in the calmer run3 (1.45–1.65x in run2, where int8 was slowed by contention), at 3.6x fewer resident bytes. That is a consistent improvement over the 1.10–1.30x-of-int8 blk32 path, but it does not reach 1.5x over the current packed path, so it is recorded as not worth a ping. If the bitnetd numpy fallback is kept, switch its packed B=1 GEMV from `blk32_lut` to the activation LUT with G=8 over a transposed packed layout.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-12T01:05:00Z (~18:05 MT) wrote pair_A/pair_B **no gap** ("18:05 11/11: JuniorHome f281d62 (nico juniorcloudllc) Automations PAUSED left to LLC; roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). JuniorLLM tip at clone time: [b3ea51a](https://github.com/cloudcover95/JuniorLLM/commit/b3ea51a) (the prior findings note); no Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours.

## Next question

The activation LUT spends most of its time on the separate uint16 index-add pass. If the packed weights are stored pre-offset (the 2 index bytes per packed byte computed once at load, as a uint16 `PT + 256·row` array), the per-call add disappears but resident bytes double to 2.00x of int8. Does that pre-offset activation LUT at G=8 reach 1.5x of `blk32_lut` at B=1, and if it does, is a 2x memory win at 1.5x speed a better trade than the current 4x memory win at ~1.3x?

close: c6a9d6373286ad56 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-11_1455)
