---
local: true
date: 2026-11-12
wave: "2026-11-12 14:55 (ran ~14:55–15:00 MT; Chief last_run 2026-11-12T21:12:00Z ~14:12 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, n-layer, cold-weights, llc, w158, t-mac, int8-baseline, numpy, not-viable]
---

# Cold weights across a 1–12 layer stack do not open a speed gap: packed tmac_g8 stays 0.91–1.14x of a scratch-reusing int8 GEMV (calm runs), and the old int8-expand baseline was allocator-bound

## Question

The prior note [[2026-11-12_1055_fused-qkv-gateup-stacked-tmac-index-packed-w158-layer-b1-vs-separate-tmac-g8]] asked: once N × int8 layer bytes exceed the box's last-level cache, so every layer's weights arrive cold from DRAM, at what layer count does the 4x smaller `tmac_g8` packed path beat cold `int8` by at least 1.5x on B=1 decode? Quant Chief `last_tips.json` `last_run` 2026-11-12T21:12:00Z (~14:12 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

GQA bitnetd-shaped layer (hidden 2560, FFN 6912, 20 q heads × 128, 5 KV heads; assumed shapes), seven GEMVs per layer, B=1, single-threaded. Twelve independent layers are resident at once (int8 795 MiB, packed 199 MiB); each timed "token" walks layers 0..N−1 in order for N = 1, 2, 4, 8, 12 and reports ms per layer. The box reports L2 16 MiB (8 × 2 MiB) and a 320 MiB shared L3 (virtualised Xeon, 8 cores, ~2–4 GiB free RAM), so int8 stacks of N ≥ 5 and packed stacks of N ≥ 20 cannot stay cache-resident; this run covers int8 past LLC and packed below it.

- `int8`: int8-resident, `W8.astype(f32)` fresh each call, then matmul (the int8-expand baseline used by the prior notes).
- `int8_scratch`: same int8-resident bytes, but expanded into one reused preallocated f32 buffer (`np.copyto`), so no per-call 70 MB allocation.
- `blk32_lut`: 2-bit packed codes, 32-row weight-LUT expand into reused scratch.
- `tmac_g8`: f32 activation table over the transposed uint8 packed index, G=8 (prior winner).

Uniform random trits (timing does not depend on the distribution), one f32 scale per matrix, error = max |y − ref| / max |ref| vs an f64 GEMV on layer 0. numpy 2.5.4, seed `202611121455`, 1 warm-up token, 5 timed tokens per N per form, forms interleaved per N, median and min–max, three back-to-back runs (load 3.4 → 6.7 across the runs, shared box).

Commands, from `/workspace/bench1455_1112`: `python3 bench.py results_run1.json`, `python3 bench.py results_run2.json`, `python3 bench.py results_run3.json`.

```python
# N-layer cold-weight B=1 decode stack (GQA bitnetd shapes: hidden 2560, FFN 6912, 20 q heads x 128, 5 KV heads),
# single-threaded. Each token pass walks layers 0..N-1 in order, so once N x resident bytes exceed the last-level
# cache every layer's weights arrive cold from DRAM. Forms: int8-resident (expand to f32 + matmul), blk32 weight-LUT
# expand of 2-bit packed codes, and tmac_g8 (f32 activation table over transposed uint8 packed index; prior winner).
# int8_scratch reuses one preallocated f32 buffer so the baseline is not allocator/page-fault bound.
# Reports per-layer ms (token ms / N) for N in NS, resident bytes, and max rel error vs f64 on layer 0.
import numpy as np, time, json, sys
SEED = 202611121455
Hd, F, D, BS, KVH = 2560, 6912, 128, 32, 5
NS = (1, 2, 4, 8, 12)
REP, WARM = 5, 1
rng = np.random.default_rng(SEED)
LUT = np.array([[((b >> (2 * k)) & 3) for k in range(4)] for b in range(256)], dtype=np.int8) - 1
LUT[LUT == 2] = 0
LUTF_T = LUT.astype(np.float32).T.copy()
def pack(T):
    c = (T + 1).astype(np.uint8).reshape(T.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
SI8 = np.empty((BS, F), np.int8); SF = np.empty((BS, F), np.float32)
def blk_lut(P, s, x, y):
    q = P.shape[1]; i = q * 4
    for r in range(0, P.shape[0], BS):
        n = min(BS, P.shape[0] - r)
        b8 = SI8[:n, :i].reshape(n, q, 4); np.take(LUT, P[r:r + n], axis=0, out=b8)
        bf = SF[:n, :i]; np.copyto(bf, SI8[:n, :i], casting='unsafe')
        np.matmul(bf, x, out=y[r:r + n])
    y *= s
G = 8
TS_IDX = np.empty((G, F), np.uint16); TS_F = np.empty((G, F), np.float32); TS_ACC = np.empty(F, np.float32)
OFFS = (np.arange(G, dtype=np.uint16) * 256)[:, None]
def tmac(PT, s, x, y):
    q, o = PT.shape; Tf = (x.reshape(q, 4) @ LUTF_T).ravel(); y[:] = 0
    for c in range(0, q, G):
        g = min(G, q - c)
        idx = TS_IDX[:g, :o]; np.add(PT[c:c + g], OFFS[:g], out=idx, dtype=np.uint16)
        v = TS_F[:g, :o]; np.take(Tf[c * 256:(c + g) * 256], idx, out=v)
        np.add.reduce(v, axis=0, out=TS_ACC[:o]); y += TS_ACC[:o]
    y *= s
def i8(W8, s, x, y): np.matmul(W8.astype(np.float32), x, out=y); y *= s  # fresh f32 temp each call (prior notes' baseline)
WF = np.empty(F * Hd, np.float32)
def i8s(W8, s, x, y):  # int8-resident with one reused f32 scratch: no per-call allocation / page faults
    o, i = W8.shape; w = WF[:o * i].reshape(o, i); np.copyto(w, W8, casting='unsafe'); np.matmul(w, x, out=y); y *= s
shapes = {"q": (Hd, Hd), "k": (KVH * D, Hd), "v": (KVH * D, Hd), "o": (Hd, Hd), "gate": (F, Hd), "up": (F, Hd), "down": (Hd, F)}
NMAX = max(NS)
L8, LP, LPT, LS = [], [], [], []
for l in range(NMAX):
    a, b, c, d = {}, {}, {}, {}
    for nm, (o, i) in shapes.items():
        T8 = rng.integers(-1, 2, size=(o, i), dtype=np.int8); P = pack(T8)
        a[nm] = T8; b[nm] = P; c[nm] = np.ascontiguousarray(P.T); d[nm] = np.float32(rng.uniform(0.01, 0.05))
    L8.append(a); LP.append(b); LPT.append(c); LS.append(d)
x = rng.standard_normal(Hd).astype(np.float32); xf = rng.standard_normal(F).astype(np.float32)
Y = {nm: np.empty(o, np.float32) for nm, (o, i) in shapes.items()}
forms = {"int8": (i8, L8), "int8_scratch": (i8s, L8), "blk32_lut": (blk_lut, LP), "tmac_g8": (tmac, LPT)}
def token(f, W, n):
    for l in range(n):
        for nm, (o, i) in shapes.items():
            f(W[l][nm], LS[l][nm], xf if i == F else x, Y[nm])
out = {"seed": SEED, "numpy": np.__version__, "NS": NS, "per_layer_ms": {}, "layer_bytes": {}, "max_rel_err": {}}
for form, (f, W) in forms.items():
    out["layer_bytes"][form] = int(sum(W[0][nm].nbytes for nm in shapes))
    e = 0.0
    for nm, (o, i) in shapes.items():
        v = xf if i == F else x
        ref = (L8[0][nm].astype(np.float64) @ v.astype(np.float64)) * float(LS[0][nm])
        f(W[0][nm], LS[0][nm], v, Y[nm]); e = max(e, float(np.max(np.abs(Y[nm] - ref)) / np.max(np.abs(ref))))
    out["max_rel_err"][form] = e
for n in NS:  # interleave forms per N so drift hits all forms alike
    for form, (f, W) in forms.items():
        for _ in range(WARM): token(f, W, n)
        ts = []
        for _ in range(REP):
            t0 = time.perf_counter(); token(f, W, n); ts.append((time.perf_counter() - t0) * 1e3 / n)
        out["per_layer_ms"].setdefault(form, {})[str(n)] = [float(np.median(ts)), float(min(ts)), float(max(ts))]
    r = {k: out["per_layer_ms"][k][str(n)][0] for k in forms}
    print(f"N={n:2d} stack int8 {n*out['layer_bytes']['int8']/2**20:6.1f} MiB packed {n*out['layer_bytes']['tmac_g8']/2**20:6.1f} MiB | " +
          " ".join(f"{k} {r[k]:7.2f}" for k in r) + f" ms/layer | tmac vs int8 {r['int8']/r['tmac_g8']:.2f}x vs int8_scratch {r['int8_scratch']/r['tmac_g8']:.2f}x lut vs int8_scratch {r['int8_scratch']/r['blk32_lut']:.2f}x", flush=True)
print("bytes", out["layer_bytes"], "err", out["max_rel_err"])
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Per-layer ms, median run1 / run2 / run3, with min–max per run. Speed column is `int8_scratch` median ÷ form median (above 1 means faster than the fair int8 baseline). Resident bytes per layer: int8 69,468,160 (66.25 MiB), packed 17,367,040 (16.56 MiB), exactly 4.00x less. Max rel err: int8 / int8_scratch / blk32_lut 3.5e-07, tmac_g8 4.8e-07.

| N layers | stack MiB | form | ms/layer r1 / r2 / r3 | min–max r1 / r2 / r3 | speed vs int8_scratch |
|---|---|---|---|---|---|
| 1 | 66.2 | int8 | 215.73 / 166.25 / 108.44 | 194.8–372.4 / 163.9–197.2 / 102.9–127.1 | 0.19x / 0.16x / 0.24x |
| 1 | 66.2 | int8_scratch | 42.03 / 26.75 / 26.28 | 34.8–44.6 / 23.0–31.6 / 22.8–34.2 | 1.00x / 1.00x / 1.00x |
| 1 | 16.6 | blk32_lut | 34.62 / 28.80 / 28.20 | 30.4–35.6 / 27.1–29.1 / 27.6–33.2 | 1.21x / 0.93x / 0.93x |
| 1 | 16.6 | tmac_g8 | 26.43 / 22.60 / 31.87 | 25.7–27.0 / 22.0–24.7 / 24.8–33.0 | 1.59x / 1.18x / 0.82x |
| 2 | 132.5 | int8 | 181.79 / 172.90 / 104.98 | 175.4–260.9 / 160.7–196.3 / 93.4–124.8 | 0.20x / 0.13x / 0.25x |
| 2 | 132.5 | int8_scratch | 36.02 / 22.57 / 26.36 | 30.5–41.0 / 21.4–25.7 / 22.2–32.2 | 1.00x / 1.00x / 1.00x |
| 2 | 33.1 | blk32_lut | 30.91 / 28.29 / 28.12 | 30.3–32.1 / 28.0–30.9 / 28.0–33.7 | 1.17x / 0.80x / 0.94x |
| 2 | 33.1 | tmac_g8 | 25.29 / 23.66 / 23.86 | 25.1–26.2 / 23.3–27.0 / 23.3–24.3 | 1.42x / 0.95x / 1.10x |
| 4 | 265.0 | int8 | 244.31 / 168.18 / 104.94 | 225.5–284.9 / 152.4–198.6 / 92.2–114.1 | 0.21x / 0.14x / 0.23x |
| 4 | 265.0 | int8_scratch | 51.55 / 22.96 / 24.08 | 41.0–74.9 / 22.6–25.3 / 22.1–27.0 | 1.00x / 1.00x / 1.00x |
| 4 | 66.2 | blk32_lut | 31.46 / 28.23 / 28.25 | 30.3–34.8 / 28.0–29.0 / 28.1–30.2 | 1.64x / 0.81x / 0.85x |
| 4 | 66.2 | tmac_g8 | 25.58 / 24.71 / 26.35 | 25.1–26.1 / 23.6–24.9 / 24.4–27.3 | 2.01x / 0.93x / 0.91x |
| 8 | 530.0 | int8 | 123.30 / 147.73 / 90.19 | 120.7–135.4 / 101.2–155.5 / 86.9–99.8 | 0.19x / 0.19x / 0.29x |
| 8 | 530.0 | int8_scratch | 23.85 / 28.32 / 26.02 | 23.1–24.7 / 24.8–41.1 / 24.3–27.3 | 1.00x / 1.00x / 1.00x |
| 8 | 132.5 | blk32_lut | 28.03 / 30.23 / 32.45 | 28.0–28.1 / 29.7–30.4 / 31.4–33.6 | 0.85x / 0.94x / 0.80x |
| 8 | 132.5 | tmac_g8 | 23.38 / 24.93 / 26.45 | 23.3–23.6 / 24.3–26.1 / 24.9–27.6 | 1.02x / 1.14x / 0.98x |
| 12 | 795.0 | int8 | 112.36 / 34.11 / 61.09 | 105.7–129.6 / 32.0–34.6 / 55.6–64.9 | 0.22x / 0.72x / 0.40x |
| 12 | 795.0 | int8_scratch | 24.18 / 24.51 / 24.57 | 23.5–26.6 / 24.1–26.3 / 24.4–24.7 | 1.00x / 1.00x / 1.00x |
| 12 | 198.7 | blk32_lut | 28.23 / 28.29 / 28.26 | 28.0–28.5 / 28.2–28.8 / 28.1–28.8 | 0.86x / 0.87x / 0.87x |
| 12 | 198.7 | tmac_g8 | 25.06 / 23.89 / 24.69 | 23.9–25.1 / 23.7–24.5 / 24.0–25.3 | 0.97x / 1.03x / 0.99x |

Three things stand out. First, no cold-weight penalty appears: `int8_scratch` per-layer time at N = 8 and 12 (795 MiB, well past the 320 MiB L3) is 23.9–28.3 ms, no slower than at N = 1–4 (22.6–26.8 ms on the calm runs 2 and 3). At B=1 in numpy both paths are compute- and gather-bound at roughly 2.5–3 GB/s of weight traffic, far below DRAM bandwidth, so streaming int8 from DRAM costs nothing extra. Second, `tmac_g8` holds 22.6–26.5 ms per layer at every N and lands at 0.91–1.14x of `int8_scratch` across all calm cells (the only ≥1.5x cells, run1 N=1 and N=4, come from `int8_scratch` contention spikes of 42 and 51.6 ms with min–max reaching 74.9 ms, not from cold weights). `blk32_lut` is 0.80–0.94x of it on calm runs. Third, the fresh-temp `int8` baseline from earlier notes is allocator/page-fault bound: it ran 34–244 ms per layer, bimodal and drifting, 4–9x slower than the same bytes expanded into a reused buffer. The earlier "packed is ~1.2–1.3x of int8-expand" figures were measured against that inflated baseline, so the packed speed edge in numpy is roughly zero once the baseline reuses scratch; the exact 4.00x memory win stands.

## Verdict

**Not viable on speed.** Cold weights across a 1–12 layer stack do not open the gap: `tmac_g8` is 0.91–1.14x of a scratch-reusing int8 GEMV at every N on calm runs, far from 1.5x, while the 4.00x resident-memory win (66.25 vs 16.56 MiB per layer, 795 vs 199 MiB for 12 layers) is exact and unchanged, with error ≤4.8e-07. The ternary case in this stack is a memory case (more layers fit in RAM), not a numpy speed case. Future benches should use `int8_scratch`, not `W8.astype(f32)` per call, as the int8 baseline.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-12T21:12:00Z (~14:12 MT) wrote pair_A/pair_B **no gap** ("14:05 11/12: JuniorLLM 68cd205 no-item vault; JuniorHome 548c111 ledger only; roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). JuniorLLM tip at clone time: [68cd205](https://github.com/cloudcover95/JuniorLLM/commit/68cd205) (this lane's 10:55 note); no Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours.

## Next question

The prior packed-activation results ([[2026-11-08_0555_output-tiled-mask-sign-packed-gemv-b1-b8-crossover-vs-int8-expand]]: 8.70x/4.20x/2.30x/1.20x at B=1/2/4/8) were measured against the fresh-temp int8-expand baseline shown here to be allocator-bound. Re-run that tiled mask/sign packed×packed GEMV at B=1–8 against `int8_scratch`: does any batch size still clear 1.5x on speed?

close: 59f61b695be55dc4 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-12_1055)
