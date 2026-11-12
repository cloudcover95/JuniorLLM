---
local: true
date: 2026-11-12
wave: "2026-11-12 02:55 (ran ~02:50–02:51 MT; Chief last_run 2026-11-12T09:06:00Z ~02:06 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, w158, t-mac, activation-lut, pre-offset, numpy, not-viable]
---

# Pre-offsetting the packed index to uint16 makes the activation LUT only 1.06–1.26x faster than T-MAC G=8 while halving the memory win, so it is not worth it

## Question

The prior note [[2026-11-11_1855_tmac-activation-lut-byte-index-packed-w158-layer-b1-vs-blk32-lut-int8]] found the T-MAC-style activation LUT at G=8 (`tmac_g8`) is the best packed B=1 path (~1.3x of `blk32_lut`) and that a separate per-call uint16 index-add pass costs it time. If the index `PT + 256·(row mod G)` is stored once at load as uint16 (2.00x of int8 bytes instead of 4.00x), does the activation LUT reach 1.5x of `blk32_lut`, and is a 2x memory win at that speed a better trade than the 4x memory win of `tmac_g8`? A second variant, `take_along_axis` on the uint8 layout, tests whether the add can be dropped while keeping 4x. Quant Chief `last_tips.json` `last_run` 2026-11-12T09:06:00Z (~02:06 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same layer as the prior four notes: B=1, hidden 2560, FFN 6912 (gate, up, down), 20 query heads × 128, KV heads 20 (MHA) and 5 (GQA), assumed shapes; absmean ternary weights, one f32 scale per matrix, 2-bit codes four per byte; layer time is the sum of the seven GEMVs, single-threaded Python loops.

- `int8`: int8-resident cast + GEMV. `blk32_lut`: 32-row weight-LUT expand. `tmac_g8`: prior winner (uint8 `PT`, per-call index add, take, reduce).
- `pre16_gG`: index stored at load as uint16 `PT + 256·(row mod G)`; per call only the activation-table build, the `take` and the reduce. G = 8, 16, 32.
- `pre32_g8`: full-table int32 index `PT + 256·row` (same bytes as int8, control).
- `tal_gG`: `np.take_along_axis(T[c:c+G], PT[c:c+G], axis=1)` on uint8 `PT`, no offset (keeps 4x). G = 8, 16.

numpy 2.5.4, 8-core shared box (load 0.9–2.6 during the runs), seed `202611120255`, 2 warm-ups, 7 repeats × 3 passes, median and min–max, three back-to-back runs. Error is max |y − ref| / max |ref| against an f64 GEMV on the same trits.

Commands, from `/workspace/bench0255_1112`: `python3 bench.py results_run1.json`, `python3 bench.py results_run2.json`, `python3 bench.py results_run3.json`.

```python
# Full bitnetd-shaped layer at B=1 (hidden 2560, FFN 6912, 20 q heads x 128, KV heads 20 and 5), single-threaded:
# pre-offset activation LUT (uint16 PT + 256*row stored at load, no per-call index add) vs T-MAC G=8 (uint8 PT + per-call add,
# prior winner) vs take_along_axis on uint8 PT (no offset, keeps 4x) vs blk32 weight-LUT expand vs int8-resident.
import numpy as np, time, json, sys
SEED = 202611120255
Hd, F, D, BS = 2560, 6912, 128, 32
REP, PASS, WARM = 7, 3, 2
rng = np.random.default_rng(SEED)
LUT = np.array([[((b >> (2 * k)) & 3) for k in range(4)] for b in range(256)], dtype=np.int8) - 1
LUT[LUT == 2] = 0
LUTF_T = LUT.astype(np.float32).T.copy()
def tern(o, i):
    W = rng.standard_normal((o, i)).astype(np.float32); s = np.abs(W).mean()
    return np.clip(np.rint(W / s), -1, 1).astype(np.int8), np.float32(s)
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
GMAX = 64
TS_IDX = np.empty((GMAX, F), np.uint16); TS_F = np.empty((GMAX, F), np.float32); TS_ACC = np.empty(F, np.float32)
OFFS = (np.arange(GMAX, dtype=np.uint16) * 256)[:, None]
def make_tmac(G):  # prior winner: uint8 PT, per-call uint16 index add
    def f(PT, s, x, y):
        q, o = PT.shape; Tf = (x.reshape(q, 4) @ LUTF_T).ravel(); y[:] = 0
        for c in range(0, q, G):
            g = min(G, q - c)
            idx = TS_IDX[:g, :o]; np.add(PT[c:c + g], OFFS[:g], out=idx, dtype=np.uint16)
            v = TS_F[:g, :o]; np.take(Tf[c * 256:(c + g) * 256], idx, out=v)
            np.add.reduce(v, axis=0, out=TS_ACC[:o]); y += TS_ACC[:o]
        y *= s
    return f
def make_pre(G):  # pre-offset: PO = PT + 256*(row mod G) as uint16, computed once at load (resident 2x of uint8 PT)
    def f(PO, s, x, y):
        q, o = PO.shape; Tf = (x.reshape(q, 4) @ LUTF_T).ravel(); y[:] = 0
        for c in range(0, q, G):
            g = min(G, q - c)
            v = TS_F[:g, :o]; np.take(Tf[c * 256:(c + g) * 256], PO[c:c + g], out=v)
            np.add.reduce(v, axis=0, out=TS_ACC[:o]); y += TS_ACC[:o]
        y *= s
    return f
def make_pre_flat(G):  # pre-offset over the whole table: PO = PT + 256*row as int32 indices into flat table (4x of uint8 PT = int8 bytes)
    def f(PO, s, x, y):
        q, o = PO.shape; Tf = (x.reshape(q, 4) @ LUTF_T).ravel(); y[:] = 0
        for c in range(0, q, G):
            g = min(G, q - c)
            v = TS_F[:g, :o]; np.take(Tf, PO[c:c + g], out=v)
            np.add.reduce(v, axis=0, out=TS_ACC[:o]); y += TS_ACC[:o]
        y *= s
    return f
def make_tal(G):  # take_along_axis on uint8 PT, no offset (keeps 4x)
    def f(PT, s, x, y):
        q, o = PT.shape; T = (x.reshape(q, 4) @ LUTF_T); y[:] = 0
        for c in range(0, q, G):
            g = min(G, q - c)
            v = np.take_along_axis(T[c:c + g], PT[c:c + g], axis=1)
            np.add.reduce(v, axis=0, out=TS_ACC[:o]); y += TS_ACC[:o]
        y *= s
    return f
def i8(W8, s, x, y): np.matmul(W8.astype(np.float32), x, out=y); y *= s
def bench(f, *a):
    for _ in range(WARM): f(*a)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for _ in range(PASS): f(*a)
        ts.append((time.perf_counter() - t0) / PASS * 1e3)
    return float(np.median(ts)), float(min(ts)), float(max(ts))
forms = {"int8": i8, "blk32_lut": blk_lut, "tmac_g8": make_tmac(8), "pre16_g8": make_pre(8), "pre16_g16": make_pre(16),
         "pre16_g32": make_pre(32), "pre32_g8": make_pre_flat(8), "tal_g8": make_tal(8), "tal_g16": make_tal(16)}
def wfor(k, T8, P, PT):
    if k == "int8": return T8
    if k == "blk32_lut": return P
    if k.startswith("pre16"):
        G = int(k.split("_g")[1]); return (PT.astype(np.uint16) + (np.arange(PT.shape[0]) % G * 256).astype(np.uint16)[:, None])
    if k.startswith("pre32"): return PT.astype(np.int32) + (np.arange(PT.shape[0], dtype=np.int32) * 256)[:, None]
    return PT
def scr(k, i, o):
    if k == "blk32_lut": return BS * F * 5
    if k == "int8": return o * i * 4  # transient f32 cast
    G = int(k.split("_g")[1])
    return (i // 4) * 256 * 4 + G * o * (6 if k.startswith("tmac") else 4) + o * 4
out = {"seed": SEED, "numpy": np.__version__, "layers": {}}
x = rng.standard_normal(Hd).astype(np.float32); xf = rng.standard_normal(F).astype(np.float32)
for kvh in (20, 5):
    shapes = {"q": (Hd, Hd), "k": (kvh * D, Hd), "v": (kvh * D, Hd), "o": (Hd, Hd), "gate": (F, Hd), "up": (F, Hd), "down": (Hd, F)}
    acc = {k: [0.0, 0.0, 0.0] for k in forms}; by = {k: 0 for k in forms}; err = {k: 0.0 for k in forms}; sc = {k: 0 for k in forms}
    for nm, (o, i) in shapes.items():
        T8, s = tern(o, i); P = pack(T8); PT = np.ascontiguousarray(P.T); v = xf if i == F else x
        ref = (T8.astype(np.float64) @ v.astype(np.float64)) * float(s); y = np.empty(o, np.float32)
        for k, f in forms.items():
            W = wfor(k, T8, P, PT)
            m, lo, hi = bench(f, W, s, v, y)
            acc[k] = [acc[k][0] + m, acc[k][1] + lo, acc[k][2] + hi]; by[k] += W.nbytes; sc[k] = max(sc[k], scr(k, i, o))
            f(W, s, v, y); err[k] = max(err[k], float(np.max(np.abs(y - ref)) / np.max(np.abs(ref))))
            del W
        del T8, P, PT
    out["layers"][str(kvh)] = dict(ms=acc, bytes=by, scratch=sc, max_rel_err=err)
    print("kvh", kvh, flush=True)
    for k in forms:
        print(f"  {k:10s} {acc[k][0]:8.3f} ms [{acc[k][1]:.3f}-{acc[k][2]:.3f}] {by[k]/2**20:7.2f} MiB scr {sc[k]/2**20:5.2f} err {err[k]:.1e} vs_int8 {acc['int8'][0]/acc[k][0]:.2f}x vs_lut {acc['blk32_lut'][0]/acc[k][0]:.2f}x vs_tmac {acc['tmac_g8'][0]/acc[k][0]:.2f}x mem_vs_int8 {by['int8']/by[k]:.2f}x", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Seven-matrix layer weight stage, ms median run1 / run2 / run3 (run3 min–max). The `int8` row is noisy under contention (GQA medians 50.0 / 36.7 / 32.5 ms), so the ratios against `blk32_lut` and `tmac_g8` are the ones to trust.

| KV heads | form | resident MiB | mem vs int8 | scratch MiB | ms run1 / run2 / run3 (run3 min–max) | vs blk32_lut (r1 / r2 / r3) | vs tmac_g8 (r1 / r2 / r3) | vs int8 (r1 / r2 / r3) | max rel err |
|---|---|---|---|---|---|---|---|---|---|
| 20 | int8 | 75.62 | 1.00x | 67.50 | 48.08 / 44.29 / 43.51 (40.45–51.48) | 0.70x / 0.76x / 0.82x | 0.54x / 0.60x / 0.58x | 1.00x / 1.00x / 1.00x | 3.7e-07 |
| 20 | blk32_lut | 18.91 | 4.00x | 1.05 | 33.59 / 33.61 / 35.70 (34.57–38.23) | 1.00x / 1.00x / 1.00x | 0.78x / 0.79x / 0.70x | 1.43x / 1.32x / 1.22x | 3.7e-07 |
| 20 | tmac_g8 | 18.91 | 4.00x | 1.81 | 26.15 / 26.40 / 25.15 (24.64–29.09) | 1.28x / 1.27x / 1.42x | 1.00x / 1.00x / 1.00x | 1.84x / 1.68x / 1.73x | 8.0e-07 |
| 20 | pre16_g8 | 37.81 | 2.00x | 1.78 | 22.71 / 22.95 / 21.59 (21.07–23.20) | 1.48x / 1.46x / 1.65x | 1.15x / 1.15x / 1.17x | 2.12x / 1.93x / 2.02x | 8.0e-07 |
| 20 | **pre16_g16** | 37.81 | 2.00x | 1.85 | 22.43 / 22.92 / 21.70 (21.33–23.15) | 1.50x / 1.47x / 1.64x | 1.17x / 1.15x / 1.16x | 2.14x / 1.93x / 2.00x | 2.9e-07 |
| 20 | pre16_g32 | 37.81 | 2.00x | 2.01 | 25.23 / 25.97 / 24.59 (23.75–26.26) | 1.33x / 1.29x / 1.45x | 1.04x / 1.02x / 1.02x | 1.91x / 1.71x / 1.77x | 2.3e-07 |
| 20 | pre32_g8 | 75.62 | 1.00x | 1.78 | 22.86 / 24.77 / 22.41 (21.88–23.57) | 1.47x / 1.36x / 1.59x | 1.14x / 1.07x / 1.12x | 2.10x / 1.79x / 1.94x | 8.0e-07 |
| 20 | tal_g8 | 18.91 | 4.00x | 1.78 | 63.90 / 65.47 / 61.43 (60.47–64.00) | 0.53x / 0.51x / 0.58x | 0.41x / 0.40x / 0.41x | 0.75x / 0.68x / 0.71x | 8.0e-07 |
| 20 | tal_g16 | 18.91 | 4.00x | 1.85 | 61.40 / 61.59 / 59.16 (58.21–63.27) | 0.55x / 0.55x / 0.60x | 0.43x / 0.43x / 0.43x | 0.78x / 0.72x / 0.74x | 2.9e-07 |
| 5 | int8 | 66.25 | 1.00x | 67.50 | 50.04 / 36.74 / 32.54 (30.34–38.96) | 0.66x / 0.82x / 0.95x | 0.51x / 0.64x / 0.73x | 1.00x / 1.00x / 1.00x | 4.1e-07 |
| 5 | blk32_lut | 16.56 | 4.00x | 1.05 | 33.13 / 30.02 / 30.91 (29.63–33.22) | 1.00x / 1.00x / 1.00x | 0.77x / 0.79x / 0.77x | 1.51x / 1.22x / 1.05x | 4.1e-07 |
| 5 | tmac_g8 | 16.56 | 4.00x | 1.81 | 25.58 / 23.58 / 23.87 (22.87–24.83) | 1.30x / 1.27x / 1.29x | 1.00x / 1.00x / 1.00x | 1.96x / 1.56x / 1.36x | 5.3e-07 |
| 5 | pre16_g8 | 33.12 | 2.00x | 1.78 | 20.58 / 22.30 / 20.49 (19.91–21.20) | 1.61x / 1.35x / 1.51x | 1.24x / 1.06x / 1.16x | 2.43x / 1.65x / 1.59x | 5.3e-07 |
| 5 | **pre16_g16** | 33.12 | 2.00x | 1.85 | 20.26 / 20.81 / 20.47 (19.77–21.06) | 1.63x / 1.44x / 1.51x | 1.26x / 1.13x / 1.17x | 2.47x / 1.77x / 1.59x | 3.0e-07 |
| 5 | pre16_g32 | 33.12 | 2.00x | 2.01 | 22.70 / 23.47 / 22.67 (21.84–23.55) | 1.46x / 1.28x / 1.36x | 1.13x / 1.00x / 1.05x | 2.20x / 1.57x / 1.44x | 2.7e-07 |
| 5 | pre32_g8 | 66.25 | 1.00x | 1.78 | 20.50 / 20.50 / 21.33 (20.04–22.95) | 1.62x / 1.46x / 1.45x | 1.25x / 1.15x / 1.12x | 2.44x / 1.79x / 1.53x | 5.3e-07 |
| 5 | tal_g8 | 16.56 | 4.00x | 1.78 | 60.04 / 56.60 / 57.30 (55.67–62.75) | 0.55x / 0.53x / 0.54x | 0.43x / 0.42x / 0.42x | 0.83x / 0.65x / 0.57x | 5.3e-07 |
| 5 | tal_g16 | 16.56 | 4.00x | 1.85 | 52.60 / 53.32 / 54.94 (52.57–63.05) | 0.63x / 0.56x / 0.56x | 0.49x / 0.44x / 0.43x | 0.95x / 0.69x / 0.59x | 3.0e-07 |

Scratch for `int8` is the transient f32 cast of the largest matrix. Removing the index add saves ~3.5 ms per layer (about 15%): `pre16_g8`/`pre16_g16` run 20.3–23.0 ms against 23.6–26.4 ms for `tmac_g8`. Against `blk32_lut` that is 1.44–1.65x, which crosses 1.5x in four of six run×shape cells but not in all three runs for either shape. The full int32 index (`pre32_g8`) is no faster than uint16, so the gain is the removed pass, not narrower indices. `take_along_axis` on uint8 keeps 4x but is 0.40–0.49x of `tmac_g8`, because numpy builds a broadcast fancy index internally on every call. G=32 already loses ground, as before, once the G×Out scratch leaves L2. Error stays at f32 noise (≤8.0e-07).

## Verdict

**Not viable.** The pre-offset index buys only 1.06–1.26x over the current best packed path (`tmac_g8`) and gives up half its memory win (2.00x vs 4.00x of int8 resident). Against the older `blk32_lut` it sits right at the bar (1.44–1.65x) and does not clear 1.5x in every run. A 2x memory win at about 1.15x more speed is a worse trade than keeping `tmac_g8` at 4x memory, so keep the uint8 transposed layout with the per-call add. `take_along_axis` is not a way to drop the add in numpy.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-12T09:06:00Z (~02:06 MT) wrote pair_A/pair_B **no gap** ("02:05 11/12: JuniorHome 23568df (LLC, 07:17Z) touched web3node GGUF path, left to LLC; roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). JuniorLLM tip at clone time: [4f9759c](https://github.com/cloudcover95/JuniorLLM/commit/4f9759c) (`rails/freshpath`, cloudcover95, 02:49 MT, not a vault path); no Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours.

## Next question

The activation table is f32 (4 bytes per entry, 1.7 MiB for the down matrix), so the `take` moves 4 bytes per packed byte. T-MAC quantizes activations to int8, so table entries fit in int16. Does an int16 activation table (x quantized per token to A8, entries summed in int16, reduce in int32) at G=8 on the uint8 layout beat `tmac_g8` by at least 1.5x at B=1 while keeping 4x memory, and what does the A8 activation quantization cost in error against the f32 path?

close: bce5eef9a3a19d2d (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-11_1855)
