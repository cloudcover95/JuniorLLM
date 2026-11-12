---
local: true
date: 2026-11-12
wave: "2026-11-12 05:55 (ran ~05:49–05:50 MT; Chief last_run 2026-11-12T12:14:00Z ~05:14 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, w158, t-mac, activation-lut, a8, int16, numpy, not-viable]
---

# An A8 int16 activation table makes the T-MAC LUT no faster (0.89–1.00x of f32 tmac_g8) and adds ~1% output error, so it is not worth it in numpy

## Question

The prior note [[2026-11-12_0255_pre-offset-uint16-activation-lut-packed-w158-layer-b1-vs-tmac-g8-blk32]] kept `tmac_g8` (f32 activation table, uint8 transposed packed index, per-call uint16 index add) as the best packed B=1 path at 4.00x less resident memory than int8. Its table is f32, so each `take` moves 4 bytes per packed byte. If the activation is quantized per token to A8 (absmax, int8) the table entries are sums of four trit×int8 products (|v| ≤ 508), which fit int16. Does an int16 table, with the G-row reduce done in int16 (|sum| ≤ G·508, safe for G ≤ 64) and accumulation in int32, beat `tmac_g8` by at least 1.5x at B=1 while keeping 4x memory, and what does the A8 quantization cost in error against the f32 path? Quant Chief `last_tips.json` `last_run` 2026-11-12T12:14:00Z (~05:14 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same layer as the prior five notes: B=1, hidden 2560, FFN 6912 (gate, up, down), 20 query heads × 128, KV heads 20 (MHA) and 5 (GQA), assumed shapes; absmean ternary weights, one f32 scale per matrix, 2-bit codes four per byte; layer time is the sum of the seven GEMVs, single-threaded Python loops.

- `int8`: int8-resident cast + GEMV. `blk32_lut`: 32-row weight-LUT expand. `tmac_g8`: prior winner (f32 table).
- `a8i16_gG`: x → int8 by per-token absmax, int16 table built with an int16 matmul against the 256×4 trit LUT, `take` into int16 scratch, reduce over G rows in int16, accumulate int32, one f32 rescale (`s · s_x`). G = 8, 16, 32, 64.
- `a8i32r_g8`: same, but the G-row reduce uses `dtype=int32` (control for numpy's mixed-type reduce cost).

numpy 2.5.4, 8-core shared box (load 0.9–2.1 during the runs), seed `202611120555`, 2 warm-ups, 7 repeats × 3 passes, median and min–max, three back-to-back runs. Error is max |y − ref| / max |ref| against an f64 GEMV on the same trits with the unquantized f32 activation, so the A8 forms carry their activation-quantization error and the f32 forms show only rounding.

Commands, from `/workspace/bench0555_1112`: `python3 bench.py results_run1.json`, `python3 bench.py results_run2.json`, `python3 bench.py results_run3.json`. The script is the prior note's harness with the A8 forms added; the unused `make_pre`/`make_tal` helpers and their `wfor` branches are carried over and never called.

```python
# Full bitnetd-shaped layer at B=1 (hidden 2560, FFN 6912, 20 q heads x 128, KV heads 20 and 5), single-threaded:
# T-MAC activation LUT with A8 activations and an int16 table (entries = sum of 4 trits * int8 x, |v|<=508), take into int16,
# reduce over G in int16 (|sum|<=G*508 fits for G<=64) or int32, accumulate int32, one f32 rescale, vs f32-table tmac_g8 (prior winner),
# blk32 weight-LUT expand and int8-resident. uint8 transposed PT layout (4x memory) for all LUT forms.
import numpy as np, time, json, sys
SEED = 202611120555
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
LUTI_T = LUT.astype(np.int16).T.copy()
TS_I = np.empty((GMAX, F), np.int16); TS_R16 = np.empty(F, np.int16); TS_R32 = np.empty(F, np.int32); YI = np.empty(F, np.int32)
def make_a8(G, red16):  # A8 per-token absmax quant, int16 activation table, uint8 PT + per-call uint16 index add
    def f(PT, s, x, y):
        q, o = PT.shape; sx = np.float32(np.abs(x).max() / 127.0)
        xq = np.clip(np.rint(x / sx), -127, 127).astype(np.int16)
        Ti = (xq.reshape(q, 4) @ LUTI_T).ravel(); acc = YI[:o]; acc[:] = 0
        for c in range(0, q, G):
            g = min(G, q - c)
            idx = TS_IDX[:g, :o]; np.add(PT[c:c + g], OFFS[:g], out=idx, dtype=np.uint16)
            v = TS_I[:g, :o]; np.take(Ti[c * 256:(c + g) * 256], idx, out=v)
            if red16:
                np.add.reduce(v, axis=0, out=TS_R16[:o]); acc += TS_R16[:o]
            else:
                np.add.reduce(v, axis=0, out=TS_R32[:o], dtype=np.int32); acc += TS_R32[:o]
        np.multiply(acc, s * sx, out=y, casting='unsafe')
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
forms = {"int8": i8, "blk32_lut": blk_lut, "tmac_g8": make_tmac(8), "a8i16_g8": make_a8(8, True), "a8i16_g16": make_a8(16, True),
         "a8i16_g32": make_a8(32, True), "a8i16_g64": make_a8(64, True), "a8i32r_g8": make_a8(8, False)}
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
    if k.startswith("a8"): return (i // 4) * 256 * 2 + G * o * 4 + o * 8
    return (i // 4) * 256 * 4 + G * o * 6 + o * 4
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

Seven-matrix layer weight stage, ms median run1 / run2 / run3 (run3 min–max). The `int8` row is noisy under contention (MHA medians 55.6 / 44.2 / 34.1 ms), so the ratios against `tmac_g8` and `blk32_lut` are the ones to trust.

| KV heads | form | resident MiB | mem vs int8 | scratch MiB | ms run1 / run2 / run3 (run3 min–max) | vs blk32_lut (r1 / r2 / r3) | vs tmac_g8 (r1 / r2 / r3) | vs int8 (r1 / r2 / r3) | max rel err |
|---|---|---|---|---|---|---|---|---|---|
| 20 | int8 | 75.62 | 1.00x | 67.50 | 55.61 / 44.18 / 34.11 (32.16–43.34) | 0.63x / 0.79x / 1.07x | 0.50x / 0.64x / 0.83x | 1.00x / 1.00x / 1.00x | 4.5e-07 |
| 20 | blk32_lut | 18.91 | 4.00x | 1.05 | 35.22 / 35.07 / 36.63 (34.99–39.37) | 1.00x / 1.00x / 1.00x | 0.79x / 0.80x / 0.77x | 1.58x / 1.26x / 0.93x | 4.5e-07 |
| 20 | tmac_g8 | 18.91 | 4.00x | 1.81 | 27.95 / 28.12 / 28.38 (27.27–29.52) | 1.26x / 1.25x / 1.29x | 1.00x / 1.00x / 1.00x | 1.99x / 1.57x / 1.20x | 4.6e-07 |
| 20 | a8i16_g8 | 18.91 | 4.00x | 0.94 | 29.70 / 29.75 / 29.96 (29.03–31.64) | 1.19x / 1.18x / 1.22x | 0.94x / 0.95x / 0.95x | 1.87x / 1.49x / 1.14x | 1.1e-02 |
| 20 | **a8i16_g16** | 18.91 | 4.00x | 1.02 | 28.26 / 29.26 / 28.36 (27.63–30.41) | 1.25x / 1.20x / 1.29x | 0.99x / 0.96x / 1.00x | 1.97x / 1.51x / 1.20x | 1.1e-02 |
| 20 | a8i16_g32 | 18.91 | 4.00x | 1.21 | 30.10 / 29.89 / 30.14 (28.78–32.16) | 1.17x / 1.17x / 1.22x | 0.93x / 0.94x / 0.94x | 1.85x / 1.48x / 1.13x | 1.1e-02 |
| 20 | a8i16_g64 | 18.91 | 4.00x | 2.05 | 33.18 / 32.94 / 31.87 (31.08–34.81) | 1.06x / 1.06x / 1.15x | 0.84x / 0.85x / 0.89x | 1.68x / 1.34x / 1.07x | 1.1e-02 |
| 20 | a8i32r_g8 | 18.91 | 4.00x | 0.94 | 31.74 / 33.61 / 31.26 (30.26–32.86) | 1.11x / 1.04x / 1.17x | 0.88x / 0.84x / 0.91x | 1.75x / 1.31x / 1.09x | 1.1e-02 |
| 5 | int8 | 66.25 | 1.00x | 67.50 | 55.16 / 29.44 / 32.52 (30.75–37.55) | 0.57x / 1.07x / 0.93x | 0.44x / 0.84x / 0.77x | 1.00x / 1.00x / 1.00x | 4.2e-07 |
| 5 | blk32_lut | 16.56 | 4.00x | 1.05 | 31.27 / 31.36 / 30.12 (28.99–31.66) | 1.00x / 1.00x / 1.00x | 0.78x / 0.79x / 0.83x | 1.76x / 0.94x / 1.08x | 4.2e-07 |
| 5 | tmac_g8 | 16.56 | 4.00x | 1.81 | 24.25 / 24.82 / 24.97 (23.58–27.26) | 1.29x / 1.26x / 1.21x | 1.00x / 1.00x / 1.00x | 2.27x / 1.19x / 1.30x | 4.6e-07 |
| 5 | a8i16_g8 | 16.56 | 4.00x | 0.94 | 27.31 / 26.76 / 26.88 (26.11–28.61) | 1.15x / 1.17x / 1.12x | 0.89x / 0.93x / 0.93x | 2.02x / 1.10x / 1.21x | 1.1e-02 |
| 5 | **a8i16_g16** | 16.56 | 4.00x | 1.02 | 26.51 / 26.28 / 25.68 (24.64–27.21) | 1.18x / 1.19x / 1.17x | 0.91x / 0.94x / 0.97x | 2.08x / 1.12x / 1.27x | 1.1e-02 |
| 5 | a8i16_g32 | 16.56 | 4.00x | 1.21 | 27.59 / 27.39 / 27.16 (26.10–28.26) | 1.13x / 1.15x / 1.11x | 0.88x / 0.91x / 0.92x | 2.00x / 1.07x / 1.20x | 1.1e-02 |
| 5 | a8i16_g64 | 16.56 | 4.00x | 2.05 | 29.72 / 29.15 / 28.75 (27.86–29.58) | 1.05x / 1.08x / 1.05x | 0.82x / 0.85x / 0.87x | 1.86x / 1.01x / 1.13x | 1.1e-02 |
| 5 | a8i32r_g8 | 16.56 | 4.00x | 0.94 | 29.35 / 28.96 / 28.35 (26.85–29.47) | 1.07x / 1.08x / 1.06x | 0.83x / 0.86x / 0.88x | 1.88x / 1.02x / 1.15x | 1.1e-02 |

The int16 table halves the table bytes (0.94 MiB scratch at G=8 against 1.81 MiB) but does not make the layer faster: the best A8 form, `a8i16_g16`, runs 25.7–29.3 ms against 24.3–28.4 ms for `tmac_g8` (0.91–1.00x), and G=8 is 0.89–0.95x. At B=1 the table is already L2-resident, so the `take` is bound by numpy's per-element gather overhead, not by bytes moved, and an int16 gather is no cheaper than an f32 one. The extra per-call work (absmax, round, clip, int16 cast, int32 accumulate and final rescale) eats whatever is saved. Reducing in int32 instead of int16 costs another 4–13% (mixed-type reduce). G=64 loses ground again (0.82–0.89x) as the G×Out scratch leaves L2. Resident memory is unchanged at 4.00x less than int8 for every LUT form. The A8 activation adds about 1.1e-02 max relative error against the f32 path (4.6e-07), which is a normal A8 cost but buys nothing here.

## Verdict

**Not viable.** An int16 A8 activation table is 0.89–1.00x of `tmac_g8` at the same 4.00x memory win and costs ~1.1% output error, far from the 1.5x bar. Keep the f32 table: in numpy the gather count, not table width, sets the cost. The int16 table only pays where the lookup is a register shuffle (T-MAC's `vpshufb`/`tbl` kernels), which numpy cannot express.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-12T12:14:00Z (~05:14 MT) wrote pair_A/pair_B **no gap** ("05:05 11/12: no LLC commit in 6h; JuniorHome 2888b30 ledger only; roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). JuniorLLM tip at clone time: [739c6e1](https://github.com/cloudcover95/JuniorLLM/commit/739c6e1) (this lane's 02:55 note); no Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours.

## Next question

The packed path now ends at `tmac_g8` for the GEMVs, but the per-matrix Python loop issues ~q/G numpy calls per matrix. Does fusing q, k and v (and gate with up) into one stacked transposed index so they share one activation table and one G-loop cut the B=1 layer time by at least 1.5x against seven separate `tmac_g8` calls, at the same 4x memory and unchanged error?

close: 3dfd4aca9981b579 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-12_0255)
