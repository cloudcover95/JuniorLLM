---
local: true
date: 2026-11-12
wave: "2026-11-12 10:55 (ran ~10:57–10:58 MT; Chief last_run 2026-11-12T17:12:00Z ~10:12 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorOS overlay hardened bitnetd on 127.0.0.1:8765"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnetd, decode, layer, w158, t-mac, activation-lut, fused-qkv, numpy, not-viable]
---

# Fusing q/k/v and gate/up into one stacked T-MAC index does not speed up the B=1 layer (0.95–0.99x of separate tmac_g8 on calm runs), so it is not worth it in numpy

## Question

The prior note [[2026-11-12_0555_a8-int16-activation-lut-packed-w158-layer-b1-vs-f32-tmac-g8]] kept `tmac_g8` (f32 activation table, uint8 transposed packed index) as the best packed B=1 path and asked: does stacking q, k and v (and gate with up), which share the same input, into one transposed index so they build one activation table and run one G-loop, cut the B=1 layer time by at least 1.5x against seven separate `tmac_g8` calls, at the same 4x memory and unchanged error? Quant Chief `last_tips.json` `last_run` 2026-11-12T17:12:00Z (~10:12 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same layer as the prior six notes: B=1, hidden 2560, FFN 6912 (gate, up, down), 20 query heads × 128, KV heads 20 (MHA) and 5 (GQA), assumed shapes; absmean ternary weights, one f32 scale per matrix, 2-bit codes four per byte; single-threaded Python loops.

- `int8`, `blk32_lut`, `tmac_g8`, `tmac_g16`: seven separate GEMVs, layer time = sum of the seven medians.
- `fused_gG` (G = 8, 16, 32): four calls per layer. q|k|v columns of the transposed index are concatenated into one `(640, o_q+o_k+o_v)` uint8 array and gate|up into one `(640, 13824)` array (built once at load, same bytes); o and down stay single. One activation table and one G-loop per group; per-matrix scales are applied as a per-row f32 vector after the reduce, and outputs are split back by row ranges for the error check.

numpy 2.5.4, 8-core shared box (load 0.3 before, 4.2 at the end of the runs), seed `202611121055`, 2 warm-ups, 7 repeats × 3 passes, median and min–max, three back-to-back runs. Error is max |y − ref| / max |ref| against an f64 GEMV on the same trits.

Commands, from `/workspace/bench1055_1112`: `python3 bench.py results_run1.json`, `python3 bench.py results_run2.json`, `python3 bench.py results_run3.json`.

```python
# Full bitnetd-shaped layer at B=1 (hidden 2560, FFN 6912, 20 q heads x 128, KV heads 20 and 5), single-threaded:
# fused T-MAC activation LUT: stack q|k|v and gate|up row-wise into one transposed uint8 packed index (PT columns concatenated)
# so each shared-input group builds ONE f32 activation table and runs ONE G-loop, vs seven separate tmac_g8 calls (prior winner),
# blk32 weight-LUT expand and int8-resident. Per-matrix f32 scales are applied after the fused GEMV by a per-row scale vector.
import numpy as np, time, json, sys
SEED = 202611121055
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
OMAX = 2 * F
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
TS_IDX = np.empty((GMAX, OMAX), np.uint16); TS_F = np.empty((GMAX, OMAX), np.float32); TS_ACC = np.empty(OMAX, np.float32)
OFFS = (np.arange(GMAX, dtype=np.uint16) * 256)[:, None]
def make_tmac(G):  # prior winner; s may be a scalar or a per-output-row scale vector (fused)
    def f(PT, s, x, y):
        q, o = PT.shape; Tf = (x.reshape(q, 4) @ LUTF_T).ravel(); y[:] = 0
        for c in range(0, q, G):
            g = min(G, q - c)
            idx = TS_IDX[:g, :o]; np.add(PT[c:c + g], OFFS[:g], out=idx, dtype=np.uint16)
            v = TS_F[:g, :o]; np.take(Tf[c * 256:(c + g) * 256], idx, out=v)
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
def lay(ms_list):  # sum medians/mins/maxes
    return [sum(m[0] for m in ms_list), sum(m[1] for m in ms_list), sum(m[2] for m in ms_list)]
out = {"seed": SEED, "numpy": np.__version__, "layers": {}}
x = rng.standard_normal(Hd).astype(np.float32); xf = rng.standard_normal(F).astype(np.float32)
for kvh in (20, 5):
    shapes = {"q": (Hd, Hd), "k": (kvh * D, Hd), "v": (kvh * D, Hd), "o": (Hd, Hd), "gate": (F, Hd), "up": (F, Hd), "down": (Hd, F)}
    M = {}
    for nm, (o, i) in shapes.items():
        T8, s = tern(o, i); P = pack(T8); PT = np.ascontiguousarray(P.T); M[nm] = (T8, s, P, PT)
    refs = {nm: (M[nm][0].astype(np.float64) @ (xf if shapes[nm][1] == F else x).astype(np.float64)) * float(M[nm][1]) for nm in shapes}
    groups = {"qkv": ["q", "k", "v"], "o": ["o"], "gu": ["gate", "up"], "down": ["down"]}
    res = {}; byt = {}; err = {}
    # separate forms: seven calls
    for form, f, wsel in (("int8", i8, 0), ("blk32_lut", blk_lut, 2), ("tmac_g8", make_tmac(8), 3), ("tmac_g16", make_tmac(16), 3)):
        parts = []; e = 0.0; b = 0
        for nm, (o, i) in shapes.items():
            W = M[nm][wsel]; s = M[nm][1]; v = xf if i == F else x; y = np.empty(o, np.float32)
            parts.append(bench(f, W, s, v, y)); f(W, s, v, y); b += W.nbytes
            e = max(e, float(np.max(np.abs(y - refs[nm])) / np.max(np.abs(refs[nm]))))
        res[form] = lay(parts); byt[form] = b; err[form] = e
    # fused forms: qkv and gate|up stacked along the output axis of PT
    for G in (8, 16, 32):
        form = f"fused_g{G}"; f = make_tmac(G); parts = []; e = 0.0; b = 0
        for gname, mem in groups.items():
            PTs = np.ascontiguousarray(np.concatenate([M[m][3] for m in mem], axis=1))
            sv = np.concatenate([np.full(shapes[m][0], M[m][1], np.float32) for m in mem])
            o = PTs.shape[1]; v = xf if shapes[mem[0]][1] == F else x; y = np.empty(o, np.float32)
            parts.append(bench(f, PTs, sv, v, y)); f(PTs, sv, v, y); b += PTs.nbytes
            r0 = 0
            for m in mem:
                n = shapes[m][0]; e = max(e, float(np.max(np.abs(y[r0:r0 + n] - refs[m])) / np.max(np.abs(refs[m])))); r0 += n
            del PTs
        res[form] = lay(parts); byt[form] = b; err[form] = e
    out["layers"][str(kvh)] = dict(ms=res, bytes=byt, max_rel_err=err)
    print("kvh", kvh, flush=True)
    for k in res:
        print(f"  {k:10s} {res[k][0]:8.3f} ms [{res[k][1]:.3f}-{res[k][2]:.3f}] {byt[k]/2**20:7.2f} MiB err {err[k]:.1e} vs_int8 {res['int8'][0]/res[k][0]:.2f}x vs_lut {res['blk32_lut'][0]/res[k][0]:.2f}x vs_tmac {res['tmac_g8'][0]/res[k][0]:.2f}x mem_vs_int8 {byt['int8']/byt[k]:.2f}x", flush=True)
    del M
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
```

## Results

Layer weight stage, ms median run1 / run2 / run3. Contention was heavy in parts of the runs: `int8` medians swung 53–223 ms (MHA), and `tmac_g8` MHA hit 37.9 ms in run2 and 49.4 ms in run3 against ~31 ms calm. Run1 and the GQA rows of all three runs are the calm comparisons; the run3 MHA ratio against `tmac_g8` (1.63x) is a `tmac_g8` contention outlier, not a fusion win (fused_g8 itself stayed at 30.3–32.1 ms throughout).

| KV heads | form | resident MiB | mem vs int8 | ms run1 / run2 / run3 | vs tmac_g8 (r1 / r2 / r3) | vs blk32_lut (r1 / r2 / r3) | max rel err |
|---|---|---|---|---|---|---|---|
| 20 | int8 | 75.62 | 1.00x | 108.04 / 223.10 / 53.47 | 0.29x / 0.17x / 0.92x | 0.37x / 0.20x / 0.98x | 3.7e-07 |
| 20 | blk32_lut | 18.91 | 4.00x | 40.03 / 43.95 / 52.36 | 0.77x / 0.86x / 0.94x | 1.00x | 3.7e-07 |
| 20 | tmac_g8 | 18.91 | 4.00x | 30.96 / 37.92 / 49.39 | 1.00x | 1.29x / 1.16x / 1.06x | 5.5e-07 |
| 20 | tmac_g16 | 18.91 | 4.00x | 30.62 / 36.37 / 39.51 | 1.01x / 1.04x / 1.25x | 1.31x / 1.21x / 1.33x | 4.6e-07 |
| 20 | **fused_g8** | 18.91 | 4.00x | 31.15 / 32.11 / 30.35 | 0.99x / 1.18x / 1.63x | 1.29x / 1.37x / 1.73x | 5.5e-07 |
| 20 | fused_g16 | 18.91 | 4.00x | 35.91 / 37.83 / 33.15 | 0.86x / 1.00x / 1.49x | 1.11x / 1.16x / 1.58x | 4.6e-07 |
| 20 | fused_g32 | 18.91 | 4.00x | 35.97 / 39.23 / 36.10 | 0.86x / 0.97x / 1.37x | 1.11x / 1.12x / 1.45x | 2.4e-07 |
| 5 | int8 | 66.25 | 1.00x | 65.78 / 95.93 / 34.14 | 0.42x / 0.28x / 0.80x | 0.53x / 0.37x / 1.02x | 3.6e-07 |
| 5 | blk32_lut | 16.56 | 4.00x | 34.67 / 35.28 / 34.89 | 0.79x / 0.76x / 0.78x | 1.00x | 3.6e-07 |
| 5 | tmac_g8 | 16.56 | 4.00x | 27.34 / 26.98 / 27.26 | 1.00x | 1.27x / 1.31x / 1.28x | 4.2e-07 |
| 5 | tmac_g16 | 16.56 | 4.00x | 28.53 / 27.07 / 27.68 | 0.96x / 1.00x / 0.98x | 1.22x / 1.30x / 1.26x | 3.0e-07 |
| 5 | **fused_g8** | 16.56 | 4.00x | 28.88 / 28.04 / 28.27 | 0.95x / 0.96x / 0.96x | 1.20x / 1.26x / 1.23x | 4.2e-07 |
| 5 | fused_g16 | 16.56 | 4.00x | 30.78 / 29.89 / 30.17 | 0.89x / 0.90x / 0.90x | 1.13x / 1.18x / 1.16x | 3.0e-07 |
| 5 | fused_g32 | 16.56 | 4.00x | 32.36 / 31.32 / 33.74 | 0.84x / 0.86x / 0.81x | 1.07x / 1.13x / 1.03x | 2.2e-07 |

Fusion saves table builds and loop iterations (3 tables and 240 iterations for q/k/v become 1 and 80 at G=8), but at B=1 the per-call overhead was already small: the gather and reduce of every packed byte dominates, and that work is identical fused or not. The wider fused rows (up to 13824 outputs for gate|up) push the G×Out f32 and uint16 scratch from ~0.4 MiB to ~1.2 MiB at G=8 and out of L2 at G≥16, which is why fused_g16/g32 lose 10–19% on calm runs. On calm runs fused_g8 is 0.95–0.99x of separate `tmac_g8`; its one real property is steadiness under load (30.3–32.1 ms MHA across all three runs while separate calls drifted to 49 ms), which is not a speed win. Memory is identical (4.00x less than int8) and error unchanged (≤5.5e-07).

## Verdict

**Not viable.** Stacking q/k/v and gate/up into one T-MAC index is 0.95–0.99x of seven separate `tmac_g8` calls on calm runs at the same 4.00x memory and error, far from 1.5x. Keep separate `tmac_g8` per matrix; in numpy the per-byte gather, not call count or table builds, sets the B=1 cost.

## Pair files this wave

- None. Quant Chief `last_tips.json` `last_run` 2026-11-12T17:12:00Z (~10:12 MT) wrote pair_A/pair_B **no gap** ("10:05 11/12: JuniorHome 423c68a re-advances landed ROADMAP #3 i2sd; JuniorLLM 3579160 no-item vault; roadmap blobs unchanged; both pairs no gap"; JuniorHome ROADMAP `4efb666` / JuniorLLM BETA_TO_OS `bac6923`). JuniorLLM tip at clone time: [3579160](https://github.com/cloudcover95/JuniorLLM/commit/3579160) (this lane's 05:55 note); no Trit Pack, Handshake or LLC commit touched `junior_osai/vault/findings/` in the last 6 hours.

## Next question

The packed numpy path has now plateaued at ~1.2–1.3x of int8-expand on speed (gather-bound) with the exact 4x memory win. Instead of another GEMV variant, does the 4x smaller resident weight set change end-to-end B=1 decode for a stack of N layers once N×int8 bytes exceed the box's last-level cache, i.e. at what layer count does `tmac_g8` on cold (non-cache-resident) weights beat cold `int8` by at least 1.5x?

close: 800e2ce415745039 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-12_0555)
