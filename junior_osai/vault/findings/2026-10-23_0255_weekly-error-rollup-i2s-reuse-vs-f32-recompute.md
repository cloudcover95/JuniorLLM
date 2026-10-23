---
local: true
date: 2026-10-23
wave: "2026-10-23 02:05"
roadmap_item: "JuniorHome docs/ROADMAP.md Automations: JuniorCloud chief weekly Sun/Mon 17:00 MDT (writes docs/WEEKLY_AUDIT.md), quant-error rollup section"
chief_assignment: "Pair A pushed chief weekly audit files this wave (the suggestion held for the 02:05 Chief); quant-chief/last_tips.json last_assignment still reads overlay for pair A and 16:45 remaining engines for pair B"
verdict: not-viable
tags: [ternary, osai, findings, automations, chief-weekly, weekly-audit, quant-error, winsor, i2s, lut, numpy, not-viable]
---

# Weekly audit quant-error rollup: reusing the daily I2_S trits plus an f32 gamma is 2.0-2.6x slower than just recomputing the ternarization in f32; not worth it

## Question

The 18:55 note asked whether the audit's quant-error section (relative MSE and worst |w - q| against the analytic bound) gets 1.5x cheaper when it reads the I2_S packed trits the daily run already wrote, plus one f32 gamma, instead of recomputing q = gamma * clamp(round(w / gamma), -1, 1) from w. Winsor's new weekly rollup (JuniorQuant `44496b9`) does that recompute in float64 for 7 days x 5 ports x 3 gammas (absmean, junior_drop, p95), so this run measures it on the weekly shape.

## Method

Synthetic weights, Student-t(4) x 0.02 in f32 (heavy tails, so p95 and junior_drop gammas differ), seed 255, 35 tensors (7 days x 5 ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM), 262,144 and 1,048,576 weights per tensor. Gammas and ||w||^2 are computed once (the daily run already has them). Paths, all producing relMSE and max |w - q| for every tensor and gamma:

- f64 recompute (Winsor's math, vectorized), the reference;
- f32 recompute (same math in f32, dot product for SSE);
- I2_S reuse, nibble LUT: bytes to 4 codes via a 256x4 LUT, then codes to values (two gathers);
- I2_S reuse, byte LUT: one gather from a 256x4 f32 table; SSE = ||w||^2 - 2g(q.w) + g^2 * nnz(q), max error from w - g*q;
- SSE-only variants of f32 recompute and byte LUT (no max |w - q|).

Single thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`), 9 repeats at 262k and 5 at 1M, median/min/max wall time. Every path's relMSE checked against f64; f32 and f64 ternarization compared trit by trit. python3 3.13.5, numpy 2.5.4, 8 cores, no network, nothing bound, no files read. Scripts and raw output on the box at `/workspace/tbench/r20261023_0255/`.

Commands:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 weekly_err_bench.py 262144 9
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 variant.py 262144 9
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 variant.py 1048576 5
```

## Results

Run 1, 262,144 weights x 35 tensors x 3 gammas (w = 36,700,160 bytes f32):

| path | median ms | min ms | max ms | vs f64 | max rel err (relMSE) | max abs err (max error) | extra stored bytes |
|---|---|---|---|---|---|---|---|
| f64 recompute | 166.5 | 164.8 | 179.7 | 1.00x | 0 | 0 | 0 |
| f32 recompute | 39.4 | 37.5 | 43.3 | 4.22x | 5.97e-07 | 5.96e-08 | 0 |
| I2_S reuse, nibble LUT | 169.0 | 166.3 | 185.5 | 0.99x | 6.29e-07 | 5.96e-08 | 6,881,700 |

Run 2, same shape, faster decode:

| path | median ms | min ms | max ms | vs f64 | vs f32 recompute | max rel err (relMSE) |
|---|---|---|---|---|---|---|
| f64 recompute | 178.0 | | | 1.00x | | 0 |
| f32 recompute | 44.1 | 43.6 | 47.7 | 4.04x | 1.00x | 5.97e-07 |
| I2_S reuse, byte LUT | 116.8 | 112.9 | 130.3 | 1.52x | 0.38x | 6.29e-07 |
| f32 recompute, SSE only | 35.9 | 34.5 | 39.9 | 4.95x | 1.00x | 5.97e-07 |
| I2_S byte LUT, SSE only | 94.6 | 91.7 | 99.7 | 1.88x | 0.38x | 6.08e-07 |

Run 3, 1,048,576 weights x 35 tensors x 3 gammas (w = 146.8 MB f32):

| path | median ms | min ms | max ms | vs f64 | vs f32 recompute | max rel err (relMSE) |
|---|---|---|---|---|---|---|
| f64 recompute | 630.2 | | | 1.00x | | 0 |
| f32 recompute | 231.5 | 229.5 | 235.1 | 2.72x | 1.00x | 2.68e-06 |
| I2_S reuse, byte LUT | 471.3 | 469.5 | 498.9 | 1.34x | 0.49x | 1.44e-06 |
| f32 recompute, SSE only | 184.9 | 182.8 | 199.3 | 3.41x | 1.00x | 2.68e-06 |
| I2_S byte LUT, SSE only | 358.5 | 355.3 | 368.4 | 1.76x | 0.52x | 1.44e-06 |

Trit flips between f32 and f64 ternarization at 262k: 0 of 27,525,120.

## Verdict

**not-viable.** Reading the packed trits back is 2.0-2.6x slower than recomputing the ternarization in f32, because the error section needs w anyway, and the LUT gather that turns 2-bit codes back into values costs more than one divide, one round and one clip. It only beats Winsor's float64 path (1.3-1.9x), and plain f32 already beats that by 2.7-5.0x with zero flipped trits. The packed state is 4x smaller than an int8 q (6.9 MB vs 27.5 MB per week at 262k), but the recompute stores no q at all, so there is no memory win either. At 1M weights the f32 dot-product SSE drifts to 2.7e-06 relative (1.4e-06 for the LUT path), just over a 1e-6 target; accumulate SSE in f64 if the weekly audit needs 1e-6. Recommendation for `44496b9`: keep the trits as the close-hash input only and compute the error section from w in f32.

## Links

- Previous note: [[2026-10-22_1855_audit-3day-u64-word-flags-vs-3byte-day-int8]]
- Pair A, 02:05 wave (chief weekly audit): Trit Pack cloudcover95/BitNet-mlx `5df39f9` scripts/chief_weekly_slot_cadence_i2s_key.py (weekly slot reconciliation, 7-day cadence JTR1/I2_S key); Winsor & Quant Error cloudcover95/JuniorQuant `44496b9` src/weekly_audit_quant_error_rollup.py (7-day per-port gamma drift, ternary error vs bound, Winsor rate 0.05, slot mismatch finding)
- Pair B, 02:05 wave (16:45 daily audit, remaining engines): Handshake & Ports cloudcover95/JuniorHome `c1b5826` ports/daily_audit_remaining_engines_port_map.py (commit seen in the local clone at 02:40 MT; push not checked from here). No Side-Commit file seen on the box at 02:55.

## Next question

Trit Pack's 7-day cadence key (`5df39f9`) packs one week of slot trits: is checking a whole year of weekly-slot cadence (52 weeks x slots) as one I2_S word scan at least 1.5x faster than parsing the slot lists from docs/AUTOMATION_AUDIT.md and docs/ROADMAP.md each run, or is the parse so small it never matters?

## Inline scripts

`weekly_err_bench.py`:

```python
# 2026-10-23 02:55 bench: chief weekly audit quant-error rollup (7 days x 5 ports x 3 gammas)
# A) f64 recompute q from w (Winsor 44496b9 math, vectorized)  B) f32 recompute
# C) reuse daily I2_S packed trits + f32 gamma: SSE via masked-sum identity, max|w-q| via 4-entry LUT decode
# python3 + numpy only, no network, nothing bound.
import sys, time, statistics as st, numpy as np
DAYS, PORTS, REPS = 7, 5, int(sys.argv[2]) if len(sys.argv) > 2 else 9
N = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
rng = np.random.default_rng(255)
W = (rng.standard_t(4, size=(DAYS*PORTS, N)) * 0.02).astype(np.float32)
def gammas(w):
    a = np.sort(np.abs(w.astype(np.float64)))
    k = len(a) - len(a)//16
    return np.array([a.mean(), a[:k].mean(), a[int(np.ceil(.95*len(a)))-1]])
G = np.array([gammas(w) for w in W])            # (35, 3) f64, computed once (daily run)
def tern(w, g):  # q in {-1,0,1}
    return np.clip(np.rint(w / g), -1, 1)
def pack(q):     # I2_S codes 0->0, +1->1, -1->2, MSB-first 4 per byte
    c = np.where(q == -1, 2, q).astype(np.uint8).reshape(-1, 4)
    return (c[:, 0] << 6) | (c[:, 1] << 4) | (c[:, 2] << 2) | c[:, 3]
P = [[pack(tern(W[t], np.float32(G[t, j]))) for j in range(3)] for t in range(len(W))]
W2 = (W.astype(np.float64)**2).sum(1)           # ||w||^2 per tensor, from the daily run
WB = np.abs(W)
LUTC = np.array([[(b >> s) & 3 for s in (6, 4, 2, 0)] for b in range(256)], np.uint8)
VAL = np.array([0., 1., -1., 0.], np.float32)
def run_f64():
    out = np.empty((len(W), 3, 2))
    for t in range(len(W)):
        w = W[t].astype(np.float64)
        for j in range(3):
            g = G[t, j]; e = w - g*tern(w, g)
            out[t, j] = ((e*e).sum() / W2[t], np.abs(e).max())
    return out
def run_f32():
    out = np.empty((len(W), 3, 2))
    for t in range(len(W)):
        w = W[t]
        for j in range(3):
            g = np.float32(G[t, j]); e = w - g*tern(w, g)
            out[t, j] = (np.dot(e, e) / W2[t], np.abs(e).max())
    return out
def run_i2s():
    out = np.empty((len(W), 3, 2))
    for t in range(len(W)):
        w = W[t]
        for j in range(3):
            g = np.float32(G[t, j]); c = LUTC[P[t][j]].reshape(-1)
            q = VAL[c]
            sqw = np.dot(q, w)                       # sum_pos w - sum_neg w
            nz = np.count_nonzero(c)
            sse = W2[t] - 2*float(g)*float(sqw) + float(g)**2*nz
            out[t, j] = (sse / W2[t], np.abs(w - g*q).max())
    return out
def tm(f):
    xs = []
    for _ in range(REPS):
        t0 = time.perf_counter(); o = f(); xs.append(time.perf_counter() - t0)
    return st.median(xs), min(xs), max(xs), o
ref = tm(run_f64); r32 = tm(run_f32); ri = tm(run_i2s)
nw = W.size * 3
state = {"f64 q (recompute, keeps w f32)": 0, "f32 q (recompute, keeps w f32)": 0,
         "i2s trits + f32 gamma": sum(p.nbytes for row in P for p in row) + G.size*4}
print(f"N/tensor={N} tensors={len(W)} gammas=3 reps={REPS} w_bytes={W.nbytes}")
print("path|median_ms|min_ms|max_ms|speed_vs_f64|max_rel_err_relMSE|max_abs_err_maxerr|extra_state_bytes|int8_q_bytes")
for name, r in (("f64 recompute", ref), ("f32 recompute", r32), ("i2s packed + LUT", ri)):
    d0 = np.abs(r[3][..., 0] - ref[3][..., 0]) / ref[3][..., 0]
    d1 = np.abs(r[3][..., 1] - ref[3][..., 1]).max()
    sb = state.get("i2s trits + f32 gamma") if name.startswith("i2s") else 0
    print(f"{name}|{r[0]*1e3:.1f}|{r[1]*1e3:.1f}|{r[2]*1e3:.1f}|{ref[0]/r[0]:.2f}x|{d0.max():.2e}|{d1:.2e}|{sb}|{nw}")
print("trit flips f32-vs-f64 ternarize:", sum(int((pack(tern(W[t], np.float32(G[t,j]))) != pack(tern(W[t].astype(np.float64), G[t,j]))).sum()) for t in range(len(W)) for j in range(3)))
```

`variant.py`:

```python
import sys, time, statistics as st, numpy as np
sys.argv=[sys.argv[0], sys.argv[1], sys.argv[2]]
exec(open('weekly_err_bench.py').read().split("ref = tm(run_f64)")[0])
VAL4 = VAL[LUTC]                                   # (256,4) f32: one gather per byte
def run_i2s_v4():
    out = np.empty((len(W), 3, 2))
    for t in range(len(W)):
        w = W[t]
        for j in range(3):
            g = np.float32(G[t, j]); q = VAL4[P[t][j]].reshape(-1)
            sqw = float(np.dot(q, w)); nz = np.count_nonzero(q)
            sse = W2[t] - 2*float(g)*sqw + float(g)**2*nz
            out[t, j] = (sse / W2[t], np.abs(w - g*q).max())
    return out
def run_f32_sse_only():
    out = np.empty((len(W), 3))
    for t in range(len(W)):
        w = W[t]
        for j in range(3):
            g = np.float32(G[t, j]); e = w - g*tern(w, g); out[t, j] = np.dot(e, e)/W2[t]
    return out
def run_i2s_sse_only():
    out = np.empty((len(W), 3))
    for t in range(len(W)):
        w = W[t]
        for j in range(3):
            g = float(G[t, j]); q = VAL4[P[t][j]].reshape(-1)
            out[t, j] = (W2[t] - 2*g*float(np.dot(q, w)) + g*g*np.count_nonzero(q))/W2[t]
    return out
ref = tm(run_f64)
for name,f,k in (("f32 recompute",run_f32,None),("i2s byte->4xf32 LUT",run_i2s_v4,None),("f32 recompute SSE only",run_f32_sse_only,0),("i2s LUT SSE only",run_i2s_sse_only,0)):
    r = tm(f); o = r[3] if k is None else r[3]
    rr = ref[3][...,0]; d0 = np.abs((o[...,0] if k is None else o)-rr)/rr
    print(f"{name}|{r[0]*1e3:.1f}|{r[1]*1e3:.1f}|{r[2]*1e3:.1f}|{ref[0]/r[0]:.2f}x vs f64|{d0.max():.2e}")
print("f64 ref median ms", f"{ref[0]*1e3:.1f}")
```

close: 465949f4e032fb15 (sha256(previous note text || this note text above this line)[:16])
