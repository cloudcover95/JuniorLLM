---
local: true
date: 2026-10-16T14:55:00-06:00
wave: "2026-10-16 14:05"
roadmap_item: "Live beta: juniorctl health/ask/night/security + test_junior_aie + test_linux_security"
roadmap_ref: "JuniorLLM docs/BETA_TO_OS.md ## Live beta, step 2 juniorctl ask (Quant Chief pair_A assignment, wave 14:05)"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnet, bitplane, popcount, int8-activations, int4-activations, matvec, b1, juniorctl-ask, live-beta, benchmark]
---

# B=1 ternary matvec with int8/int4 activation bitplanes and popcount vs fp32 BLAS

## Vault index
- Prev: [[2026-10-16_1055_packed-tile-gemm-vs-fp32]] close fabaef8eaf2be669
- Next: (to be written)

## Question
`juniorctl ask` decodes one token at a time, so every layer is a B=1 matvec. The 10:55 note showed that expanding packed trits back to fp32 always loses to BLAS. Can numpy skip the fp32 expand altogether? Keep the ternary weights as two packed bitmasks (+1 mask, -1 mask, 2 bits per weight), quantize the activation vector to k-bit integers, split it into k bitplanes, and compute W@x as signed popcounts with `np.bitwise_count`. Is that at least 1.5x faster than fp32 BLAS matvec, and what does activation quantization cost in error?

This is not the 2026-10-15 20:25 bit-plane bench: that one split the weights into boolean matrices and still multiplied by a float x through matmul. Here x is integer-quantized and the whole product is AND plus popcount, with no matmul over weights.

## Method
- Synthetic ternary weights T in {-1, 0, +1} with p = 0.3 / 0.4 / 0.3, seed 20261016; x ~ N(0, 1) fp32. No GGUF, no model pull, no network, nothing bound.
- Shapes (out x in): 4096x4096, 6912x2560 and 2560x6912 (2B4T FFN-like up and down projections).
- Weights: `P = packbits(T==1)`, `N = packbits(T==-1)` as uint64 words, plus an int64 row sum.
- Activations: symmetric absmax quant to k bits (k = 8 and k = 4), offset to unsigned u = q + 2^(k-1), k bitplanes packed to uint64.
- `W@q = sum_b 2^b * (popcount(P & U_b) - popcount(N & U_b)) - 2^(k-1) * rowsum`, then scale by s. Quantization time is included in the popcount timing.
- Reference: fp32 `Wf @ x` (OpenBLAS). Error is relative L2 against a float64 ternary matvec, so the weights add no error and all error is activation quantization.
- 31 timed repeats after one warm-up per path; median and interquartile range (p25 to p75) in ms. Run at 1 thread (the single-core SBC case, where numpy bitwise ops are already single-threaded) and at 8 threads (the box default).
- Integer exactness check: the k=8 popcount result divided by s equals `T.astype(int64) @ q` exactly at every shape (`pc8_int_exact: true`).
- numpy 2.5.4, 8 vCPU box, peak process RSS 339 MB for the whole run.

## Command
```
cd /workspace/tbench
for t in 1 8; do OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t python3 popcount_matvec_bench.py 31 > r1455/threads$t.json; done
```

## Script (`popcount_matvec_bench.py`, inline)
```python
#!/usr/bin/env python3
"""B=1 ternary matvec via int-quantized activation bitplanes + popcount (numpy only, offline).
y = W_t @ x, W_t in {-1,0,1}. Weights kept as two packed bitmasks (pos, neg) as uint64 words.
x quantized symmetric to k bits, offset to unsigned u = q + 2^(k-1), split into k bitplanes.
W@q = sum_b 2^b (popcnt(P&U_b) - popcnt(N&U_b)) - 2^(k-1) * rowsum(W)."""
import sys, time, json, numpy as np
assert hasattr(np, "bitwise_count")
SEED = 20261016; REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 31
SHAPES = [(4096, 4096), (6912, 2560), (2560, 6912)]  # (out, in); 2B4T FFN-like dims

def pack_mask(m):  # bool (R, C) -> uint64 (R, C/64)
    return np.packbits(m, axis=1, bitorder="little").view(np.uint64)

def quant_planes(x, k):
    half = 2 ** (k - 1); qmax = half - 1
    s = np.abs(x).max() / qmax
    q = np.clip(np.rint(x / s), -qmax, qmax).astype(np.int64)
    u = (q + half).astype(np.uint8)
    bits = ((u[None, :] >> np.arange(k, dtype=np.uint8)[:, None]) & 1).astype(bool)  # (k, C)
    return s, half, np.packbits(bits, axis=1, bitorder="little").view(np.uint64)  # (k, C/64)

def popcount_matvec(P, N, rowsum, x, k, w8):
    s, half, U = quant_planes(x, k)
    cp = np.bitwise_count(P[:, None, :] & U[None, :, :]).sum(axis=2, dtype=np.int64)  # (R, k)
    cn = np.bitwise_count(N[:, None, :] & U[None, :, :]).sum(axis=2, dtype=np.int64)
    acc = (cp - cn) @ w8[:k] - half * rowsum
    return acc.astype(np.float32) * np.float32(s)

def med(f, reps):
    f(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1e3)
    a = np.array(ts); return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))

rng = np.random.default_rng(SEED); w8 = (1 << np.arange(8)).astype(np.int64)
out = []
for R, C in SHAPES:
    T = rng.choice(np.array([-1, 0, 1], np.int8), size=(R, C), p=[0.3, 0.4, 0.3])
    Wf = T.astype(np.float32); x = rng.standard_normal(C).astype(np.float32)
    P, N = pack_mask(T == 1), pack_mask(T == -1); rowsum = T.sum(axis=1, dtype=np.int64)
    ref = Wf.astype(np.float64) @ x.astype(np.float64)
    row = {"shape": f"{R}x{C}", "fp32_bytes": Wf.nbytes, "mask_bytes": P.nbytes + N.nbytes}
    row["fp32_ms"] = med(lambda: Wf @ x, REPS)
    row["fp32_relerr"] = float(np.linalg.norm(Wf @ x - ref) / np.linalg.norm(ref))
    for k in (8, 4):
        y = popcount_matvec(P, N, rowsum, x, k, w8)
        row[f"pc{k}_ms"] = med(lambda: popcount_matvec(P, N, rowsum, x, k, w8), REPS)
        row[f"pc{k}_relerr"] = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
    # exactness check of integer path: compare against int64 matvec on the same quantized q
    s, half, _ = quant_planes(x, 8); q = np.clip(np.rint(x / s), -(half - 1), half - 1).astype(np.int64)
    row["pc8_int_exact"] = bool(np.array_equal(np.rint(popcount_matvec(P, N, rowsum, x, 8, w8) / np.float32(s)).astype(np.int64), T.astype(np.int64) @ q))
    out.append(row); del T, Wf, P, N
import resource
print(json.dumps({"reps": REPS, "seed": SEED, "numpy": np.__version__, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "rows": out}, indent=1))
```

## Results
Times are median ms with the interquartile range in brackets. "vs fp32" is fp32 time divided by popcount time, so above 1.0x would mean popcount is faster.

| threads | shape | fp32 BLAS ms | popcount int8 ms | int8 vs fp32 | int8 rel err | popcount int4 ms | int4 vs fp32 | int4 rel err |
|---|---|---|---|---|---|---|---|---|
| 1 | 4096x4096 | 2.16 (2.14-2.20) | 8.20 (7.49-8.65) | 0.26x | 7.93e-03 | 3.90 (3.86-4.07) | 0.55x | 1.42e-01 |
| 1 | 6912x2560 | 2.27 (2.24-2.31) | 9.67 (9.39-10.12) | 0.23x | 8.78e-03 | 4.85 (4.75-5.10) | 0.47x | 1.64e-01 |
| 1 | 2560x6912 | 2.47 (2.43-2.56) | 7.46 (7.29-7.91) | 0.33x | 8.94e-03 | 4.01 (3.76-4.15) | 0.62x | 1.63e-01 |
| 8 | 4096x4096 | 0.30 (0.30-0.31) | 7.52 (7.44-8.28) | 0.04x | 7.93e-03 | 3.90 (3.76-4.10) | 0.08x | 1.42e-01 |
| 8 | 6912x2560 | 0.61 (0.59-0.68) | 11.40 (10.80-12.45) | 0.05x | 8.78e-03 | 5.11 (5.00-5.19) | 0.12x | 1.64e-01 |
| 8 | 2560x6912 | 0.33 (0.32-0.35) | 7.99 (7.78-8.45) | 0.04x | 8.94e-03 | 3.89 (3.82-3.97) | 0.08x | 1.63e-01 |

| bytes per layer | fp32 | +1/-1 bitmasks | ratio |
|---|---|---|---|
| 4096x4096 | 67,108,864 | 4,194,304 | 16.0x smaller |
| 6912x2560 and 2560x6912 | 70,778,880 | 4,423,680 | 16.0x smaller |

Raw JSON: `/workspace/tbench/r1455/threads1.json` and `threads8.json` (box-local, not pushed).

## Verdict
**Not viable.** With int8 activations (error 0.79% to 0.89%, inside the 1% bar) the popcount path is 3.0x to 4.3x slower than single-thread fp32 BLAS and 19x to 25x slower than 8-thread BLAS. Dropping to int4 halves the popcount time, but it is still 1.6x to 2.1x slower than single-thread BLAS and the error jumps to 14% to 16%, which is not acceptable. The weight memory win is the same 16x already recorded in earlier notes, so there is nothing new on memory.

Why it loses: each output row costs 2k AND-plus-popcount passes over the packed row in numpy (16 passes for int8), each one materializing a temporary array, while BLAS does one fused pass over fp32. Popcount matvec only pays off inside a native kernel that keeps the planes in registers (bitnetd or an I2_S kernel in i2sd), not from numpy. For `juniorctl ask` on numpy today, fp32-resident BLAS stays the fastest decode path whenever the weights fit in RAM.

## Ops check
`python3 scripts/terraform_lean_prod.py code` and `... llm` in local `/workspace/JuniorLLM` (offline): both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false` (blender off), `"model_pull": false`, `"launch": false`.

## This wave's pair files (2026-10-16 14:05)
Quant Chief last_assignment: pair_A Live beta (juniorctl health/ask/night/security + tests); pair_B UE5 launched JUNIOR_UE5=1 watts>=80, os_route.launch false until then.
- Trit Pack: [JuniorLLM scripts/live_beta_gate_jtr1.py @ 7313790](https://github.com/cloudcover95/JuniorLLM/blob/7313790/scripts/live_beta_gate_jtr1.py) (Live beta, matches pair_A)
- Winsor: [JuniorStock quant/calibration/winsor_night_drift.py @ 873b833](https://github.com/cloudcover95/JuniorStock/blob/873b833/quant/calibration/winsor_night_drift.py) (juniorctl night drift, Live beta, matches pair_A)
- Side-Commit: [stocksnode src/stocksnode/core/ue5_gate_receipt_ledger.py @ 944e7f3](https://github.com/cloudcover95/stocksnode/blob/944e7f3/src/stocksnode/core/ue5_gate_receipt_ledger.py) (ROADMAP #5 UE5, matches pair_B)
- Handshake & Ports: [JuniorHome ports/ue5_launch_gate.py @ d257277](https://github.com/cloudcover95/JuniorHome/blob/d257277/ports/ue5_launch_gate.py) (ROADMAP #5 UE5, matches pair_B)

## Next question
Winsor error vs speed on the activation side: if x is Winsor-clipped at its P0.5/P99.5 percentiles before quantizing (instead of absmax), how far does int4 and int6 matvec error fall from 14-16%, and does the per-token `np.percentile` cost (versus `np.abs(x).max()`) stay under 5% of a fp32 B=1 matvec? A clip that gets int4 under 1% would halve the bitplane count for any future native kernel.

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-16_1055_packed-tile-gemm-vs-fp32.md.

close: 5290f9aaa733dbf9
