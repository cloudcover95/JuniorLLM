---
local: true
date: 2026-10-16T18:55:00-06:00
wave: "2026-10-16 18:05"
roadmap_item: "i2sd userspace 127.0.0.1:8767 no docker.sock"
roadmap_ref: "JuniorHome docs/ROADMAP.md item 3 (i2sd userspace, 127.0.0.1:8767, no docker.sock); Quant Chief pair_A assignment, wave 18:05"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnet, winsor, activation-quant, int4, int6, int8, popcount, matvec, b1, i2sd, benchmark]
---

# Winsor-clipped vs absmax activation quant for B=1 ternary matvec

## Vault index
- Prev: [[2026-10-16_1455_popcount-bitplane-matvec-vs-fp32]] close 5290f9aaa733dbf9
- Next: (to be written)

## Question
The 14:55 note found that int4 activations would halve the bitplane count of a popcount ternary matvec, but absmax int4 costs 14% to 16% output error. Following its next question: if the activation vector is Winsor-clipped at P0.1/P99.9 or P0.5/P99.5 before quantizing (instead of absmax), does int4 or int6 error fall to an acceptable level (at or under 1%, the bar used in earlier notes), and does the per-token percentile cost stay under 5% of a fp32 B=1 matvec? If yes, a fewer-plane ternary kernel for i2sd would get its 1.5x+ from planes alone.

## Method
- Synthetic ternary weights T in {-1, 0, +1} with p = 0.3 / 0.4 / 0.3, seed 20261016. Shapes (out x in) 4096x4096 and 2560x6912 (2B4T down-projection-like). No GGUF, no model pull, no network, nothing bound.
- Three activation distributions per shape: Gaussian N(0, 1); Student-t with 4 degrees of freedom (heavy tails); and Gaussian with a fixed 0.5% of channels scaled x20 (the outlier-channel pattern seen in transformer activations).
- Clip rules: absmax; w99.9 = symmetric clip at max(|P0.1|, |P99.9|); w99.5 = max(|P0.5|, |P99.5|). Then symmetric round-to-nearest quant to k = 4, 6, 8 bits.
- Error is relative L2 of `T @ deq(q(x))` against float64 `T @ x`, so weights add no error and all error is activation quantization plus clipping.
- Timing (Student-t vector): per-token scale cost via absmax, `np.percentile`, and an `np.partition` version that returns the identical interpolated percentile (asserted equal); the k=6 quantize pass; fp32 BLAS matvec; and the packed +1/-1 bitmask popcount matvec from the 14:55 note at k = 4, 6, 8 planes.
- 31 timed repeats after one warm-up; median and interquartile range (p25 to p75) in ms. 1 thread (single-core SBC case) and 8 threads (box default).
- Integer exactness: the k=6 popcount result on the clipped int vector equals `T.astype(int64) @ q` at every shape (`pc6_int_exact: true`).
- numpy 2.5.4, 8 vCPU box, peak process RSS 411.6 MB.

## Command
```
cd /workspace/tbench
for t in 1 8; do OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t python3 winsor_act_quant_bench.py 31 > r1855/threads$t.json; done
```

## Script (`winsor_act_quant_bench.py`, inline)
```python
#!/usr/bin/env python3
"""Winsor-clipped vs absmax activation quant for B=1 ternary matvec (numpy only, offline).
y = T @ x, T in {-1,0,1}. Error = rel L2 of T @ deq(q(x)) vs float64 T @ x (weights exact).
Timing: per-token scale cost (absmax / np.percentile / np.partition), full quantize pass,
fp32 BLAS matvec, and the packed-bitplane popcount matvec at k = 4, 6, 8 planes."""
import sys, time, json, resource, numpy as np
assert hasattr(np, "bitwise_count")
SEED = 20261016; REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 31
SHAPES = [(4096, 4096), (2560, 6912)]  # (out, in)
DISTS = ["gauss", "student_t4", "outlier_ch"]
CLIPS = ["absmax", "w99.9", "w99.5"]  # w99.5 = clip at max(|P0.5|, |P99.5|)
KS = (4, 6, 8)

def make_x(rng, d, C):
    if d == "gauss": return rng.standard_normal(C).astype(np.float32)
    if d == "student_t4": return rng.standard_t(4, C).astype(np.float32)
    x = rng.standard_normal(C).astype(np.float32)          # 0.5% fixed outlier channels x20
    idx = rng.choice(C, max(1, C // 200), replace=False); x[idx] *= 20.0; return x

def clip_val(x, clip):
    if clip == "absmax": return float(np.abs(x).max())
    p = {"w99.9": (0.1, 99.9), "w99.5": (0.5, 99.5)}[clip]
    lo, hi = np.percentile(x, p); return float(max(-lo, hi))

def clip_partition(x, clip):  # same value as np.percentile (linear interp) via np.partition, no full sort
    if clip == "absmax": return float(np.abs(x).max())
    q = {"w99.9": 0.001, "w99.5": 0.005}[clip]; n = x.size
    lo_h, hi_h = q * (n - 1), (1 - q) * (n - 1)
    idx = sorted({int(np.floor(lo_h)), int(np.ceil(lo_h)), int(np.floor(hi_h)), int(np.ceil(hi_h))})
    part = np.partition(x, idx).astype(np.float64)
    f = lambda h: part[int(np.floor(h))] + (h - np.floor(h)) * (part[int(np.ceil(h))] - part[int(np.floor(h))])
    return float(max(-f(lo_h), f(hi_h)))

def quant(x, k, c):
    qmax = 2 ** (k - 1) - 1; s = c / qmax
    return np.clip(np.rint(x / s), -qmax, qmax).astype(np.int64), s

def pack_mask(m): return np.packbits(m, axis=1, bitorder="little").view(np.uint64)

def popcount_matvec(P, N, rowsum, q, k, w8):
    half = 2 ** (k - 1); u = (q + half).astype(np.uint8)
    bits = ((u[None, :] >> np.arange(k, dtype=np.uint8)[:, None]) & 1).astype(bool)
    U = np.packbits(bits, axis=1, bitorder="little").view(np.uint64)
    cp = np.bitwise_count(P[:, None, :] & U[None, :, :]).sum(axis=2, dtype=np.int64)
    cn = np.bitwise_count(N[:, None, :] & U[None, :, :]).sum(axis=2, dtype=np.int64)
    return (cp - cn) @ w8[:k] - half * rowsum

def med(f, reps):
    f(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1e3)
    a = np.array(ts); return [round(float(np.median(a)), 4), round(float(np.percentile(a, 25)), 4), round(float(np.percentile(a, 75)), 4)]

rng = np.random.default_rng(SEED); w8 = (1 << np.arange(8)).astype(np.int64)
rows = []
for R, C in SHAPES:
    T = rng.choice(np.array([-1, 0, 1], np.int8), size=(R, C), p=[0.3, 0.4, 0.3])
    Wf = T.astype(np.float32); Td = T.astype(np.float64)
    P, N = pack_mask(T == 1), pack_mask(T == -1); rowsum = T.sum(axis=1, dtype=np.int64)
    shape = {"shape": f"{R}x{C}", "fp32_w_bytes": Wf.nbytes, "mask_bytes": P.nbytes + N.nbytes,
             "x_fp32_bytes": C * 4, "err": {}, "timing": {}}
    xs = {d: make_x(rng, d, C) for d in DISTS}
    for d, x in xs.items():
        ref = Td @ x.astype(np.float64); nr = np.linalg.norm(ref)
        for clip in CLIPS:
            c = clip_val(x, clip); assert abs(c - clip_partition(x, clip)) <= 1e-6 * max(1.0, c)
            for k in KS:
                q, s = quant(x, k, c)
                y = (Td @ q.astype(np.float64)) * s
                shape["err"][f"{d}|{clip}|k{k}"] = float(np.linalg.norm(y - ref) / nr)
        if d == "gauss":  # exactness of integer popcount path on clipped int6
            q, s = quant(x, 6, clip_val(x, "w99.5"))
            shape["pc6_int_exact"] = bool(np.array_equal(popcount_matvec(P, N, rowsum, q, 6, w8), T.astype(np.int64) @ q))
    x = xs["student_t4"]; tm = shape["timing"]
    tm["fp32_matvec"] = med(lambda: Wf @ x, REPS)
    tm["scale_absmax"] = med(lambda: clip_val(x, "absmax"), REPS)
    tm["scale_percentile_w99.5"] = med(lambda: clip_val(x, "w99.5"), REPS)
    tm["scale_partition_w99.5"] = med(lambda: clip_partition(x, "w99.5"), REPS)
    c = clip_val(x, "w99.5")
    tm["quantize_pass_k6"] = med(lambda: quant(x, 6, c), REPS)
    for k in KS:
        q, _ = quant(x, k, c)
        tm[f"popcount_k{k}"] = med(lambda: popcount_matvec(P, N, rowsum, q, k, w8), REPS)
    rows.append(shape); del T, Wf, Td, P, N
print(json.dumps({"reps": REPS, "seed": SEED, "numpy": np.__version__,
                  "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), "rows": rows}, indent=1))
```

## Results

### Output error (relative L2 vs float64), identical at 1 and 8 threads
| shape | activations | bits | absmax | Winsor P0.1/P99.9 | Winsor P0.5/P99.5 |
|---|---|---|---|---|---|
| 4096x4096 | Gaussian N(0,1) | int4 | 14.2% | 12.6% | 11.0% |
| 4096x4096 | Gaussian N(0,1) | int6 | 3.2% | 2.9% | 4.2% |
| 4096x4096 | Gaussian N(0,1) | int8 | 0.8% | 1.1% | 3.5% |
| 4096x4096 | Student-t df=4 | int4 | 50.6% | 24.5% | 23.6% |
| 4096x4096 | Student-t df=4 | int6 | 12.0% | 14.0% | 19.2% |
| 4096x4096 | Student-t df=4 | int8 | 2.9% | 13.3% | 19.0% |
| 4096x4096 | Gaussian + 0.5% channels x20 | int4 | 56.9% | 59.5% | 75.7% |
| 4096x4096 | Gaussian + 0.5% channels x20 | int6 | 32.2% | 46.4% | 75.4% |
| 4096x4096 | Gaussian + 0.5% channels x20 | int8 | 7.8% | 45.4% | 75.4% |
| 2560x6912 | Gaussian N(0,1) | int4 | 16.6% | 12.5% | 11.3% |
| 2560x6912 | Gaussian N(0,1) | int6 | 3.9% | 3.5% | 4.6% |
| 2560x6912 | Gaussian N(0,1) | int8 | 0.9% | 2.2% | 4.0% |
| 2560x6912 | Student-t df=4 | int4 | 40.4% | 24.4% | 21.9% |
| 2560x6912 | Student-t df=4 | int6 | 9.4% | 9.6% | 17.3% |
| 2560x6912 | Student-t df=4 | int8 | 2.3% | 8.3% | 17.1% |
| 2560x6912 | Gaussian + 0.5% channels x20 | int4 | 54.2% | 57.2% | 78.5% |
| 2560x6912 | Gaussian + 0.5% channels x20 | int6 | 28.8% | 41.0% | 78.1% |
| 2560x6912 | Gaussian + 0.5% channels x20 | int8 | 6.9% | 39.7% | 78.1% |

### Per-token scale cost (Student-t vector), ms median (IQR); percent is of the fp32 matvec
| threads | shape | fp32 matvec | absmax | np.percentile P0.5/P99.5 | np.partition P0.5/P99.5 | quantize pass int6 |
|---|---|---|---|---|---|---|
| 1 | 4096x4096 | 2.269 (2.200-2.289) | 0.002 (0.002-0.002) | 0.050 (0.047-0.054) (2.2%) | 0.029 (0.028-0.032) (1.3%) | 0.008 (0.008-0.008) |
| 1 | 2560x6912 | 2.457 (2.349-2.511) | 0.003 (0.002-0.003) | 0.121 (0.119-0.127) (4.9%) | 0.072 (0.071-0.074) (2.9%) | 0.010 (0.010-0.010) |
| 8 | 4096x4096 | 0.290 (0.281-0.307) | 0.003 (0.002-0.003) | 0.079 (0.073-0.088) (27.2%) | 0.037 (0.036-0.039) (12.8%) | 0.011 (0.011-0.011) |
| 8 | 2560x6912 | 0.590 (0.570-0.625) | 0.004 (0.004-0.004) | 0.136 (0.127-0.144) (23.0%) | 0.092 (0.090-0.094) (15.6%) | 0.015 (0.014-0.015) |

### Popcount matvec by plane count, ms median (IQR)
| threads | shape | fp32 BLAS | popcount 8 planes | 6 planes | 4 planes | 8-plane / 4-plane | fp32 / 4-plane |
|---|---|---|---|---|---|---|---|
| 1 | 4096x4096 | 2.269 (2.200-2.289) | 8.393 (8.091-8.693) | 6.314 (5.602-6.482) | 4.299 (4.067-4.405) | 1.95x | 0.53x |
| 1 | 2560x6912 | 2.457 (2.349-2.511) | 7.883 (7.521-8.002) | 5.686 (5.535-5.848) | 3.833 (3.641-3.937) | 2.06x | 0.64x |
| 8 | 4096x4096 | 0.290 (0.281-0.307) | 8.066 (7.268-8.122) | 6.077 (5.923-6.256) | 4.249 (4.076-4.556) | 1.90x | 0.07x |
| 8 | 2560x6912 | 0.590 (0.570-0.625) | 7.894 (7.133-8.118) | 5.641 (5.468-5.734) | 3.886 (3.603-3.955) | 2.03x | 0.15x |

Bytes per layer are unchanged from earlier notes: fp32 weights 67,108,864 (4096x4096) and 70,778,880 (2560x6912) vs +1/-1 bitmasks 4,194,304 and 4,423,680, 16.0x smaller. The activation vector is 16 KB or 27 KB, so activation precision changes no meaningful memory.

Raw JSON: `/workspace/tbench/r1855/threads1.json` and `threads8.json` (box-local, not pushed).

## Verdict
**Not viable.** Winsor clipping never brings int4 or int6 near 1%. It helps only at int4 on light tails (Gaussian 14.2% to 11.0% at 4096x4096; Student-t 50.6% to 23.6%), and at int6 and int8 it makes error worse on every heavy-tailed vector, because in a matvec the clipped outliers carry a large share of the output energy. On the outlier-channel vector, clipping is catastrophic (int8 error rises from 7.8% with absmax to 75% with P0.5/P99.5), and even absmax int8 misses the 1% bar there. Halving planes does give the expected speedup (4 planes are 1.95x to 2.06x faster than 8 at 1 thread), but 4-plane numpy popcount is still 1.6x to 1.9x slower than single-thread fp32 BLAS and 7x to 15x slower than 8-thread BLAS, at errors of 11% to 79%. The scale cost question passes at 1 thread (`np.percentile` 2.2% to 4.9% of the matvec, `np.partition` 1.3% to 2.9%) but fails at 8 threads (`np.percentile` 23% to 27%, `np.partition` 13% to 16%), against 0.1% for absmax.

For i2sd: keep per-token absmax (or per-group absmax) for activations; percentile clipping is the wrong tool for activation outliers. The quant-error budget belongs on the weight side (Winsor P5/P95 on 2B4T reconstruct, as in JuniorQuant `src/i2sd_residual_bound.py`), not on x.

## Ops check
`python3 scripts/terraform_lean_prod.py code` and `... llm` in local `/workspace/JuniorLLM` (offline, blender off): both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false`, `"launch": false`.

## This wave's pair files (2026-10-16 18:05)
Quant Chief last_assignment: pair_A i2sd userspace 127.0.0.1:8767 no docker.sock; pair_B JuniorOS overlay (Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH).
- Trit Pack: BitNet-mlx `scripts/i2sd_userspace_policy_key.py`, local commit 962f8e3, not on GitHub as of 18:59 MT (i2sd, matches pair_A)
- Winsor: JuniorQuant `i2sd_act_accum_bound.py`, local draft only, no push as of 18:59 MT (i2sd, matches pair_A)
- Side-Commit: [web3node bitnetd_unit_hardening_receipt.py @ 5dda3d5](https://github.com/cloudcover95/web3node/blob/5dda3d5/bitnetd_unit_hardening_receipt.py) (JuniorOS overlay, matches pair_B)
- Handshake & Ports: [JuniorHome ports/overlay_base_handshake.py @ c2d08b0](https://github.com/cloudcover95/JuniorHome/blob/c2d08b0/ports/overlay_base_handshake.py) (JuniorOS overlay, matches pair_B)

## Next question
Outlier-channel split instead of clipping: keep the top 0.5% |x| channels in fp32 and run them as a dense side matvec over just those weight columns, and send the rest through int4 or int6 popcount. Does that reach at most 1% error at int6 on the outlier-channel and Student-t vectors, and what is the gather plus side-matvec overhead against a fp32 B=1 matvec at 1 and 8 threads?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-16_1455_popcount-bitplane-matvec-vs-fp32.md.

close: 46a6ec876c84f026
