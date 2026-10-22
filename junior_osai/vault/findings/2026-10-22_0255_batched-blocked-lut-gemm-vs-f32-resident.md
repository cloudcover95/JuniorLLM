---
local: true
date: 2026-10-22
wave: "2026-10-22 02:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (batched decode side of the native I2_S path on overlay hosts under bitnetd MemoryMax)"
chief_assignment: "pair_A JuniorOS overlay remaining gaps; pair_B Live beta step 5 non-quant-error side (Quant Chief 02:09 MT, last_tips.json)"
verdict: not-viable
tags: [ternary, osai, findings, junioros-overlay, i2s, decode, gemm, batching, lut, numpy, not-viable]
---

# Batched blocked LUT decode into sgemm: bit-identical, but still 6-35x slower than f32-resident on numpy

Previous note: [[2026-10-21_1855_blocked-lut-unpack-matvec-vs-full-unpack]]

## Question

The 18:55 note asked: with a batch of k activation vectors (prefill or the ask index), does one blocked LUT decode of the packed I2_S bytes feeding sgemm amortize the decode, so packed I2_S comes within 1.5x of an f32-resident sgemm at k >= 16 while keeping the 16x smaller resident weights?

## Method

- Weights: 2560x2560, 6912x2560 and 2560x6912, Gaussian and Student-t(4) at scale 0.02, new seed 255, quantized with the landed lean path (abs-mean gamma, threshold gamma/2, SWAR pack, byte = c3 | c2<<2 | c1<<4 | c0<<6).
- Activations: int8-valued f32 matrices X of shape (in, k), k in {1, 4, 16, 64, 256}. Every partial sum is an integer below 2^24, so results must be bit-identical.
- Paths: **f32_resident** (trit matrix kept as f32, one `Wf @ X`) vs **blocked_gemm** (512 KiB f32 decode buffer, `np.take` LUT gather of R rows then `np.matmul(..., out=Y[s:s+m])`, the best block from the 18:55 note).
- 11 timed repeats after a warm-up, median and min-max in ms. Identity sweep: 30 random odd shapes and k against an int64 reference.
- Host: box, 8 cores, numpy 2.5.4. Offline, numpy and stdlib only, nothing bound, no downloads.

Command: `python3 bench.py` (script below; writes results.json).

## Results

Identity sweep: 30 outputs, 0 mismatches. Every row below is bit-identical to f32_resident.

Times in ms, median (min-max); ratio = blocked / f32_resident (lower is better, target <= 1.5x).

| shape | kind | k | f32_resident | blocked_gemm 512k | ratio | identical |
|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 1 | 0.10 (0.10-0.14) | 3.04 (2.78-3.20) | 29.81x | yes |
| 2560x2560 | gauss | 4 | 0.22 (0.22-0.30) | 3.55 (3.45-4.29) | 15.95x | yes |
| 2560x2560 | gauss | 16 | 0.26 (0.26-0.32) | 7.86 (7.71-12.60) | 29.91x | yes |
| 2560x2560 | gauss | 64 | 0.64 (0.63-0.73) | 18.08 (17.67-23.57) | 28.10x | yes |
| 2560x2560 | gauss | 256 | 2.25 (2.17-6.83) | 19.46 (17.71-23.54) | 8.64x | yes |
| 2560x2560 | t4 | 1 | 0.12 (0.11-0.14) | 3.12 (2.80-5.24) | 26.75x | yes |
| 2560x2560 | t4 | 4 | 0.22 (0.22-0.29) | 3.46 (3.40-3.63) | 15.71x | yes |
| 2560x2560 | t4 | 16 | 0.26 (0.26-0.28) | 5.82 (5.67-6.35) | 22.19x | yes |
| 2560x2560 | t4 | 64 | 0.70 (0.65-9.88) | 9.92 (9.24-15.27) | 14.12x | yes |
| 2560x2560 | t4 | 256 | 2.22 (2.14-5.88) | 18.40 (16.24-27.46) | 8.30x | yes |
| 6912x2560 | gauss | 1 | 0.29 (0.29-0.40) | 8.50 (8.05-14.11) | 29.22x | yes |
| 6912x2560 | gauss | 4 | 0.88 (0.85-1.14) | 8.81 (8.43-13.36) | 10.01x | yes |
| 6912x2560 | gauss | 16 | 0.93 (0.91-1.06) | 15.30 (15.02-22.90) | 16.47x | yes |
| 6912x2560 | gauss | 64 | 1.94 (1.70-3.37) | 25.64 (24.12-45.46) | 13.23x | yes |
| 6912x2560 | gauss | 256 | 12.57 (7.64-14.01) | 105.13 (61.60-475.80) | 8.37x | yes |
| 6912x2560 | t4 | 1 | 0.93 (0.28-4.36) | 8.31 (7.93-8.79) | 8.94x | yes |
| 6912x2560 | t4 | 4 | 0.85 (0.83-3.07) | 8.75 (8.28-14.20) | 10.29x | yes |
| 6912x2560 | t4 | 16 | 0.89 (0.88-4.81) | 24.00 (21.34-27.02) | 26.89x | yes |
| 6912x2560 | t4 | 64 | 2.63 (1.70-5.66) | 50.34 (48.44-53.49) | 19.12x | yes |
| 6912x2560 | t4 | 256 | 8.73 (5.65-10.61) | 103.38 (78.93-153.90) | 11.85x | yes |
| 2560x6912 | gauss | 1 | 0.30 (0.30-0.37) | 9.02 (8.58-13.64) | 29.84x | yes |
| 2560x6912 | gauss | 4 | 0.88 (0.59-3.40) | 9.50 (9.28-94.26) | 10.84x | yes |
| 2560x6912 | gauss | 16 | 0.83 (0.69-3.49) | 29.17 (26.47-29.96) | 35.26x | yes |
| 2560x6912 | gauss | 64 | 8.63 (5.81-13.13) | 55.63 (53.67-56.82) | 6.45x | yes |
| 2560x6912 | gauss | 256 | 13.70 (6.98-80.27) | 166.22 (139.91-251.81) | 12.13x | yes |
| 2560x6912 | t4 | 1 | 0.29 (0.29-0.32) | 8.32 (7.84-8.66) | 28.22x | yes |
| 2560x6912 | t4 | 4 | 0.71 (0.59-2.15) | 9.41 (9.00-9.48) | 13.17x | yes |
| 2560x6912 | t4 | 16 | 0.72 (0.70-1.34) | 16.23 (15.98-18.07) | 22.69x | yes |
| 2560x6912 | t4 | 64 | 2.10 (1.73-5.51) | 47.95 (46.99-55.74) | 22.82x | yes |
| 2560x6912 | t4 | 256 | 6.12 (5.68-24.17) | 72.59 (68.43-77.03) | 11.85x | yes |

Resident bytes: packed I2_S 1.64 MB (2560x2560) and 4.42 MB (FFN shapes) vs 26.2 MB and 70.8 MB for f32, still 16x smaller. Decode scratch stays at the fixed 0.52 MB buffer plus the output.

## Verdict

**Not viable on speed.** Batching does not amortize the decode on the numpy fallback: the best ratio anywhere is 6.45x slower than f32-resident, and at k >= 16 it is 6.45-35.26x slower, nowhere near the 1.5x target. The ratio does shrink at k = 256 (8-12x), but the 512 KiB block holds only about 50 rows of a 2560-wide matrix (about 19 of a 6912-wide one), so sgemm gets many thin calls and loses most of its blocking efficiency, while the LUT gather itself costs about as much as the whole f32 sgemm. The memory win from the 18:55 note (16x resident, sub-MB scratch) still holds, so packed I2_S on numpy is a memory format, not a speed format. Closing that gap needs the native bitnetd/i2sd kernel decoding in-register; numpy cannot get there by batching.

## This wave's pair files

- Pair A (JuniorOS overlay gaps): Trit Pack, BitNet-mlx [33e813e](https://github.com/cloudcover95/BitNet-mlx/commit/33e813e) `scripts/junioros_oci_runtime_spec_i2s_key.py` (rootless OCI runtime-spec closure). No Winsor push seen by 02:55.
- Pair B (Live beta step 5 non-QE): no Side-Commit or Handshake push seen by 02:55 (Handshake clone at 02:35, no commit yet).
- Also seen, not a pair file: JuniorLLM [8a55f08](https://github.com/cloudcover95/JuniorLLM/commit/8a55f08) `rails/schedule/autonomy_window.py`; this note does not touch that path.

## Next question

Since numpy decode cannot beat f32 on time, does the memory side pay off at the service level: under a bitnetd MemoryMax=2G budget, how many 2B-model-sized layer sets fit resident as packed I2_S plus a 512 KiB decode buffer vs f32 or fp16 resident, and what is the per-token decode-time cost of that fit?

## Script

```python
#!/usr/bin/env python3
"""JuniorOS overlay decode side, batched: does one blocked LUT decode of I2_S bytes feeding sgemm
(k activation columns at once) amortize the decode so packed I2_S comes within 1.5x of an
f32-resident sgemm, while weights stay 16x smaller resident? Activations are int8-valued
(|x| <= 127) so all partial sums are integers < 2^24 and f32 results are exact: outputs must be
bit-identical. numpy + stdlib only, offline, no sockets, no installs."""
import json, time, tracemalloc, hashlib
import numpy as np

SHAPES = [(2560, 2560), (6912, 2560), (2560, 6912)]
KINDS = ["gauss", "t4"]
SEED = 255
KS = [1, 4, 16, 64, 256]
REP = 11
BLOCK = 1 << 19                       # 512 KiB f32 decode buffer (best block from 18:55 note)
M = np.uint32((1 << 30) | (1 << 20) | (1 << 10) | 1); LO = np.uint32(0x01010101)
TRIT = np.array([0, 1, -1, 0], np.int8)
LUTF = np.array([[TRIT[(b >> s) & 3] for s in (6, 4, 2, 0)] for b in range(256)], np.float32)

def tstat(f, n=REP):
    f(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts); return {"med": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": n}

def peak(f):
    tracemalloc.start(); tracemalloc.reset_peak(); f(); p = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return p

def make(kind, shape, rng):
    if kind == "gauss": return rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)
    return (rng.standard_t(4, size=shape) * 0.02).astype(np.float32)

def quant_pack(w):
    f = w.reshape(-1); g = np.float32(np.abs(f).mean(dtype=np.float32) + np.float32(1e-7)); T = np.float32(g * np.float32(0.5))
    q = (f > T).view(np.int8) - (f < -T).view(np.int8)
    x = q.view(np.uint32); c = (x & LO) + ((x >> 7) & LO); c *= M; c >>= 24
    return c.astype(np.uint8).reshape(w.shape[0], w.shape[1] // 4), q.reshape(w.shape)

def state(P, k):
    rows, nb = P.shape; R = max(1, min(rows, BLOCK // (nb * 16)))
    return {"R": R, "buf": np.empty((R, nb, 4), np.float32), "Y": np.empty((rows, k), np.float32)}

def blocked_gemm(P, X, st):
    R, buf, Y = st["R"], st["buf"], st["Y"]; rows = P.shape[0]
    for s in range(0, rows, R):
        m = min(R, rows - s); b = buf[:m]
        np.take(LUTF, P[s:s + m], axis=0, out=b)
        np.matmul(b.reshape(m, -1), X, out=Y[s:s + m])
    return Y

def main():
    out = {"numpy": np.__version__, "rep": REP, "block_bytes": BLOCK, "ks": KS, "rows": []}
    rng = np.random.default_rng(22); idn = {"n": 0, "mismatch": 0}
    for _ in range(30):
        r_, c_ = int(rng.integers(1, 2000)), int(rng.integers(1, 1500)) * 4; k = int(rng.integers(1, 80))
        P, q = quant_pack(make("t4", (r_, c_), rng)); X = rng.integers(-127, 128, (c_, k)).astype(np.float32)
        ref = (q.astype(np.int64) @ X.astype(np.int64)).astype(np.float32)
        idn["n"] += 1; idn["mismatch"] += int(not np.array_equal(blocked_gemm(P, X, state(P, k)), ref))
    out["identity"] = idn; print("identity", idn, flush=True)
    for shape in SHAPES:
        for kind in KINDS:
            rng = np.random.default_rng(SEED); P, q = quant_pack(make(kind, shape, rng)); Wf = q.astype(np.float32)
            for k in KS:
                X = rng.integers(-127, 128, (shape[1], k)).astype(np.float32); st = state(P, k)
                ref = Wf @ X; same = bool(np.array_equal(blocked_gemm(P, X, st), ref))
                tr = tstat(lambda: Wf @ X); tb = tstat(lambda: blocked_gemm(P, X, st))
                pk = peak(lambda: blocked_gemm(P, X, st))
                r = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "k": k, "f32_res": tr, "blocked": tb,
                     "ratio": tb["med"] / tr["med"], "packed_bytes": P.nbytes, "f32_bytes": Wf.nbytes,
                     "fixed_scratch": st["buf"].nbytes, "peak_py": pk, "identical": same,
                     "y_sha16": hashlib.sha256(ref.tobytes()).hexdigest()[:16]}
                out["rows"].append(r)
                print("%s %s k=%d res %.2f(%.2f-%.2f) blk %.2f(%.2f-%.2f) ratio %.2fx id=%s" % (
                    r["shape"], kind, k, tr["med"], tr["min"], tr["max"], tb["med"], tb["min"], tb["max"], r["ratio"], same), flush=True)
    json.dump(out, open("results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
```

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-21_1855_blocked-lut-unpack-matvec-vs-full-unpack.md.

close: 13de6ad01505fae5
