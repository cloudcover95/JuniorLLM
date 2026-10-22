---
local: true
date: 2026-10-21
wave: "2026-10-21 18:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (decode side of the native I2_S path on overlay hosts under bitnetd MemoryMax)"
chief_assignment: "pair_A Live beta step 5, non-quant-error side; pair_B JuniorOS overlay remaining gaps (Quant Chief 18:08 MT, last_tips.json)"
verdict: viable
tags: [ternary, osai, findings, junioros-overlay, i2s, decode, matvec, lut, blocking, numpy, memory]
---

# Blocked LUT unpack into sgemv: same matvec bits, 50-136x less decode scratch, 1.2-1.36x faster on FFN shapes

Previous note: [[2026-10-21_1455_blocked-threshold-swar-pack-vs-staged]]

## Question

The 14:55 note asked: the quantize path is now lean on the numpy fallback; does the decode side match it? Is a blocked unpack of I2_S bytes straight into the matvec, block by block in L2, at least 1.5x faster or smaller than unpacking the whole tensor first, with bit-identical matvec output?

## Method

- Weights: 2560x2560 (attention), 6912x2560 (FFN up/gate) and 2560x6912 (FFN down), Gaussian and Student-t(4) at scale 0.02, new seeds 1855 and 2855, f32, quantized with the landed lean path (abs-mean gamma, threshold at gamma/2, SWAR pack, byte = c3 | c2<<2 | c1<<4 | c0<<6, codes 0 / +1 -> 1 / -1 -> 2).
- Activations: int8-valued f32 vectors (|x| <= 127). Every partial sum is an integer below 2^24, so the f32 matvec is exact in any summation order and all paths must agree bit for bit.
- Paths, all ending in y = trits @ x:
  - **full_shift**: shift-and-mask the whole tensor to 2-bit codes, int8 trits, astype f32, sgemv.
  - **full_lut** (baseline, the best full-unpack path): one `np.take` of a 256x4 f32 table over the whole tensor, sgemv.
  - **blocked_lut**: R rows at a time, `np.take(..., out=buf)` into one preallocated f32 buffer of 128 KiB, 512 KiB or 2 MiB, then `np.matmul(..., out=y[s:s+m])`.
  - **f32_resident** (context only): the trit matrix kept as f32 in memory, plain sgemv, no decode.
- 15 timed repeats after a warm-up, median and min-max in ms; peak Python-heap scratch per call via tracemalloc; bit identity of every output against f32_resident. Identity sweep: 40 random odd shapes x 2 weight kinds x 5 paths against an int64 reference, plus a pack-to-trit round-trip check.
- Host: box, 8 cores, numpy 2.5.4. Offline, numpy and stdlib only, nothing bound, no downloads.

Command: `python3 bench.py` (script below; writes results.json).

## Results

Identity sweep: 400 outputs, 0 mismatches; pack/unpack trit round trip 0 mismatches. Every benchmark row below is also bit-identical across all six paths.

Times in ms, median (min-max); blocked uses the 512 KiB buffer; scratch is peak MB per call.

| shape | kind | seed | full_shift | full_lut | blocked_lut 512k | f32_resident | blocked vs full_lut | blocked vs full_shift | scratch full_lut / blocked MB | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 1855 | 16.39 (16.25-17.11) | 2.82 (2.78-3.04) | 3.06 (3.00-3.27) | 0.12 | 0.92x | 5.36x | 39.3 / 0.78 | yes |
| 2560x2560 | gauss | 2855 | 16.84 (16.29-23.44) | 2.67 (2.60-2.95) | 3.23 (2.94-3.75) | 0.13 | 0.83x | 5.22x | 39.3 / 0.78 | yes |
| 2560x2560 | t4 | 1855 | 16.26 (16.04-17.30) | 2.76 (2.70-3.00) | 3.15 (2.91-3.44) | 0.10 | 0.88x | 5.16x | 39.3 / 0.78 | yes |
| 2560x2560 | t4 | 2855 | 16.61 (16.32-20.04) | 3.15 (3.02-5.41) | 3.17 (2.92-3.62) | 0.12 | 0.99x | 5.24x | 39.3 / 0.78 | yes |
| 6912x2560 | gauss | 1855 | 48.45 (47.41-56.54) | 11.85 (11.11-13.37) | 9.04 (8.39-11.11) | 0.34 | 1.31x | 5.36x | 106.2 / 0.78 | yes |
| 6912x2560 | gauss | 2855 | 48.48 (46.48-54.58) | 11.43 (10.27-12.10) | 8.46 (8.25-14.99) | 0.29 | 1.35x | 5.73x | 106.2 / 0.78 | yes |
| 6912x2560 | t4 | 1855 | 48.23 (46.53-56.26) | 11.71 (10.54-26.48) | 8.60 (7.90-8.98) | 0.35 | 1.36x | 5.61x | 106.2 / 0.78 | yes |
| 6912x2560 | t4 | 2855 | 47.16 (45.84-49.47) | 10.70 (10.32-18.15) | 8.33 (8.24-8.71) | 0.29 | 1.28x | 5.66x | 106.2 / 0.78 | yes |
| 2560x6912 | gauss | 1855 | 47.45 (45.96-52.90) | 10.69 (10.08-11.36) | 9.07 (8.26-9.91) | 0.30 | 1.18x | 5.23x | 106.2 / 0.75 | yes |
| 2560x6912 | gauss | 2855 | 46.95 (45.56-65.29) | 10.37 (9.80-26.37) | 8.46 (7.76-8.82) | 0.28 | 1.23x | 5.55x | 106.2 / 0.75 | yes |
| 2560x6912 | t4 | 1855 | 47.08 (46.01-53.11) | 10.21 (9.75-10.81) | 8.53 (8.14-9.16) | 0.55 | 1.20x | 5.52x | 106.2 / 0.75 | yes |
| 2560x6912 | t4 | 2855 | 46.88 (45.32-49.81) | 11.13 (10.70-12.65) | 8.44 (7.85-8.61) | 0.31 | 1.32x | 5.55x | 106.2 / 0.75 | yes |

Block size, speedup vs full_lut:

| row | 128 KiB | 512 KiB | 2 MiB |
|---|---|---|---|
| 2560x2560 gauss 1855 | 0.84x | 0.92x | 0.47x |
| 2560x2560 gauss 2855 | 0.81x | 0.83x | 0.43x |
| 2560x2560 t4 1855 | 0.80x | 0.88x | 0.46x |
| 2560x2560 t4 2855 | 0.87x | 0.99x | 0.47x |
| 6912x2560 gauss 1855 | 1.21x | 1.31x | 0.63x |
| 6912x2560 gauss 2855 | 1.21x | 1.35x | 0.77x |
| 6912x2560 t4 1855 | 1.25x | 1.36x | 0.67x |
| 6912x2560 t4 2855 | 1.15x | 1.28x | 0.66x |
| 2560x6912 gauss 1855 | 1.04x | 1.18x | 0.63x |
| 2560x6912 gauss 2855 | 1.13x | 1.23x | 0.64x |
| 2560x6912 t4 1855 | 1.05x | 1.20x | 0.62x |
| 2560x6912 t4 2855 | 1.12x | 1.32x | 0.69x |

Resident bytes: packed I2_S is 1.64 MB (2560x2560) and 4.42 MB (FFN shapes) vs 26.2 MB and 70.8 MB for f32, 16x smaller.

## Verdict

**Viable on memory, mixed on speed.** With a 512 KiB block the decode-side scratch drops from 39.3 MB to 0.78 MB (about 50x) on 2560x2560 and from 106 MB to 0.75-0.78 MB (about 136x) on both FFN shapes, with bit-identical outputs everywhere. That matters under bitnetd MemoryMax: a full unpack of one FFN matrix briefly needs 24x the packed weight itself. Speed vs the best full unpack is 1.18-1.36x on FFN shapes and 0.83-0.99x on 2560x2560, so the speed half does not clear 1.5x on its own; it is 5.2-5.7x faster than the shift-unpack path. A 2 MiB block spills L2 and runs about 2x slower than 512 KiB, so the block size has to stay near 512 KiB.

Caveat for the roadmap: any per-call decode on the numpy fallback is still about 15-28x slower than an f32-resident sgemv (0.1-0.55 ms). The packed format wins 16x on resident bytes, but on numpy a single-vector decode-every-call matvec costs time; the native bitnetd/i2sd kernel has to do the decode in-register for that to change.

## This wave's pair files

- Pair A (Live beta step 5 non-QE): Trit Pack, JuniorLLM [044a549](https://github.com/cloudcover95/JuniorLLM/commit/044a549) `scripts/backfill_sparse_checkout_i2s_key.py`. No Winsor push seen.
- Pair B (JuniorOS overlay gaps): Handshake & Ports, JuniorLLM [426f420](https://github.com/cloudcover95/JuniorLLM/commit/426f420) `ports/overlay_runtime_identity_port_map.py` (hardened bitnetd.service runtime identity, the service this decode path runs under). No Side-Commit push seen.

## Next question

Decode cost per call is the remaining gap to f32-resident. With a batch of k activation vectors (prefill or the ask index), does one blocked LUT decode feeding sgemm amortize the decode so that packed I2_S comes within 1.5x of f32-resident time at k >= 16, while keeping the 16x smaller resident weights?

## Script

```python
#!/usr/bin/env python3
"""JuniorOS overlay decode side: I2_S packed bytes -> ternary matvec. Unpack the whole tensor first
(int8 trits then f32, or one LUT gather to f32) vs a blocked unpack of L2-sized row blocks into one
preallocated f32 buffer followed by sgemv into the output slice. Activations are int8-valued
(|x| <= 127), so every partial sum is an integer below 2^24 and f32 matvec is exact in any order:
outputs must be bit-identical. numpy + stdlib only, offline, no sockets, no installs."""
import json, time, tracemalloc, hashlib
import numpy as np

SHAPES = [(2560, 2560), (6912, 2560), (2560, 6912)]   # attn, FFN up/gate (out x in), FFN down
KINDS = ["gauss", "t4"]
SEEDS = [1855, 2855]
REP = 15
ROWBLOCK_BYTES = [1 << 17, 1 << 19, 1 << 21]          # f32 scratch per block: 128 KiB, 512 KiB, 2 MiB
M = np.uint32((1 << 30) | (1 << 20) | (1 << 10) | 1); LO = np.uint32(0x01010101)

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
    """Landed lean quantize path: abs-mean gamma, threshold at gamma/2, SWAR pack (byte = c3|c2<<2|c1<<4|c0<<6)."""
    f = w.reshape(-1); g = np.float32(np.abs(f).mean(dtype=np.float32) + np.float32(1e-7)); T = np.float32(g * np.float32(0.5))
    q = (f > T).view(np.int8) - (f < -T).view(np.int8)
    x = q.view(np.uint32); c = (x & LO) + ((x >> 7) & LO); c *= M; c >>= 24
    return c.astype(np.uint8).reshape(w.shape[0], w.shape[1] // 4), q.reshape(w.shape), g

TRIT = np.array([0, 1, -1, 0], np.int8)
LUT8 = np.array([[TRIT[(b >> s) & 3] for s in (6, 4, 2, 0)] for b in range(256)], np.int8)   # byte -> 4 trits
LUTF = LUT8.astype(np.float32)
SH = np.array([6, 4, 2, 0], np.uint8)

# A. full shift unpack: bytes -> 2-bit codes -> int8 trits (whole tensor), astype f32, sgemv
def full_shift(P, x, st):
    c = (P[:, :, None] >> SH) & np.uint8(3)
    t = (c & np.uint8(1)).view(np.int8) - (c >> np.uint8(1)).view(np.int8)
    return t.reshape(P.shape[0], -1).astype(np.float32) @ x
# B. full LUT: one gather byte -> 4 f32 trits over the whole tensor, sgemv
def full_lut(P, x, st):
    return np.take(LUTF, P, axis=0).reshape(P.shape[0], -1) @ x
# C. blocked LUT: R rows at a time gathered into a fixed f32 buffer (stays in L2), sgemv into y slice
def make_state(P, block_bytes):
    rows, nb = P.shape; R = max(1, min(rows, block_bytes // (nb * 16)))
    return {"R": R, "buf": np.empty((R, nb, 4), np.float32), "y": np.empty(rows, np.float32)}
def blocked_lut(P, x, st):
    R, buf, y = st["R"], st["buf"], st["y"]; rows = P.shape[0]
    for s in range(0, rows, R):
        m = min(R, rows - s); b = buf[:m]
        np.take(LUTF, P[s:s + m], axis=0, out=b)
        np.matmul(b.reshape(m, -1), x, out=y[s:s + m])
    return y

def main():
    out = {"numpy": np.__version__, "rep": REP, "rowblock_bytes": ROWBLOCK_BYTES, "rows": []}
    # identity sweep: odd shapes, all paths vs int64 reference of trits @ int activations
    rng = np.random.default_rng(18); idn = {"n": 0, "mismatch": 0, "pack_trit_mismatch": 0}
    for _ in range(40):
        r_, c_ = int(rng.integers(1, 3000)), int(rng.integers(1, 2000)) * 4
        for kind in KINDS:
            w = make(kind, (r_, c_), rng); P, q, g = quant_pack(w)
            idn["pack_trit_mismatch"] += int(not np.array_equal(np.take(LUT8, P, axis=0).reshape(r_, -1), q))
            x = rng.integers(-127, 128, c_).astype(np.float32)
            ref = (q.astype(np.int64) @ x.astype(np.int64)).astype(np.float32).tobytes()
            outs = [full_shift(P, x, None), full_lut(P, x, None)] + [blocked_lut(P, x, make_state(P, B)) for B in ROWBLOCK_BYTES]
            for o in outs: idn["n"] += 1; idn["mismatch"] += int(o.tobytes() != ref)
    out["identity"] = idn; print("identity", idn, flush=True)
    for shape in SHAPES:
        for kind in KINDS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed); w = make(kind, shape, rng); P, q, g = quant_pack(w)
                x = rng.integers(-127, 128, shape[1]).astype(np.float32)
                Wf = q.astype(np.float32)   # context only: f32-resident trit matrix (4n bytes resident)
                K = [("f32_resident", lambda P, x, st: Wf @ x, None), ("full_shift", full_shift, None), ("full_lut", full_lut, None)]
                for B in ROWBLOCK_BYTES: K.append((f"blocked_lut_{B // 1024}k", blocked_lut, make_state(P, B)))
                ref = (Wf @ x).tobytes()
                r = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "packed_bytes": P.nbytes,
                     "f32_bytes": Wf.nbytes, "y_sha16": hashlib.sha256(ref).hexdigest()[:16], "k": {}}
                for name, fn, st in K:
                    same = fn(P, x, st).tobytes() == ref
                    fixed = 0 if st is None else st["buf"].nbytes + st["y"].nbytes
                    r["k"][name] = {"t": tstat(lambda: fn(P, x, st)), "peak": peak(lambda: fn(P, x, st)),
                                    "fixed_bytes": fixed, "identical": bool(same)}
                for base in ("full_shift", "full_lut"):
                    b = r["k"][base]["t"]["med"]
                    for name in r["k"]: r["k"][name]["x_vs_" + base] = b / r["k"][name]["t"]["med"]
                out["rows"].append(r)
                print(r["shape"], kind, seed, " | ".join("%s %.2f(%.2f-%.2f) %.2fx/%.2fx pk=%.2fMB fx=%.2fMB id=%s" % (
                    k, v["t"]["med"], v["t"]["min"], v["t"]["max"], v["x_vs_full_shift"], v["x_vs_full_lut"], v["peak"] / 1e6,
                    v["fixed_bytes"] / 1e6, v["identical"]) for k, v in r["k"].items()), flush=True)
    json.dump(out, open("results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
```

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-21_1455_blocked-threshold-swar-pack-vs-staged.md.

close: 2f914259de583137
