---
local: true
date: 2026-10-21
wave: "2026-10-21 05:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (ternarize step of the native f32 quantize path on overlay hosts)"
chief_assignment: "pair_A JuniorOS overlay remaining gaps; pair_B Live beta step 5, non-quant-error side (Quant Chief 05:12 MT, last_tips.json)"
verdict: viable
tags: [ternary, osai, findings, junioros-overlay, ternarize, threshold, rint, numpy, quantize]
---

# Threshold ternarize vs divide + rint + clip: same trits, about 3.5x faster

Previous note: [[2026-10-20_0555_swar-u32-i2s-pack-vs-shiftor-chain]]

## Question

The 05:55 note left ternarize (w / gamma, rint, clip, astype int8) as the main cost of the quantize path once the SWAR pack landed. Is a threshold ternarize that skips the divide and the rint, writing trits from two comparisons of w against +/-T straight into int8, at least 1.5x faster than the current path with zero trit flips, ties included?

## Method

- Same tensors as the 02:55 and 05:55 notes: 2560x2560 and 2560x6912 (2B4T attention and FFN sizes), Gaussian and Student-t(4) weights at scale 0.02, seeds 555 and 1555, f32.
- gamma = f32 mean |w| + 1e-7, as in the current path. Current ternarize: `clip(rint(w / gamma), -1, 1).astype(int8)`. rint is half-even, so after the clip a trit is +1 exactly when fl(w / gamma) > 0.5 and -1 exactly when fl(w / gamma) < -0.5 (a tie at 0.5 rounds to 0; 1.5 rounds to 2 and clips to 1).
- Exact cutoff: f32 division by a positive gamma is monotone and odd in w, so there is one largest f32 T with fl(T / gamma) <= 0.5, found by starting at fl(gamma * 0.5) and stepping with `nextafter`. Then +1 iff w > T and -1 iff w < -T, with no approximation. It costs a few microseconds per tensor (at most 4.3 us here), and on all 8 tensors T came out exactly fl(gamma / 2).
- Kernels: (1) current divide + rint + clip + astype, (2) the same arithmetic in one reused f32 buffer, (3) threshold: `(w > T).view(int8) - (w < -T).view(int8)`, (4) sign bit + magnitude mask with `np.where`.
- Checks: 1,608 edge values (128 f32 neighbours either side of T, exact 0.5 and 1.5 ties, +0 and -0, both signs, six gammas from 3e-3 to 1.0) against kernel (1); every kernel's trits against kernel (1) on every tensor; SWAR-packed bytes from both paths compared.
- Timing: 15 repeats per kernel after one warm-up (7 for gamma, 9 for end to end), median (min-max) in ms with `time.perf_counter`, peak traced allocation with `tracemalloc`. numpy 2.5.4, 8-core box, single-thread numpy elementwise, nothing bound, no network, no installs. The script ran twice; tables are run 2, and run 1 is summarised in the verdict.

Command (script kept at `/workspace/tbench/r20261021_0555/threshold_ternarize_bench.py`, reproduced in full below):

```
cd /workspace/tbench/r20261021_0555 && python3 threshold_ternarize_bench.py
```

## Results

### A. Ternarize time (ms, median (min-max), 15 repeats) and speedup over the current path

| shape | kind | seed | divide + rint + clip (current) | same, in place | threshold compare (new) | sign bit + abs mask |
|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 555 | 7.67 (7.51-8.52), 1.00x | 4.22 (4.11-4.35), 1.82x | 2.58 (2.50-2.71), 2.98x | 28.64 (27.74-30.28), 0.27x |
| 2560x2560 | gauss | 1555 | 8.22 (7.60-9.16), 1.00x | 4.29 (4.17-4.61), 1.92x | 2.52 (2.46-2.73), 3.26x | 28.52 (27.93-29.10), 0.29x |
| 2560x2560 | t4 | 555 | 5.58 (5.51-5.87), 1.00x | 4.28 (4.18-4.38), 1.30x | 2.60 (2.51-3.69), 2.14x | 28.25 (28.01-28.95), 0.20x |
| 2560x2560 | t4 | 1555 | 7.66 (7.51-8.08), 1.00x | 4.28 (4.13-13.64), 1.79x | 2.50 (2.46-2.79), 3.06x | 28.10 (27.70-29.40), 0.27x |
| 2560x6912 | gauss | 555 | 30.69 (27.94-34.03), 1.00x | 15.30 (14.74-17.99), 2.01x | 7.26 (7.06-8.59), 4.22x | 97.64 (90.02-115.05), 0.31x |
| 2560x6912 | gauss | 1555 | 27.39 (26.09-32.87), 1.00x | 14.73 (14.37-19.80), 1.86x | 7.10 (7.00-8.33), 3.86x | 92.12 (87.77-99.59), 0.30x |
| 2560x6912 | t4 | 555 | 28.98 (26.43-43.75), 1.00x | 14.94 (14.40-33.42), 1.94x | 6.97 (6.75-8.70), 4.16x | 93.81 (88.60-106.40), 0.31x |
| 2560x6912 | t4 | 1555 | 26.45 (25.78-42.07), 1.00x | 13.84 (13.64-25.32), 1.91x | 7.19 (6.82-8.07), 3.68x | 95.90 (93.46-104.51), 0.28x |

### B. Trit identity, cutoff and peak memory

| shape | kind | seed | trits | gamma | T == fl(gamma/2) | trit flips (all 4 kernels) | trit sha16 | peak MB current / in place / threshold | packed bytes identical |
|---|---|---|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 555 | 6553600 | 0.0159653 | yes | 0 | `253e32252b727441` | 52.4 / 32.8 / 19.7 | yes |
| 2560x2560 | gauss | 1555 | 6553600 | 0.0159682 | yes | 0 | `b17fd3732026e510` | 52.4 / 32.8 / 19.7 | yes |
| 2560x2560 | t4 | 555 | 6553600 | 0.0199881 | yes | 0 | `8faa3807f862ad86` | 52.4 / 32.8 / 19.7 | yes |
| 2560x2560 | t4 | 1555 | 6553600 | 0.0200021 | yes | 0 | `6c0c79f17739484b` | 52.4 / 32.8 / 19.7 | yes |
| 2560x6912 | gauss | 555 | 17694720 | 0.0159592 | yes | 0 | `2c7d2becbf4850bc` | 141.6 / 88.5 / 53.1 | yes |
| 2560x6912 | gauss | 1555 | 17694720 | 0.0159598 | yes | 0 | `3322fb3cb938e4f6` | 141.6 / 88.5 / 53.1 | yes |
| 2560x6912 | t4 | 555 | 17694720 | 0.0199892 | yes | 0 | `3f99b58c0e40b1ee` | 141.6 / 88.5 / 53.1 | yes |
| 2560x6912 | t4 | 1555 | 17694720 | 0.0199953 | yes | 0 | `50de8aeae6db78e6` | 141.6 / 88.5 / 53.1 | yes |

Edge set: 0 flips over 1608 values for the threshold and sign-mask kernels.

### C. End to end: gamma + ternarize + SWAR pack (ms, median (min-max), 9 repeats)

| shape | kind | seed | gamma (abs mean) | gamma + rint path + SWAR pack | gamma + cutoff + threshold + SWAR pack | speedup |
|---|---|---|---|---|---|---|
| 2560x2560 | gauss | 555 | 2.71 (2.65-3.06) | 12.79 (12.59-17.32) | 7.32 (7.20-7.69) | 1.75x |
| 2560x2560 | gauss | 1555 | 2.98 (2.74-3.93) | 13.18 (12.85-18.29) | 7.38 (7.19-7.50) | 1.78x |
| 2560x2560 | t4 | 555 | 2.72 (2.68-2.91) | 10.43 (10.28-13.88) | 7.33 (7.24-7.84) | 1.42x |
| 2560x2560 | t4 | 1555 | 2.80 (2.65-5.26) | 13.02 (12.78-17.54) | 7.37 (7.19-7.74) | 1.77x |
| 2560x6912 | gauss | 555 | 12.75 (11.23-16.73) | 62.04 (57.02-92.76) | 30.81 (29.54-42.33) | 2.01x |
| 2560x6912 | gauss | 1555 | 11.25 (10.61-16.60) | 57.10 (54.17-82.03) | 25.80 (25.06-31.77) | 2.21x |
| 2560x6912 | t4 | 555 | 13.80 (12.13-21.26) | 57.18 (53.65-70.11) | 25.37 (24.65-26.78) | 2.25x |
| 2560x6912 | t4 | 1555 | 9.74 (9.55-13.04) | 58.45 (55.61-85.30) | 24.64 (23.60-26.92) | 2.37x |

## Verdict

**Viable.** The threshold ternarize is 3.47x (2.14-4.22x) faster than divide + rint + clip (median over 8 tensors, range in brackets; run 1 gave 3.71x (2.23-9.77x)), with zero trit flips on every tensor and on all 1,608 edge values, so the SWAR-packed bytes, and any close key over them, are identical. Error is zero by construction because the cutoff is the exact f32 boundary of the current rule, not gamma / 2 taken on faith. Peak memory falls 2.67x (2.67-2.67x) (52.4 MB to 19.7 MB at 2560x2560): it needs two n-byte bool masks instead of an f32 quotient and its rint copy. End to end, gamma + ternarize + pack is 1.90x (1.42-2.37x) faster (run 1 2.05x (1.64-2.52x)); the smallest win is the 2560x2560 t4 seed 555 tensor, where the current path is already quick.

- **Doing the current arithmetic in place helps but is not the answer:** 1.89x (1.30-2.01x). The divide and the rint still touch every value as f32.
- **The sign bit + abs mask variant is a loss** (about 0.3x): `np.where` over int8 with a negated copy is slow in numpy.
- **What is left:** gamma (the f32 abs mean) is now the largest piece, roughly 35-55% of the end-to-end time on the new path; ternarize and pack together are about 13 ms at 2560x6912.
- **Preconditions:** gamma > 0 and finite; a NaN weight gives trit 0 here where the current path's astype of NaN is undefined, so NaN must still be rejected upstream. The cutoff has to be computed per tensor (or per block if the I2_S 128-block key's per-block gamma is used) with the same f32 gamma the current code uses; fl(gamma / 2) matched on these tensors but the `nextafter` search is what guarantees it.
- **What it does not change:** trits, pack bytes, the JTR1/I2_S format, Trit Pack's and Winsor's keys and the close rules all stay as they are. This is a drop-in for the numpy fallback ternarize; the C path (`absmean.c` / `roundf`) was not measured.
- **Caveat:** synthetic weights and single-thread numpy on this box. Ternarize time depends on the element count, not the values, so real 2B4T tensors should land in the same range.

## This wave's pair files

- pair_A (JuniorOS overlay remaining gaps): Trit Pack [BitNet-mlx f5ec303](https://github.com/cloudcover95/BitNet-mlx/commit/f5ec303) (JuniorOS overlay sysctl drop-in JTR1/I2_S key), Winsor [JuniorStock 6e35b54](https://github.com/cloudcover95/JuniorStock/commit/6e35b54) (overlay bitplane.c f32 matvec error certificate).
- pair_B (Live beta step 5, non-quant-error side): Handshake [JuniorLLM e1ea106 live beta step 5 step-order writer map](https://github.com/cloudcover95/JuniorLLM/commit/e1ea106). Side-Commit's 05:35 push was not seen in the local copies.

## Next question

With ternarize and pack down to a few ms, gamma (f32 abs mean) dominates. Is a single fused pass, such as `np.add.reduce` over a uint32 view with the sign bit masked, or a block-128 abs-sum via reshape, at least 1.5x faster than `np.abs(w).mean(dtype=float32)` while keeping gamma bit-identical (or within a bound that flips zero trits)?

## Script

```python
#!/usr/bin/env python3
"""Threshold ternarize vs the divide + rint + clip path (JuniorOS overlay quantize path).
Question: is writing trits from two comparisons of w against +/-T (T = exact f32 cutoff of
fl(w / (gamma + eps)) > 0.5, so ties match rint half-even) at least 1.5x faster than
clip(rint(w / g)).astype(int8), with zero trit flips?
numpy + stdlib only, offline, no sockets, no installs. Writes results.json."""
import json, sys, time, tracemalloc, hashlib
import numpy as np

SHAPES = [(2560, 2560), (2560, 6912)]
KINDS = ["gauss", "t4"]
SEEDS = [555, 1555]
REP = 15
EPS = np.float32(1e-7)
M = np.uint32((1 << 30) | (1 << 20) | (1 << 10) | 1)
LO = np.uint32(0x01010101)

def tstat(f, n=REP):
    f(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return {"med": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": n}

def peak(f):
    tracemalloc.start(); tracemalloc.reset_peak(); f()
    p = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return p

def make(kind, shape, rng):
    if kind == "gauss":
        return rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)
    return (rng.standard_t(4, size=shape) * 0.02).astype(np.float32)

def gamma_of(w):
    return np.float32(np.float32(np.abs(w).mean(dtype=np.float32)) + EPS)

def cutoff(g):
    """Largest f32 T >= 0 with fl(T / g) <= 0.5; then fl(w/g) > 0.5 <=> w > T (f32 division
    is monotone and odd, so fl(w/g) < -0.5 <=> w < -T)."""
    g = np.float32(g); half = np.float32(0.5)
    t = np.float32(g * half)
    while np.float32(t / g) > half: t = np.nextafter(t, np.float32(0))
    while np.float32(np.nextafter(t, np.float32(np.inf)) / g) <= half: t = np.nextafter(t, np.float32(np.inf))
    return np.float32(t)

def tern_rint(w, g):          # current path (02:55 / 05:55 notes)
    return np.clip(np.rint(w / g), -1, 1).astype(np.int8)

def tern_rint_inplace(w, g):  # same arithmetic, one f32 temp reused
    x = np.divide(w, g); np.rint(x, out=x); np.clip(x, -1, 1, out=x); return x.astype(np.int8)

def tern_thresh(w, T):        # two compares, bool viewed as int8, one int8 subtract
    return (w > T).view(np.int8) - (w < -T).view(np.int8)

def tern_thresh_abs(w, T):    # sign bit + magnitude mask
    m = (np.abs(w) > T).view(np.int8)
    return np.where(np.signbit(w), -m, m).astype(np.int8)

def pack_swar(q):
    x = q.view(np.uint32); c = (x & LO) + ((x >> 7) & LO); c *= M; c >>= 24; return c.astype(np.uint8)

def main():
    out = {"numpy": np.__version__, "rep": REP, "rows": []}
    # edge check: values straddling the cutoff for several gammas, incl. exact ties at 0.5 and 1.5
    edge = {"flips": 0, "n": 0}
    for gv in [0.0137, 0.02, 0.0159, 1.0, 0.75, 3e-3]:
        g = np.float32(gv); T = cutoff(g)
        vals = [T]
        for _ in range(64): vals.append(np.nextafter(vals[-1], np.float32(np.inf)))
        v = [T]
        for _ in range(64): v.append(np.nextafter(v[-1], np.float32(0)))
        arr = np.array(vals + v + [np.float32(0.5) * g, np.float32(1.5) * g, np.float32(0), np.float32(-0.0)], np.float32)
        arr = np.concatenate([arr, -arr])
        a, b = tern_rint(arr, g), tern_thresh(arr, T)
        edge["flips"] += int((a != b).sum()); edge["n"] += int(arr.size)
        edge["flips"] += int((tern_thresh_abs(arr, T) != a).sum())
    out["edge"] = edge; print("edge", edge, flush=True)
    for shape in SHAPES:
        for kind in KINDS:
            for seed in SEEDS:
                w = make(kind, shape, np.random.default_rng(seed)); g = gamma_of(w); T = cutoff(g)
                ref = tern_rint(w, g)
                r = {"shape": f"{shape[0]}x{shape[1]}", "kind": kind, "seed": seed, "n": int(w.size),
                     "gamma": float(g), "T": float(T), "T_minus_half_g_ulps": int(np.int32(T.view(np.int32)) - np.int32(np.float32(g * np.float32(0.5)).view(np.int32))),
                     "sha16": hashlib.sha256(ref.tobytes()).hexdigest()[:16],
                     "t_gamma": tstat(lambda: gamma_of(w), 7), "t_cutoff_us": tstat(lambda: cutoff(g), 7)["med"] * 1e3, "k": {}}
                for name, f, arg in [("rint", tern_rint, g), ("rint_inplace", tern_rint_inplace, g),
                                     ("thresh", tern_thresh, T), ("thresh_abs", tern_thresh_abs, T)]:
                    q = f(w, arg)
                    r["k"][name] = {"t": tstat(lambda: f(w, arg)), "peak": peak(lambda: f(w, arg)),
                                    "flips": int((q != ref).sum()), "dtype_ok": q.dtype == np.int8}
                b = r["k"]["rint"]["t"]["med"]
                for name in r["k"]: r["k"][name]["speedup"] = b / r["k"][name]["t"]["med"]
                r["e2e_rint"] = tstat(lambda: pack_swar(tern_rint(w, gamma_of(w))), 9)
                r["e2e_thresh"] = tstat(lambda: pack_swar(tern_thresh(w, cutoff(gamma_of(w)))), 9)
                r["e2e_same_bytes"] = bool(pack_swar(tern_rint(w, g)).tobytes() == pack_swar(tern_thresh(w, T)).tobytes())
                out["rows"].append(r)
                print(r["shape"], kind, seed, "T-g/2 ulps", r["T_minus_half_g_ulps"],
                      " ".join("%s %.2f(%.2f-%.2f) %.2fx flips=%d pk=%.1fMB" % (k, v["t"]["med"], v["t"]["min"], v["t"]["max"],
                               v["speedup"], v["flips"], v["peak"] / 1e6) for k, v in r["k"].items()),
                      "gamma %.2f e2e rint %.2f thresh %.2f same=%s" % (r["t_gamma"]["med"], r["e2e_rint"]["med"], r["e2e_thresh"]["med"], r["e2e_same_bytes"]), flush=True)
    json.dump(out, open("results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

if __name__ == "__main__":
    main()
```

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-20_0555_swar-u32-i2s-pack-vs-shiftor-chain.md.

close: 19e9850b382ce032
