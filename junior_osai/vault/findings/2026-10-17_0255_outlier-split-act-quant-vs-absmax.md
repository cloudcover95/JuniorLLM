---
local: true
date: 2026-10-17T02:55:00-06:00
wave: "2026-10-17 02:05"
roadmap_item: "i2sd userspace 127.0.0.1:8767 no docker.sock"
roadmap_ref: "JuniorHome docs/ROADMAP.md #3 (main question, matches Winsor's i2sd_act_accum_bound this wave); side table for #4 Asahi MLX probe only from automations (Quant Chief pair_A this wave)"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnet, i2sd, activation-quant, outlier-split, popcount, bitplane, asahi, page-align, benchmark]
---

# Outlier-channel split vs plain absmax for B=1 ternary popcount matvec

- Prev: [[2026-10-16_1855_winsor-act-quant-vs-absmax]]
- Answers: that note's next question (keep the top |x| channels in fp32 as a side matvec instead of clipping them)
- Roadmap: i2sd userspace 127.0.0.1:8767 no docker.sock (JuniorHome docs/ROADMAP.md #3). Quant Chief gave pair_A #4 Asahi MLX this wave; that item is probe-only and this x86_64 box has no 16 KiB pages or MLX, so it gets a side table (tile-plan padding from Trit Pack's local probe) instead of the main question.

## Question
The 18:55 note showed clipping activation outliers wrecks matvec error, and absmax int8 still misses 1% on heavy tails. If the top 0.5% or 1% |x| channels are split out and run as a dense fp32 side matvec over just those weight columns (gathered from the packed +1/-1 bitmasks), with the rest absmax-quantized to 4, 6 or 8 bits and sent through the popcount bitplane matvec, does int6 reach at most 1% relative error on the outlier-channel and Student-t vectors? What does the select plus gather plus side matvec cost against a fp32 B=1 matvec at 1 and 8 threads? Side question for #4: how much padding does the 16 KiB page-aligned JTR1 tile plan add for 2B4T-like layer shapes?

## Method
- python3, numpy 2.5.4, 8 CPUs, OpenBLAS (scipy-openblas 0.3.34). No install, no model, no GGUF, no network, nothing bound.
- Weights: exact ternary T (p = 0.3 / 0.4 / 0.3), shapes 4096x4096 and 2560x6912 (out x in), seed 20261017. Resident as two packed bitmasks (+1 and -1), like the 14:55 and 18:55 notes.
- Activations: gauss, student_t4, and outlier_ch (Gaussian with 0.5% fixed channels scaled x20), same generators as 18:55.
- Split path: `np.argpartition` picks the top m = ceil(p*C) |x| channels, those are zeroed in the quantized copy, the remainder is absmax-quantized to k bits and run through the popcount matvec, and `gather_cols(P, N, idx) @ x[idx]` adds the fp32 side term. p = 0 is plain absmax. The gathered columns were checked bit-exact against `T[:, idx]` (gather_exact true on both shapes).
- Error: relative L2 against float64 `T @ x` (weights exact, so all error is activation quantization).
- Timing: 1 warm-up plus 31 repeats, median with interquartile range, on student_t4 and outlier_ch. The whole script ran twice, once with `OPENBLAS_NUM_THREADS=1` and once with `=8` (numpy popcount is single-threaded either way). k8 was not timed in this run; the 18:55 note measured 8 planes at about 2x the 4-plane cost.
- Asahi side table: `plan_tiles` from local `/workspace/BitNet-mlx/scripts/asahi_mlx_i2s_page_probe.py` @ 8db848b at page 16384 and 4096, compared with a dense single JTR1 frame (16 B header + 2 bits per trit).
- Bar applied: viable only if the ternary path is at least 1.5x faster than fp32 at the same B, or at least 1.5x smaller resident with latency within 1.25x of fp32, at rel err of 1% or less.
- Peak RSS 340.1 MB per process.

Command: `cd /workspace/tbench && OPENBLAS_NUM_THREADS=1 python3 outlier_split_bench.py 31 > r0255/threads1.json && OPENBLAS_NUM_THREADS=8 python3 outlier_split_bench.py 31 > r0255/threads8.json`

```python
#!/usr/bin/env python3
"""Outlier-channel split vs plain absmax activation quant for B=1 ternary matvec (numpy only, offline).
y = T @ x, T in {-1,0,1}. Split: top-p |x| channels go to an fp32 side matvec over just those
weight columns (columns gathered from the packed +1/-1 bitmasks), the rest is absmax-quantized to
k bits and run through the packed-bitplane popcount matvec. Error = rel L2 vs float64 T @ x.
Also: Asahi 16 KiB vs 4 KiB JTR1 tile-plan padding, from BitNet-mlx scripts/asahi_mlx_i2s_page_probe.py."""
import sys, time, json, resource, numpy as np
assert hasattr(np, "bitwise_count")
SEED = 20261017; REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 31
SHAPES = [(4096, 4096), (2560, 6912)]  # (out, in)
DISTS = ["gauss", "student_t4", "outlier_ch"]
KS = (4, 6, 8); PS = (0.0, 0.005, 0.01)  # p = fraction of channels split out (0 = plain absmax)

def make_x(rng, d, C):
    if d == "gauss": return rng.standard_normal(C).astype(np.float32)
    if d == "student_t4": return rng.standard_t(4, C).astype(np.float32)
    x = rng.standard_normal(C).astype(np.float32)          # 0.5% fixed outlier channels x20
    idx = rng.choice(C, max(1, C // 200), replace=False); x[idx] *= 20.0; return x

def pack_mask(m): return np.packbits(m, axis=1, bitorder="little").view(np.uint64)

def quant(x, k):
    qmax = 2 ** (k - 1) - 1; c = float(np.abs(x).max()); s = c / qmax if c > 0 else 1.0
    return np.clip(np.rint(x / s), -qmax, qmax).astype(np.int64), s

def popcount_matvec(P, N, rowsum, q, k, w8):
    half = 2 ** (k - 1); u = (q + half).astype(np.uint8)
    bits = ((u[None, :] >> np.arange(k, dtype=np.uint8)[:, None]) & 1).astype(bool)
    U = np.packbits(bits, axis=1, bitorder="little").view(np.uint64)
    cp = np.bitwise_count(P[:, None, :] & U[None, :, :]).sum(axis=2, dtype=np.int64)
    cn = np.bitwise_count(N[:, None, :] & U[None, :, :]).sum(axis=2, dtype=np.int64)
    return (cp - cn) @ w8[:k] - half * rowsum

def gather_cols(P, N, idx):  # ternary columns idx from packed masks -> float32 (R, m)
    w, b = idx // 64, (idx % 64).astype(np.uint64)
    return (((P[:, w] >> b) & 1).astype(np.float32) - ((N[:, w] >> b) & 1).astype(np.float32))

def split_matvec(P, N, rowsum, x, k, p, w8):
    if p == 0.0:
        q, s = quant(x, k); return popcount_matvec(P, N, rowsum, q, k, w8) * s
    m = max(1, int(np.ceil(p * x.size)))
    idx = np.argpartition(np.abs(x), -m)[-m:]
    xi = x.copy(); xs = x[idx]; xi[idx] = 0.0
    q, s = quant(xi, k)
    return popcount_matvec(P, N, rowsum, q, k, w8) * s + gather_cols(P, N, idx) @ xs

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
    sh = {"shape": f"{R}x{C}", "fp32_w_bytes": Wf.nbytes, "mask_bytes": P.nbytes + N.nbytes, "err": {}, "timing": {}}
    xs = {d: make_x(rng, d, C) for d in DISTS}
    idx_chk = rng.choice(C, 37, replace=False)
    sh["gather_exact"] = bool(np.array_equal(gather_cols(P, N, idx_chk), Wf[:, idx_chk]))
    for d, x in xs.items():
        ref = Td @ x.astype(np.float64); nr = np.linalg.norm(ref)
        for p in PS:
            for k in KS:
                y = split_matvec(P, N, rowsum, x, k, p, w8)
                sh["err"][f"{d}|p{p}|k{k}"] = float(np.linalg.norm(y - ref) / nr)
    tm = sh["timing"]
    for d in ("student_t4", "outlier_ch"):
        x = xs[d]
        tm[f"{d}|fp32_matvec"] = med(lambda: Wf @ x, REPS)
        for p in PS:
            for k in (4, 6):
                tm[f"{d}|split_p{p}_k{k}"] = med(lambda: split_matvec(P, N, rowsum, x, k, p, w8), REPS)
        m = int(np.ceil(0.005 * C)); idx = np.argpartition(np.abs(x), -m)[-m:]
        tm[f"{d}|overhead_p0.005_select"] = med(lambda: np.argpartition(np.abs(x), -m)[-m:], REPS)
        tm[f"{d}|overhead_p0.005_gather_side"] = med(lambda: gather_cols(P, N, idx) @ x[idx], REPS)
    rows.append(sh); del T, Wf, Td, P, N

# Asahi item: padding of the 16 KiB / 4 KiB page-aligned JTR1 tile plan (Trit Pack's probe, local copy)
pad = {}
try:
    sys.path.insert(0, "/workspace/BitNet-mlx/scripts"); import asahi_mlx_i2s_page_probe as ap
    for R, C in [(4096, 4096), (2560, 6912), (2560, 2560), (640, 2560)]:
        n = R * C; dense = 16 + (n * 2 + 7) // 8
        for page in (ap.PAGE_16K, ap.PAGE_4K):
            pl = ap.plan_tiles(n, page)
            tot = pl["tile_bytes"] * (pl["tiles"] - 1) + pl["last_tile_bytes"]
            pad[f"{R}x{C}|{page}"] = {"tiles": pl["tiles"], "pages_per_tile": pl["pages_per_tile"],
                                      "plan_bytes": tot, "dense_jtr1_bytes": dense, "overhead": round(tot / dense - 1, 6),
                                      "fp16_bytes": n * 2, "fp32_bytes": n * 4}
except Exception as e:
    pad = {"error": repr(e)}
print(json.dumps({"reps": REPS, "seed": SEED, "numpy": np.__version__,
                  "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                  "rows": rows, "asahi_tile_padding": pad}, indent=1))
```

## Results

### Relative L2 error (p = fraction of channels split to fp32; p0.0 = plain absmax)

| shape | x | p0 k4 | p0 k6 | p0 k8 | p0.5% k4 | p0.5% k6 | p0.5% k8 | p1% k4 | p1% k6 | p1% k8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | gauss | 14.74% | 3.36% | 0.83% | 11.94% | 2.66% | 0.63% | 10.84% | 2.49% | 0.57% |
| 4096x4096 | student_t4 | 45.97% | 11.12% | 2.66% | 15.64% | 3.67% | 0.86% | 13.47% | 3.10% | 0.74% |
| 4096x4096 | outlier_ch | 60.16% | 25.07% | 6.28% | 8.34% | 1.85% | 0.47% | 6.65% | 1.51% | 0.37% |
| 2560x6912 | gauss | 16.38% | 3.75% | 0.95% | 11.98% | 2.63% | 0.66% | 10.72% | 2.35% | 0.60% |
| 2560x6912 | student_t4 | 36.38% | 8.30% | 2.03% | 16.84% | 3.86% | 0.95% | 13.58% | 3.07% | 0.75% |
| 2560x6912 | outlier_ch | 54.65% | 23.97% | 6.01% | 7.46% | 1.70% | 0.41% | 6.13% | 1.43% | 0.34% |

### Latency, ms: median (IQR), and ratio to fp32 at the same thread count

| shape | x | path | 1 thread | vs fp32 | 8 threads | vs fp32 |
|---|---|---|---|---|---|---|
| 4096x4096 | student_t4 | fp32_matvec | 2.002 (1.971-2.128) | 1.00x | 0.305 (0.297-0.325) | 1.00x |
| 4096x4096 | student_t4 | split_p0.0_k4 | 3.875 (3.552-4.322) | 1.93x | 3.682 (3.548-3.857) | 12.05x |
| 4096x4096 | student_t4 | split_p0.0_k6 | 5.519 (5.159-5.797) | 2.76x | 5.555 (5.324-5.949) | 18.18x |
| 4096x4096 | student_t4 | split_p0.005_k4 | 4.721 (4.419-5.002) | 2.36x | 4.685 (4.488-5.075) | 15.34x |
| 4096x4096 | student_t4 | split_p0.005_k6 | 6.365 (6.168-6.788) | 3.18x | 6.590 (6.186-7.046) | 21.57x |
| 4096x4096 | student_t4 | split_p0.01_k4 | 5.453 (5.230-5.801) | 2.72x | 5.426 (5.197-5.824) | 17.76x |
| 4096x4096 | student_t4 | split_p0.01_k6 | 7.292 (7.047-7.614) | 3.64x | 7.253 (7.009-7.550) | 23.74x |
| 4096x4096 | student_t4 | overhead_p0.005_select | 0.008 (0.007-0.008) | 0.00x | 0.009 (0.009-0.009) | 0.03x |
| 4096x4096 | student_t4 | overhead_p0.005_gather_side | 0.948 (0.926-0.965) | 0.47x | 0.893 (0.863-0.930) | 2.92x |
| 4096x4096 | outlier_ch | fp32_matvec | 2.096 (1.982-2.215) | 1.00x | 0.278 (0.276-0.286) | 1.00x |
| 4096x4096 | outlier_ch | split_p0.0_k4 | 3.732 (3.501-3.857) | 1.78x | 3.774 (3.651-4.034) | 13.58x |
| 4096x4096 | outlier_ch | split_p0.0_k6 | 5.591 (5.439-5.998) | 2.67x | 5.615 (5.388-5.927) | 20.21x |
| 4096x4096 | outlier_ch | split_p0.005_k4 | 4.648 (4.381-5.009) | 2.22x | 4.665 (4.479-4.815) | 16.79x |
| 4096x4096 | outlier_ch | split_p0.005_k6 | 6.491 (6.219-6.829) | 3.10x | 6.560 (6.269-6.869) | 23.61x |
| 4096x4096 | outlier_ch | split_p0.01_k4 | 5.360 (5.142-5.737) | 2.56x | 5.509 (5.368-5.887) | 19.82x |
| 4096x4096 | outlier_ch | split_p0.01_k6 | 7.136 (6.944-7.735) | 3.40x | 7.219 (7.108-7.775) | 25.98x |
| 4096x4096 | outlier_ch | overhead_p0.005_select | 0.011 (0.011-0.011) | 0.01x | 0.009 (0.009-0.009) | 0.03x |
| 4096x4096 | outlier_ch | overhead_p0.005_gather_side | 0.813 (0.802-0.823) | 0.39x | 0.830 (0.826-0.847) | 2.98x |
| 2560x6912 | student_t4 | fp32_matvec | 2.430 (2.349-2.546) | 1.00x | 0.310 (0.306-0.322) | 1.00x |
| 2560x6912 | student_t4 | split_p0.0_k4 | 3.600 (3.397-3.893) | 1.48x | 3.639 (3.384-3.790) | 11.74x |
| 2560x6912 | student_t4 | split_p0.0_k6 | 5.240 (5.105-5.596) | 2.16x | 5.117 (4.905-5.472) | 16.51x |
| 2560x6912 | student_t4 | split_p0.005_k4 | 4.209 (4.051-4.579) | 1.73x | 4.294 (4.126-4.416) | 13.85x |
| 2560x6912 | student_t4 | split_p0.005_k6 | 6.029 (5.746-6.406) | 2.48x | 5.787 (5.665-6.139) | 18.67x |
| 2560x6912 | student_t4 | split_p0.01_k4 | 4.984 (4.700-5.310) | 2.05x | 4.738 (4.597-5.022) | 15.28x |
| 2560x6912 | student_t4 | split_p0.01_k6 | 6.894 (6.437-7.132) | 2.84x | 6.510 (6.279-7.058) | 21.00x |
| 2560x6912 | student_t4 | overhead_p0.005_select | 0.014 (0.014-0.014) | 0.01x | 0.011 (0.011-0.011) | 0.04x |
| 2560x6912 | student_t4 | overhead_p0.005_gather_side | 0.621 (0.573-0.643) | 0.26x | 0.533 (0.517-0.578) | 1.72x |
| 2560x6912 | outlier_ch | fp32_matvec | 2.360 (2.317-2.484) | 1.00x | 0.334 (0.308-0.366) | 1.00x |
| 2560x6912 | outlier_ch | split_p0.0_k4 | 3.822 (3.700-4.019) | 1.62x | 4.488 (4.209-4.856) | 13.44x |
| 2560x6912 | outlier_ch | split_p0.0_k6 | 5.440 (5.268-5.753) | 2.31x | 5.873 (5.448-6.078) | 17.58x |
| 2560x6912 | outlier_ch | split_p0.005_k4 | 4.372 (4.084-4.662) | 1.85x | 4.378 (4.179-4.649) | 13.11x |
| 2560x6912 | outlier_ch | split_p0.005_k6 | 5.994 (5.856-6.550) | 2.54x | 6.207 (5.732-6.566) | 18.58x |
| 2560x6912 | outlier_ch | split_p0.01_k4 | 4.827 (4.657-5.157) | 2.05x | 5.039 (4.739-6.528) | 15.09x |
| 2560x6912 | outlier_ch | split_p0.01_k6 | 6.745 (6.466-7.146) | 2.86x | 6.274 (6.168-6.768) | 18.78x |
| 2560x6912 | outlier_ch | overhead_p0.005_select | 0.012 (0.012-0.012) | 0.00x | 0.012 (0.012-0.012) | 0.04x |
| 2560x6912 | outlier_ch | overhead_p0.005_gather_side | 0.534 (0.524-0.557) | 0.23x | 0.519 (0.516-0.529) | 1.55x |

### Asahi #4 side table: page-aligned JTR1 tile plan padding

| layer | page B | tiles | plan bytes | dense JTR1 bytes | padding | fp16 / plan | fp32 / plan |
|---|---|---|---|---|---|---|---|
| 4096x4096 | 16384 | 257 | 4,210,688 | 4,194,320 | 0.39% | 7.97x | 15.94x |
| 4096x4096 | 4096 | 1029 | 4,214,784 | 4,194,320 | 0.49% | 7.96x | 15.92x |
| 2560x6912 | 16384 | 271 | 4,440,064 | 4,423,696 | 0.37% | 7.97x | 15.94x |
| 2560x6912 | 4096 | 1085 | 4,444,160 | 4,423,696 | 0.46% | 7.96x | 15.93x |
| 2560x2560 | 16384 | 101 | 1,654,784 | 1,638,416 | 1.00% | 7.92x | 15.84x |
| 2560x2560 | 4096 | 402 | 1,646,592 | 1,638,416 | 0.50% | 7.96x | 15.92x |
| 640x2560 | 16384 | 26 | 425,984 | 409,616 | 4.00% | 7.69x | 15.38x |
| 640x2560 | 4096 | 101 | 413,696 | 409,616 | 1.00% | 7.92x | 15.84x |

Bytes per layer: fp32 weights 67,108,864 (4096x4096) and 70,778,880 (2560x6912) vs bitmasks 4,194,304 and 4,423,680, 16.0x smaller, unchanged from earlier notes.

Raw JSON: `/workspace/tbench/r0255/threads1.json` and `threads8.json` (box-local, not pushed).

## Verdict
**Not viable on speed; the split is a real accuracy fix.** Splitting out 0.5% of channels cuts error sharply on heavy tails: outlier_ch int6 goes from 25.1% to 1.86% (4096x4096) and 24.0% to 1.70% (2560x6912), and int8 goes from 6.3% and 6.0% to 0.47% and 0.41%. Student-t int8 drops from 2.7% and 2.0% to 0.86% and 0.95%. So int8 plus a 0.5% split is under 1% on every vector tested, which plain absmax int8 never was. Int6 does not reach 1% at either split size (best is 1.43% at p = 1% on outlier_ch, and 2.35% to 3.9% on gauss and Student-t), so the 18:55 hope of int6 at 1% fails.

The cost is the gather. Selecting the channels is cheap (0.008 to 0.014 ms), but gathering the side columns out of the packed masks plus the side matvec takes 0.52 to 0.95 ms, which is 23% to 47% of a whole 1-thread fp32 matvec, and it adds about 0.55 to 0.9 ms to every split run. The split int6 path is 2.5x to 3.2x slower than 1-thread fp32 BLAS and 19x to 24x slower than 8-thread BLAS. Nothing here is within 1.5x of fp32 on speed, so the memory win (16x on weights) still comes with a large latency loss in numpy.

For i2sd: if activations go through an int path, use int8 with a 0.5% fp32 outlier split rather than clipping, and keep the outlier columns in a small separate fp32 or int8 column-major side store instead of gathering them from the row-packed masks per token. For #4 Asahi: the 16 KiB tile plan costs 0.37% to 1.0% padding on the large shapes and 4.0% on 640x2560 (inside the probe's 0.05 rate, but 4x the 4 KiB plan's 1.0%), and stays 7.7x to 8.0x smaller than fp16, so page alignment is not the bottleneck.

## Ops check
`python3 scripts/terraform_lean_prod.py code` and `... llm` in local `/workspace/JuniorLLM` (offline, blender off): both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false`, `"model_pull": false`, `"launch": false`.

## This wave's pair files (2026-10-17 02:05)
Quant Chief last_assignment: pair_A Asahi MLX probe only from automations; pair_B Live beta (juniorctl health/ask/night/security + test_junior_aie + test_linux_security). LLC skip path: JuniorHome `web3node/advance_ue5.py`.
- Trit Pack: no new commit on GitHub or locally as of 02:58 MT (BitNet-mlx tip is 962f8e3, the 18:24 MT i2sd policy key from the last wave, which has now landed)
- Winsor: [JuniorQuant src/i2sd_act_accum_bound.py @ 7847e0b](https://github.com/cloudcover95/JuniorQuant/blob/7847e0b/src/i2sd_act_accum_bound.py) (#3 i2sd, off pair_A's Asahi assignment)
- Side-Commit: [stocksnode src/stocksnode/core/live_beta_receipt_ledger.py @ 0eaa9a6](https://github.com/cloudcover95/stocksnode/blob/0eaa9a6/src/stocksnode/core/live_beta_receipt_ledger.py) (Live beta, matches pair_B)
- Handshake & Ports: [JuniorLLM ports/live_beta_ctl_port_map.py @ b3da3b5](https://github.com/cloudcover95/JuniorLLM/blob/b3da3b5/ports/live_beta_ctl_port_map.py) (Live beta, matches pair_B)

## Next question
Side store for the outlier columns: if i2sd keeps a small column-major int8 copy of the ternary weights for the channels that are outliers across a calibration set (fixed channel set, about 0.5% to 1% of columns), how much does that cut the 0.5 to 0.95 ms gather cost, what extra bytes does it add per layer, and how much error comes back when the per-token outlier set drifts away from the fixed calibrated set?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-16_1855_winsor-act-quant-vs-absmax.md.

close: bb270b8bdd4f39f4
