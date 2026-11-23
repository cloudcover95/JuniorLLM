---
local: true
date: 2026-11-23
wave: "2026-11-23 05:55 (fired 05:48 MT, ran ~05:49–06:05 MT; no pair reports reached chat this wave, quant builders idle)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (batching amortizes the shared 2-bit weight pass, B 32 -> 128 aggregate tok/s 2.42/1.93x, but the int8 gap only narrows to 1.187/1.274x at B 128, not below 1.1x; bytes unchanged at 3.984x vs int8)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, weights, u32-lut, batching, amortization, b128, numpy, not-viable]
---

# Batch amortization of the shared 2-bit u32-LUT weight pass at B = 8 / 32 / 64 / 128: aggregate tok/s rises 2.42/1.93x from B = 32 to 128 and matmul grows to ~70% of the pass, but int8 is still 1.187/1.274x faster at B = 128, so the 2-bit/int8 gap never drops below 1.1x: not viable as a speed result; the 3.98x weight-memory win is unchanged

The 02:55 note ([[2026-11-23_0255_k-blocked-u32-lut-2-bit-weight-gemm-per-worker-l2-sized-f32-unpack-scratch-strided-vs-k-block-major-vs-f32-lut-take-vs-row-tile-full-weight-pass-b32-b8]]) found the numpy unpack lever exhausted on the weight side and asked: with one shared 2-bit unpack per step, how does the base256 weight pass split between LUT take, int8->f32 cast and BLAS matmul at B = 32 / 64 / 128, and does growing B from 32 to 128 raise aggregate weight-pass tok/s by >= 1.5x while the 2-bit/int8 time gap drops below 1.1x, at the same 4.00x weight bytes?

Answers:
- **Amortization works:** aggregate weight-pass tok/s for the current 2-bit path goes 1507/1738 (B 32) -> 3643/3351 (B 128), 2.42/1.93x (>= 1.5x met; 7.5/7.9x vs B 8). Pass time grows only 21.2/18.4 -> 35.1/38.2 ms for 4x the streams. This is a batching effect that int8 and f32 share, not a ternary-specific one.
- **Split:** per-worker take/cast/matmul shares go from ~0.41/0.18/0.41 at B 8 to ~0.34/0.16/0.50 at B 32 and ~0.20/0.08/0.71 at B 128. The unpack (take + cast) is a fixed per-step cost that shrinks relative to matmul as B grows.
- **Gap to int8 narrows but stays above 1.1x:** int8 (cast only) over 2-bit is 1.400/1.786x at B 8, 1.444/1.322x at B 32, 1.476/1.166x at B 64 and 1.187/1.274x at B 128. The no-unpack f32-resident control is 1.802/1.623x at B 32 and 1.512/1.512x at B 128. The 1.1x bar is missed in both seeds.
- **Bytes:** resident 2-bit 16.65 MiB vs int8 66.34 MiB (3.984x) vs f32 265.00 MiB (15.916x). At B 128 the 2-bit path gives 3.98x less weight memory than int8 at 0.84/0.78x its speed, and 15.9x less than f32 at 0.66/0.66x. These are the known memory wins, not new ones.
- **Error:** the 2-bit path vs the float64 trit reference max abs <= 2.4e-06, cos 1.00000000 at every B. The int8 control's cos ~0.886 reflects different weights (int8 absmax of the f32 W), as in the 02:55 note.
- **Verdict: not-viable** for the stated question. The amortization half passes, but the 2-bit/int8 gap stays at 1.19–1.27x at B 128, so ternary still pays a speed tax for its memory win in numpy. rowT128 is a small, repeatable gain over T 256 (1.05–1.19x at B >= 32).

## Method

- Weights: base.build() from /workspace/bench1455_1118 (bitnetd layer: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits). par.prep() fuses q|k|v and gate|up into resident 2-bit PR rows with a per-row gamma. par.prep8() is the int8 control. f32res is the float64 trit*gamma reference cast to f32.
- Timed op: one full weight pass of a decode step (qkv 2560->3840, o 2560->2560, gate|up 2560->13824, down 6912->2560) on the W = 8 thread pool, OPENBLAS_NUM_THREADS=1, row tiles split contiguously across workers.
- 5 rounds x 5 reps (2 warm-up) per variant per B. The variant order rotates each round, and the speed ratio (base/this) comes from per-round medians. Two input seeds (XSEED 1, 2) on an 8-core box. The op split uses an instrumented copy of base256 that sums perf_counter time around take, cast and matmul in each worker over 10 passes.
- No installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench0555_1123 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 amort.py run$s.json > run$s.txt 2>&1; done

Script (amort.py):

    #!/usr/bin/env python3
    """Batch amortization of the shared 2-bit u32-LUT weight pass (05:55 11/23).
    One full decode-step weight pass (qkv 2560->3840, o 2560->2560, gate|up 2560->13824, down 6912->2560) on the W=8 pool, at
    B = 8 / 32 / 64 / 128 streams sharing one unpack. Variants (same pool, same row tiling):
      base256  current u32-LUT path (par.fmm, T 256)       rowT128  same, T 128
      int8_256 int8 resident rows cast to f32 (no LUT)    f32res   f32 resident trit*gamma rows, matmul only (no unpack at all)
    Plus an instrumented copy of base256 that sums take / cast / matmul time per worker (op split).
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=... python3 amort.py out.json"""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import D, F
    W = 8; ROUNDS, REPS, WARM = 5, 5, 2
    BS = [8, 32, 64, 128]
    base.build(); WB = par.prep(); WB8 = par.prep8()
    SITES = [("qkv", D), ("o", D), ("gu", D), ("down", F)]
    def wref(key): return np.concatenate([base.PW[n]["Wt"].astype(np.float64) for n in par.FUSE[key]])
    WF = {k: np.ascontiguousarray(wref(k).astype(np.float32)) for k, _ in SITES}; WBF = sum(a.nbytes for a in WF.values())
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT"): base.PW[nm].pop(f, None)
    def run_f(Wf, x, y, r0, r1, T):
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0); np.matmul(x, Wf[o0:o0+n].T, out=y[:, o0:o0+n])
    def fmm_f(key, x, T=256):
        Wf = WF[key]; out = Wf.shape[0]; y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
        for f in [par.pool(W).submit(run_f, Wf, x, y, w*per*T, min(out, (w+1)*per*T), T) for w in range(W) if w*per*T < out]: f.result()
        return y
    OPS = np.zeros((W, 3))
    def run_i(PR, x, y, r0, r1, T, w):
        q = PR.shape[1]; s8, s = par.scratch(w, T, q); pc = time.perf_counter; a = OPS[w]
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0); t0 = pc()
            np.take(par.LUT_U32, PR[o0:o0+n], out=s8[:n]); t1 = pc()
            np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe"); t2 = pc()
            np.matmul(x, s[:n].T, out=y[:, o0:o0+n]); t3 = pc()
            a[0] += t1 - t0; a[1] += t2 - t1; a[2] += t3 - t2
    def fmm_i(key, x, T=256):
        P = par.FP[key]; PR = P["PR"]; out = PR.shape[0]; y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
        for f in [par.pool(W).submit(run_i, PR, x, y, w*per*T, min(out, (w+1)*per*T), T, w) for w in range(W) if w*per*T < out]: f.result()
        y *= P["g"]; return y
    V = {"base256": lambda k, x: np.concatenate(par.fmm(k, x, 256, W), axis=1),
         "rowT128": lambda k, x: np.concatenate(par.fmm(k, x, 128, W), axis=1),
         "int8_256": lambda k, x: np.concatenate(par.fmm8(k, x, 256, W), axis=1),
         "f32res": lambda k, x: fmm_f(k, x)}
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    def main():
        seed = int(os.environ.get("XSEED", "1")); rng = np.random.default_rng(seed)
        out = {"xseed": seed, "w_bytes_2bit": WB, "w_bytes_int8": WB8, "w_bytes_f32": WBF, "B": {}}
        print(f"xseed {seed} | resident 2-bit {WB/2**20:.2f} MiB int8 {WB8/2**20:.2f} MiB ({WB8/WB:.3f}x) f32 {WBF/2**20:.2f} MiB ({WBF/WB:.3f}x) | W {W} rounds {ROUNDS} x reps {REPS}", flush=True)
        for B in BS:
            X = {k: base.rms(rng.standard_normal((B, i)).astype(np.float32)) for k, i in SITES}
            REF = {k: X[k].astype(np.float64) @ wref(k).T for k, _ in SITES}
            Y0 = {k: V["base256"](k, X[k]) for k, _ in SITES}
            res = {}
            for nm, fn in V.items():
                Y = {k: fn(k, X[k]) for k, _ in SITES}
                res[nm] = dict(t=[], mab_vs_base=max(float(np.abs(Y[k]-Y0[k]).max()) for k, _ in SITES),
                               mab_vs_f64=max(float(np.abs(Y[k]-REF[k]).max()) for k, _ in SITES), cos_f64=min(cos(Y[k], REF[k]) for k, _ in SITES))
            names = list(V)
            for rd in range(ROUNDS):
                for nm in names[rd % len(names):] + names[:rd % len(names)]:
                    fn = V[nm]
                    for r in range(WARM + REPS):
                        t0 = time.perf_counter()
                        for k, _ in SITES: fn(k, X[k])
                        dt = (time.perf_counter() - t0) * 1e3
                        if r >= WARM: res[nm]["t"].append(dt)
            bm = [float(np.median(res["base256"]["t"][r*REPS:(r+1)*REPS])) for r in range(ROUNDS)]
            for nm, d in res.items():
                T = np.array(d["t"]); rm = [float(np.median(T[r*REPS:(r+1)*REPS])) for r in range(ROUNDS)]
                gap = [a/b for a, b in zip(bm, rm)]
                d.update(med=float(np.median(T)), lo=float(T.min()), hi=float(T.max()), iqr=float(np.percentile(T, 75)-np.percentile(T, 25)),
                         tps=B*1e3/float(np.median(T)), base_over=float(np.median(res["base256"]["t"]))/float(np.median(T)), bo_lo=min(gap), bo_hi=max(gap), n=len(T))
            # op split, instrumented base256
            ip = []; OPS[:] = 0
            for r in range(WARM + REPS * 2):
                if r == WARM: OPS[:] = 0
                t0 = time.perf_counter()
                for k, _ in SITES: fmm_i(k, X[k])
                if r >= WARM: ip.append((time.perf_counter()-t0)*1e3)
            tot = OPS.sum(0); n_ = REPS * 2
            res["split"] = dict(take_ms=float(tot[0]/n_/W), cast_ms=float(tot[1]/n_/W), mm_ms=float(tot[2]/n_/W), frac=[float(v) for v in tot/tot.sum()], wall_med=float(np.median(ip)))
            for nm in names:
                d = res[nm]
                print(f"B{B:<4d}{nm:9s} {d['med']:7.2f} ms ({d['lo']:.2f}-{d['hi']:.2f}) iqr {d['iqr']:5.2f} | {d['tps']:8.0f} tok/s | base/this {d['base_over']:.3f}x (rounds {d['bo_lo']:.2f}-{d['bo_hi']:.2f}) | mab vs base {d['mab_vs_base']:.1e} vs f64 {d['mab_vs_f64']:.1e} cos {d['cos_f64']:.8f}", flush=True)
            s = res["split"]; print(f"B{B:<4d}split    take {s['take_ms']:.2f} cast {s['cast_ms']:.2f} matmul {s['mm_ms']:.2f} ms/worker  frac {s['frac'][0]:.2f}/{s['frac'][1]:.2f}/{s['frac'][2]:.2f}  wall {s['wall_med']:.2f}", flush=True)
            out["B"][str(B)] = res
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Results (full weight pass, ms median (min–max); seed 1 / seed 2)

| B | variant | seed 1 ms | seed 2 ms | agg tok/s s1 / s2 | base/this (per-round) s1 / s2 | max abs vs f64 | min cos vs f64 |
|---|---|---|---|---|---|---|---|
| 8 | base256 | 16.42 (14.86–22.19) | 18.74 (13.64–42.38) | 487 / 427 | 1.000 (1.00–1.00) / 1.000 (1.00–1.00) | 1.7e-06 | 1.00000000 |
| 8 | rowT128 | 16.27 (14.41–20.89) | 16.07 (13.50–29.01) | 492 / 498 | 1.009 (0.89–1.07) / 1.166 (0.97–1.42) | 1.7e-06 | 1.00000000 |
| 8 | int8_256 | 11.73 (9.72–14.14) | 10.49 (9.19–13.57) | 682 / 762 | 1.400 (1.37–1.60) / 1.786 (1.43–1.95) | 4.1e+00 | 0.88278401 |
| 8 | f32res | 8.90 (7.84–17.42) | 8.23 (7.17–10.21) | 899 / 972 | 1.845 (1.77–1.96) / 2.278 (1.91–2.58) | 1.6e-06 | 1.00000000 |
| 32 | base256 | 21.23 (16.64–27.46) | 18.41 (16.67–21.49) | 1507 / 1738 | 1.000 (1.00–1.00) / 1.000 (1.00–1.00) | 2.4e-06 | 1.00000000 |
| 32 | rowT128 | 18.98 (15.62–24.29) | 17.54 (15.70–20.80) | 1686 / 1824 | 1.119 (1.00–1.37) / 1.050 (0.96–1.13) | 2.4e-06 | 1.00000000 |
| 32 | int8_256 | 14.71 (12.89–19.40) | 13.93 (12.53–18.57) | 2176 / 2297 | 1.444 (1.25–1.56) / 1.322 (1.28–1.35) | 4.4e+00 | 0.88601478 |
| 32 | f32res | 11.78 (9.23–13.24) | 11.35 (9.86–14.71) | 2717 / 2821 | 1.802 (1.63–2.32) / 1.623 (1.46–1.76) | 1.8e-06 | 1.00000000 |
| 64 | base256 | 29.92 (25.66–53.99) | 25.32 (20.69–33.71) | 2139 / 2528 | 1.000 (1.00–1.00) / 1.000 (1.00–1.00) | 2.1e-06 | 1.00000000 |
| 64 | rowT128 | 25.16 (22.33–29.13) | 23.45 (19.11–30.40) | 2543 / 2729 | 1.189 (1.14–1.34) / 1.080 (1.00–1.17) | 2.1e-06 | 1.00000000 |
| 64 | int8_256 | 20.27 (17.11–22.70) | 21.72 (16.33–26.14) | 3157 / 2947 | 1.476 (1.38–1.67) / 1.166 (1.15–1.45) | 4.1e+00 | 0.88594696 |
| 64 | f32res | 15.92 (13.87–22.87) | 16.94 (13.35–20.69) | 4020 / 3778 | 1.879 (1.76–1.95) / 1.494 (1.45–1.64) | 2.2e-06 | 1.00000000 |
| 128 | base256 | 35.14 (28.07–42.42) | 38.19 (32.95–42.82) | 3643 / 3351 | 1.000 (1.00–1.00) / 1.000 (1.00–1.00) | 2.1e-06 | 1.00000000 |
| 128 | rowT128 | 30.53 (26.53–35.07) | 34.44 (31.17–48.99) | 4192 / 3717 | 1.151 (1.05–1.20) / 1.109 (0.88–1.20) | 2.1e-06 | 1.00000000 |
| 128 | int8_256 | 29.59 (23.19–34.49) | 29.97 (25.34–35.07) | 4325 / 4270 | 1.187 (1.01–1.27) / 1.274 (1.13–1.43) | 4.1e+00 | 0.88679593 |
| 128 | f32res | 23.24 (19.02–26.23) | 25.27 (21.20–35.19) | 5508 / 5066 | 1.512 (1.20–1.60) / 1.512 (1.36–1.71) | 2.1e-06 | 1.00000000 |

Variants: base256 = current u32-LUT 2-bit path (T 256); rowT128 = same, row tile 128; int8_256 = int8 resident, cast only (4x bytes); f32res = f32 resident trit*gamma, matmul only (16x bytes).

### base256 op split (per-worker ms per pass, seed 1 / seed 2; fractions take/cast/matmul)

| B | take ms | cast ms | matmul ms | frac s1 | frac s2 |
|---|---|---|---|---|---|
| 8 | 4.92 / 4.34 | 2.16 / 1.81 | 5.00 / 4.29 | 0.41/0.18/0.41 | 0.42/0.17/0.41 |
| 32 | 5.15 / 4.19 | 2.67 / 1.76 | 7.40 / 6.24 | 0.34/0.18/0.49 | 0.34/0.14/0.51 |
| 64 | 5.72 / 4.29 | 3.02 / 1.99 | 10.17 / 8.83 | 0.30/0.16/0.54 | 0.28/0.13/0.58 |
| 128 | 4.53 / 4.12 | 1.90 / 1.70 | 15.16 / 14.73 | 0.21/0.09/0.70 | 0.20/0.08/0.72 |

## Pair files this wave

No Trit Pack or Handshake pair reports reached chat this wave (quant builders idle, roadmap/beta docs exhausted). There are no pair files to link. The 06:00 hwsw lane runs after this note.

## Next question

Batching closes most of the gap; the remaining question is where it ends. At B 256 (matmul share heading past 80%), does the 2-bit/int8 weight-pass gap drop below 1.1x? And at ctx 2048 with A8 g64 KV per stream, at what B does the per-stream cache grow large enough that total bytes (weights + KV) fall below 1.5x under int8, the point where the ternary memory edge disappears?

close: ea4bec2af8ac1057
