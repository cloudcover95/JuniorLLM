---
local: true
date: 2026-11-24
wave: "2026-11-24 02:55 slot (fired ~02:49 MT, ran ~02:50–03:05 MT; pair reports for this wave were not in hand; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (at B 1 GEMV the u32-LUT take alone is 55-58% of 2-bit thread-time and costs about as much as the whole int8 cast+dot; smaller tiles T 64/128 and a fused take into a pre-cast f32 LUT both leave all-2-bit at 1.94-2.97x of int8 GEMV, far from the 1.3x bar; weight memory stays 3.984x, already recorded)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, b1, gemv, per-op-split, tile-size, f32-lut, u32-lut, int8, thread-pool, numpy, not-viable]
---

# B 1 GEMV per-op split: the u32-LUT take is 55–65% of 2-bit work and alone matches int8's whole tile cost; T 64/128 tiles and a fused f32-LUT take both make 2-bit slower, so the gap stays 1.94–1.96x at best

The 18:55 note ([[2026-11-23_1855_b1-row-major-gemv-np-dot-tile-x-vs-np-matmul-x-tile-t-in-8-thread-pool-full-bitnetd-layer-2-bit-u32-lut-vs-int8-gap-widens]]) moved B 1 decode to a row-major GEMV (np.dot(tile, x[0])) and found all-2-bit ~1.95x slower than all-int8. It asked: at B 1 with GEMV on the 8-thread pool, what is the per-op split (LUT take / int8->f32 cast / dot) for all-2-bit, and does (a) a smaller row tile T 64/128 (vs 256) keeping each worker's u32 + f32 scratch L2-resident or (b) a fused take into a pre-cast f32 LUT (4 f32 per packed byte, no int8 view cast) bring all-2-bit GEMV within 1.3x of all-int8 GEMV?

Answers:
- **Per-op split (all-2-bit, T 256, thread-ms per layer summed over 8 workers, seed 1 / seed 2):** take 36.4 / 30.2 (**57.5 / 54.8%**), cast 15.3 / 14.1 (**24.1 / 25.6%**), dot 11.6 / 10.8 (**18.3 / 19.7%**). For int8 T 256: cast 17.1 / 15.0 (57 / 58%), dot 12.9 / 10.7 (43 / 42%), 30.0 / 25.8 thread-ms in total. **The 2-bit take alone costs about as much as int8's whole cast+dot.** So even a free cast would leave 2-bit at take+dot ≈ 42–48 vs 26–30 thread-ms, ≥ 1.6x.
- **(a) Smaller tiles: no.** All-2-bit full layer is 14.34/14.44 ms at T 64, 10.36/10.85 at T 128 and 10.20/10.87 at T 256. Take thread-time grows as T shrinks (30–36 -> 42–46 -> 71–74 ms at T 256/128/64) because numpy's per-call overhead outweighs any L2 gain. At in 6912, T 128 scratch is still 4.22 MiB/worker vs 2 MiB L2, and T 64 is 2.11 MiB. Int8 gains from T 128 (4.83/4.88 ms, 0.92/0.88x of its T 256), so **the gap vs the best int8 widens to 2.11–2.23x**.
- **(b) Fused f32-LUT take: no, slower.** One 16-byte gather per packed byte (256 x 4 f32 LUT viewed as complex128) removes the cast, but the take then costs 66–108 thread-ms, more than u32 take + cast (44–52). Full layer is 13.42/14.04 ms at T 256 and 12.23/12.11 ms at its best tile (T 128), which is 2.18–2.33x of int8 T 256 and 2.48–2.53x of int8 T 128.
- **Best 2-bit vs int8 GEMV:** all2_T256 / all8_T256 = **1.943 / 1.956x** (rounds 1.65–2.22). All2 best vs int8 best (T 128) = 2.11–2.15x / 2.22–2.23x. **Bar 1.3x: not met.**
- **Memory:** unchanged. Layer weights are 17,458,176 B (16.65 MiB) at 2-bit vs 69,559,296 B (66.34 MiB) at int8 (**3.984x**). The f32 LUT is 4 KiB vs 1 KiB for u32.
- **Error:** every config matches its own float64 reference. 2-bit max abs 9.6e-07 / 8.5e-07 (identical across T and LUT kind), int8 3.4e-06 / 3.5e-06, cos 1.00000000.
- **Verdict: not-viable.** Neither change closes the speed gap, and the 3.98x weight-memory win was already recorded. The useful finding is format-independent: **int8 B 1 GEMV is 8–12% faster at T 128 than at T 256** on this 8-core box (rounds 0.80–1.01).

## Method

- Weights: base.build() / par.prep() / par.prep8() from /workspace/bench1455_1118 (bitnetd layer: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gate|up / down). The 2-bit path stores resident PR rows (one byte = 4 trits) with per-row gamma. The int8 control is absmax int8 of the f32 W. Same W = 8 ThreadPoolExecutor, per-worker scratch (par.scratch, keyed by worker and T) and full-layer structure as the 18:55 note, GEMV path only (np.dot(s[:n], x[0], out=y[0, o0:o0+n])).
- Configs (T in 64, 128, 256): all2_T = np.take(LUT_U32, PR tile) -> int8 view copyto f32 -> dot; all2f_T = np.take(LUTC, PR tile, out=f32 tile viewed complex128), where LUTC is the 256 x 4 f32 pre-cast LUT viewed as 256 complex128 (bit-identical tile, no cast); all8_T = int8 copyto f32 -> dot.
- Timing: one full layer (rms, qkv, attention stub = v repeated over GQA groups, o, residual, rms, gate|up, silu*up, down, residual) at B 1, OPENBLAS_NUM_THREADS=1, 9 rounds x 11 reps (3 warm-up) per config, config order rotated each round. Reported value is the median of per-round medians with the min–max of round medians as spread. Ratios use the same per-round pairing.
- Split: a separate instrumented pass, perf_counter around take / cast / dot inside each worker, summed over workers (thread-ms per layer), median of 27 layers after 3 warm-up. Instrumented wall time is ~1.1–1.4x the uninstrumented time, so read it as a split, not a timing.
- Two input seeds (XSEED 1, 2), 8-core Xeon box (L2 2 MiB/core). No installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench0255_1124 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 900 python3 split.py run$s.json > run$s.txt 2>&1; done

Script (split.py):

    #!/usr/bin/env python3
    """B=1 GEMV per-op split + tile size + fused f32-LUT take, 02:55 11/24. Full bitnetd-style layer at B = 1 on the W=8 pool (base: bench1855_1123/gemv.py, gv path only).
    all2_T{64,128,256}: u32-LUT take -> int8 view cast -> f32 tile -> np.dot(tile, x[0]).
    all2f_T{64,128,256}: fused take into a pre-cast f32 LUT (256 x 4 f32 viewed as 256 complex128, one 16-byte gather per packed byte, no cast) -> np.dot.
    all8_T{64,128,256}: int8 resident cast to f32 tile -> np.dot.
    Split: separate instrumented pass, per-worker perf_counter around take / cast / dot, summed thread-time per layer."""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import D, F, KVH, QH, HD, rms, silu
    W, ROUNDS, REPS, WARM, TS = 8, 9, 11, 3, (64, 128, 256)
    base.build(); par.prep(); par.prep8()
    SITES = ("qkv", "o", "gu", "down")
    def wf64(k, bits):
        if bits == 2: return np.concatenate([base.PW[n]["Wt"].astype(np.float64) for n in par.FUSE[k]])
        return np.concatenate([base.PW[n]["q8"].astype(np.float64) * float(base.PW[n]["m8"]) for n in par.FUSE[k]])
    WR = {(k, b): wf64(k, b) for k in SITES for b in (2, 8)}
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT", "Wt"): base.PW[nm].pop(f, None)
    LUT = par.LUT_U32
    LUTF = np.ascontiguousarray(LUT.view(np.int8).reshape(256, 4).astype(np.float32))   # pre-cast f32 LUT, 4 f32 per packed byte
    LUTC = LUTF.view(np.complex128).ravel()                                              # same bytes as one 16-byte element per byte value
    ACC = np.zeros((W, 3))   # per-worker take / cast / dot seconds (instrumented pass only)
    def rows2(PR, x, y, r0, r1, w, T, inst):
        q = PR.shape[1]; s8, s = par.scratch(w, T, q); pc = time.perf_counter
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            if inst:
                t0 = pc(); np.take(LUT, PR[o0:o0+n], out=s8[:n]); t1 = pc(); np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe"); t2 = pc()
                np.dot(s[:n], x[0], out=y[0, o0:o0+n]); t3 = pc(); ACC[w] += (t1 - t0, t2 - t1, t3 - t2)
            else:
                np.take(LUT, PR[o0:o0+n], out=s8[:n]); np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe"); np.dot(s[:n], x[0], out=y[0, o0:o0+n])
    def rows2f(PR, x, y, r0, r1, w, T, inst):
        q = PR.shape[1]; _, s = par.scratch(w, T, q); pc = time.perf_counter
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            if inst:
                t0 = pc(); np.take(LUTC, PR[o0:o0+n], out=s[:n].view(np.complex128)); t1 = pc()
                np.dot(s[:n], x[0], out=y[0, o0:o0+n]); t3 = pc(); ACC[w] += (t1 - t0, 0.0, t3 - t1)
            else:
                np.take(LUTC, PR[o0:o0+n], out=s[:n].view(np.complex128)); np.dot(s[:n], x[0], out=y[0, o0:o0+n])
    def rows8(Q, x, y, r0, r1, w, T, inst):
        i = Q.shape[1]; _, s = par.scratch(w, T, i // 4); pc = time.perf_counter
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            if inst:
                t0 = pc(); np.copyto(s[:n], Q[o0:o0+n], casting="unsafe"); t1 = pc(); np.dot(s[:n], x[0], out=y[0, o0:o0+n]); t3 = pc(); ACC[w] += (0.0, t1 - t0, t3 - t1)
            else:
                np.copyto(s[:n], Q[o0:o0+n], casting="unsafe"); np.dot(s[:n], x[0], out=y[0, o0:o0+n])
    def fmm(key, x, kind, T, inst):
        bits = 8 if kind == "8" else 2
        P = par.FP[key] if bits == 2 else par.F8[key]; R = P["PR"] if bits == 2 else P["Q"]; out = R.shape[0]
        fn = {"2": rows2, "2f": rows2f, "8": rows8}[kind]
        x = np.ascontiguousarray(x, np.float32); y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
        futs = [par.pool(W).submit(fn, R, x, y, w * per * T, min(out, (w + 1) * per * T), w, T, inst) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
        y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
    def layer(x, kind, T, inst=False):
        mm = lambda k, a: fmm(k, a, kind, T, inst)
        h1 = rms(x); q, kk, v = mm("qkv", h1)
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        o = mm("o", att)[0]; x2 = x + o; h2 = rms(x2)
        g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
        return x2 + mm("down", z)[0]
    def layer_ref(x, bits):
        x = x.astype(np.float64); r = lambda a: a / np.sqrt(np.mean(a ** 2, axis=-1, keepdims=True) + 1e-6); sg = lambda a: a / (1 + np.exp(-a))
        qkv = r(x) @ WR[("qkv", bits)].T; v = qkv[:, D + KVH * HD:]
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        x2 = x + att @ WR[("o", bits)].T; gu = r(x2) @ WR[("gu", bits)].T; z = sg(gu[:, :F]) * gu[:, F:]
        return x2 + z @ WR[("down", bits)].T
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    CFG = {f"all{k}_T{T}": (k, T) for k in ("2", "2f", "8") for T in TS}
    def main():
        seed = int(os.environ.get("XSEED", "1")); rng = np.random.default_rng(seed)
        res = {"xseed": seed}
        W2 = sum(par.FP[k]["PR"].nbytes + par.FP[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        res["wbytes"] = dict(all2=W2, all8=W8, lutf=int(LUTF.nbytes), lut32=int(LUT.nbytes))
        print(f"layer weights bytes all2 {W2} ({W2/2**20:.2f} MiB) all8 {W8} ({W8/2**20:.2f} MiB) ({W8/W2:.3f}x); LUT u32 {LUT.nbytes} B, f32 LUT {LUTF.nbytes} B", flush=True)
        for T in TS: print(f"scratch per worker T{T}: in{F} u32 {T*F:,} B + f32 {T*F*4:,} B = {T*F*5/2**20:.2f} MiB; in{D} {T*D*5/2**20:.2f} MiB (L2 2 MiB/core)", flush=True)
        x = (rng.standard_normal((1, D)) * 0.5).astype(np.float32); R = {b: layer_ref(x, b) for b in (2, 8)}
        d = {}
        for c, (k, T) in CFG.items():
            y = layer(x, k, T); b = 8 if k == "8" else 2; d[c] = dict(t=[], mab=float(np.abs(y - R[b]).max()), cos=cos(y, R[b]))
        names = list(CFG)
        for rd in range(ROUNDS):
            for c in names[rd % len(names):] + names[:rd % len(names)]:
                for r in range(WARM + REPS):
                    t0 = time.perf_counter(); layer(x, *CFG[c]); dt = (time.perf_counter() - t0) * 1e3
                    if r >= WARM: d[c]["t"].append(dt)
        rm = {c: [float(np.median(d[c]["t"][r*REPS:(r+1)*REPS])) for r in range(ROUNDS)] for c in names}
        for c in names:
            Tm = np.array(d[c]["t"]); d[c].update(med=float(np.median(rm[c])), rlo=min(rm[c]), rhi=max(rm[c]), lo=float(Tm.min()), hi=float(Tm.max()), rm=rm[c])
            del d[c]["t"]
            print(f"B1 {c:12s} {d[c]['med']:7.3f} ms (round medians {d[c]['rlo']:.3f}-{d[c]['rhi']:.3f}; all {d[c]['lo']:.3f}-{d[c]['hi']:.3f}) | {1e3/d[c]['med']:6.1f} tok/s/layer | mab {d[c]['mab']:.1e} cos {d[c]['cos']:.8f}", flush=True)
        rr = lambda a, b: [p / q for p, q in zip(rm[a], rm[b])]
        S = {}
        pairs = {f"{a}/all8_T256": (a, "all8_T256") for a in names if a != "all8_T256"}
        best8 = min((c for c in names if c.startswith("all8_")), key=lambda c: d[c]["med"])
        for a in names:
            if a.startswith("all2"):
                pairs[f"{a}/{best8} (best int8)"] = (a, best8)
                pairs[f"{a}/all8_T{CFG[a][1]} (same T)"] = (a, f"all8_T{CFG[a][1]}")
        for nm, (a, b) in pairs.items():
            v = rr(a, b); S[nm] = dict(r=d[a]["med"] / d[b]["med"], lo=min(v), hi=max(v))
            print(f"ratio {nm:40s} {S[nm]['r']:.3f}x (rounds {S[nm]['lo']:.2f}-{S[nm]['hi']:.2f})", flush=True)
        res["B1"] = d; res["ratios"] = S; res["best8"] = best8
        # instrumented split: thread-time per op, summed over the 8 workers, per layer
        sp = {}
        for c in names:
            k, T = CFG[c]
            for _ in range(WARM): layer(x, k, T, True)
            per = []
            for r in range(ROUNDS * 3):
                ACC[:] = 0; t0 = time.perf_counter(); layer(x, k, T, True); wall = (time.perf_counter() - t0) * 1e3
                per.append(list(ACC.sum(0) * 1e3) + [wall, float(ACC.sum(1).max() * 1e3)])
            m = np.median(np.array(per), axis=0); tot = m[:3].sum()
            sp[c] = dict(take=m[0], cast=m[1], dot=m[2], wall=m[3], maxworker=m[4], take_pct=100*m[0]/tot, cast_pct=100*m[1]/tot, dot_pct=100*m[2]/tot)
            print(f"split {c:12s} thread-ms/layer take {m[0]:6.2f} ({sp[c]['take_pct']:4.1f}%) cast {m[1]:6.2f} ({sp[c]['cast_pct']:4.1f}%) dot {m[2]:6.2f} ({sp[c]['dot_pct']:4.1f}%) | busiest worker {m[4]:.2f} ms, wall {m[3]:.2f} ms", flush=True)
        res["split"] = sp
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    main()

## Results

Full layer, B 1, W 8, GEMV (ms; median of per-round medians, min–max of round medians):

| config | seed 1 ms | seed 1 rounds | seed 2 ms | seed 2 rounds | tok/s/layer (s1/s2) | max abs vs own f64 |
|---|---|---|---|---|---|---|
| all2_T64 (u32 take + cast) | 14.342 | 13.99–15.72 | 14.436 | 13.81–15.34 | 69.7 / 69.3 | 9.6e-07 / 8.5e-07 |
| all2_T128 | 10.364 | 9.80–10.59 | 10.849 | 10.14–13.72 | 96.5 / 92.2 | 9.6e-07 / 8.5e-07 |
| all2_T256 (18:55 path) | 10.196 | 9.70–10.78 | 10.866 | 9.76–11.78 | 98.1 / 92.0 | 9.6e-07 / 8.5e-07 |
| all2f_T64 (fused f32-LUT take) | 13.839 | 12.96–14.30 | 13.691 | 13.05–14.75 | 72.3 / 73.0 | 9.6e-07 / 8.5e-07 |
| all2f_T128 | 12.225 | 11.58–12.83 | 12.114 | 11.63–14.43 | 81.8 / 82.5 | 9.6e-07 / 8.5e-07 |
| all2f_T256 | 13.419 | 13.08–14.66 | 14.038 | 13.16–14.39 | 74.5 / 71.2 | 9.6e-07 / 8.5e-07 |
| all8_T64 | 6.281 | 6.02–6.77 | 6.145 | 5.84–7.11 | 159.2 / 162.7 | 3.4e-06 / 3.5e-06 |
| all8_T128 | 4.828 | 4.65–5.21 | 4.878 | 4.51–5.44 | 207.1 / 205.0 | 3.4e-06 / 3.5e-06 |
| all8_T256 (18:55 path) | 5.247 | 5.13–5.97 | 5.554 | 5.14–5.90 | 190.6 / 180.0 | 3.4e-06 / 3.5e-06 |

Ratios vs int8 GEMV (seed 1 / seed 2, per-round range; bar <= 1.3x):

| 2-bit config | / all8_T256 | / all8_T128 (best int8) | / all8 same T |
|---|---|---|---|
| all2_T64 | 2.734x (2.53–2.91) / 2.599x (2.42–2.81) | 2.971x / 2.960x | 2.283x / 2.349x |
| all2_T128 | 1.975x (1.73–2.03) / 1.953x (1.79–2.55) | 2.147x / 2.224x | 2.147x / 2.224x |
| all2_T256 | **1.943x (1.65–2.05) / 1.956x (1.81–2.22)** | 2.112x / 2.228x | 1.943x / 1.956x |
| all2f_T64 | 2.638x (2.28–2.72) / 2.465x (2.26–2.64) | 2.866x / 2.807x | 2.203x / 2.228x |
| all2f_T128 | 2.330x (2.03–2.49) / 2.181x (2.03–2.68) | 2.532x / 2.484x | 2.532x / 2.484x |
| all2f_T256 | 2.558x (2.21–2.75) / 2.528x (2.29–2.80) | 2.779x / 2.878x | 2.558x / 2.528x |
| (int8 T64 / T128 vs T256) | 1.197x / 1.106x; 0.920x / 0.878x | — | — |

Per-op split (instrumented pass; thread-ms per layer summed over 8 workers, % of take+cast+dot; seed 1 / seed 2):

| config | take | cast | dot | busiest worker ms |
|---|---|---|---|---|
| all2_T64 | 70.85 (64.2%) / 73.61 (65.4%) | 22.47 (20.4%) / 22.06 (19.6%) | 17.07 (15.5%) / 16.85 (15.0%) | 14.38 / 14.66 |
| all2_T128 | 46.37 (60.9%) / 41.99 (60.8%) | 17.56 (23.1%) / 15.84 (22.9%) | 12.19 (16.0%) / 11.27 (16.3%) | 10.70 / 9.71 |
| all2_T256 | 36.43 (57.5%) / 30.21 (54.8%) | 15.28 (24.1%) / 14.11 (25.6%) | 11.61 (18.3%) / 10.84 (19.7%) | 10.12 / 8.88 |
| all2f_T64 | 108.04 (83.1%) / 83.84 (83.2%) | — | 21.92 / 16.92 | 17.07 / 13.50 |
| all2f_T128 | 85.87 (85.3%) / 75.30 (85.2%) | — | 14.85 / 13.10 | 14.88 / 12.61 |
| all2f_T256 | 73.44 (84.9%) / 66.37 (84.8%) | — | 13.10 / 11.88 | 13.84 / 12.40 |
| all8_T64 | — | 33.55 (54.6%) / 26.47 (58.0%) | 27.91 / 19.17 | 8.12 / 6.10 |
| all8_T128 | — | 22.82 (56.3%) / 18.28 (59.7%) | 17.73 / 12.35 | 5.98 / 4.47 |
| all8_T256 | — | 17.06 (56.9%) / 15.04 (58.4%) | 12.90 / 10.73 | 5.13 / 4.41 |

Per-worker scratch at the widest input (in 6912): T 64 = 2.11 MiB, T 128 = 4.22 MiB, T 256 = 8.44 MiB (u32 + f32). At in 2560: 0.78 / 1.56 / 3.12 MiB. L2 is 2 MiB per core.

Weights per layer: 2-bit 17,458,176 B (16.65 MiB), int8 69,559,296 B (66.34 MiB), 3.984x. LUT: u32 1,024 B, pre-cast f32 4,096 B.

## Pair files this wave

This wave's pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) were not available to this lane, and none are linked. The newest JuniorHome ledger with a local copy is docs/ledger/2026-11-22_1805.md (/workspace/JuniorHome-ledger-20261122-1805, close 458388ccc92e0af5, "none open; blobs exhausted"). The 18:55 note cited the 2026-11-23_1405 ledger. No newer ledger is in hand. JuniorLLM's last commit before this one was 52982be (this lane's 18:55 note); no Trit Pack, Handshake or LLC commits fell in the last 6 hours.

## Next question

The u32 take is 55–65% of 2-bit B 1 work and alone costs about as much as int8's whole tile, and expanding to f32 in any form (cast or 16-byte gather) can't beat it in numpy. At B 1 on the 8-thread pool, does a T-MAC-style activation LUT avoid weight expansion entirely and bring all-2-bit within 1.3x of all8_T128 GEMV at the same 3.98x weight bytes? The LUT is built once per token: for each group of 4 inputs, a 256-entry f32 table of partial sums over every packed-byte trit pattern (in/4 x 256 x 4 B = 640 KiB at in 2560, 1.69 MiB at in 6912, L2-resident). Each row is then y[row] = sum over g of Tab[g, PR[row, g]], a take of out x in/4 f32 values plus a row sum, with no f32 weight tile and no dot.

close: e5668f7427243bee
