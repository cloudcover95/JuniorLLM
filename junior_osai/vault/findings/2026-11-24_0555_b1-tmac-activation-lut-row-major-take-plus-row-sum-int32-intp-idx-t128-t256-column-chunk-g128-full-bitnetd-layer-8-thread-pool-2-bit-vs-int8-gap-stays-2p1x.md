---
local: true
date: 2026-11-24
wave: "2026-11-24 05:55 slot (fired ~05:43 MT, ran ~05:43–06:00 MT; pair reports for this wave were not in hand; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (a T-MAC activation LUT on row-major 2-bit rows on the W 8 pool removes the f32 weight tile, cast and dot, but its out x in/4 f32 gather from a 0.64-1.69 MiB table plus the per-tile index add costs as much as the u32 take + cast + dot it replaces: best tm64_T256 is 1.01-1.02x of the 02:55 u32 path and 2.12-2.21x of all8_T128 GEMV, far from the 1.3x bar; weight memory stays 3.984x, already recorded)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, b1, gemv, t-mac, activation-lut, per-op-split, tile-size, u32-lut, int8, thread-pool, numpy, not-viable]
---

# B 1 T-MAC activation LUT on the 8-thread pool: no weight expansion, but the gather from a 0.64–1.69 MiB table plus the index add costs as much as the u32 take + cast + dot, so all-2-bit stays 2.12–2.21x of int8 T 128 GEMV

The 02:55 note ([[2026-11-24_0255_b1-gemv-per-op-split-u32-lut-take-cast-dot-tile-t64-t128-t256-vs-fused-pre-cast-f32-lut-take-full-bitnetd-layer-8-thread-pool-2-bit-vs-int8-gap-stays-1p94x]]) found the u32-LUT take is 55–65% of 2-bit B 1 GEMV work, and that expanding to f32 in any form (cast or a 16-byte gather) can't beat it in numpy. It asked: at B 1 on the 8-thread pool, does a T-MAC-style activation LUT avoid weight expansion entirely and bring all-2-bit within 1.3x of all8_T128 GEMV at the same 3.98x weight bytes? The LUT is built once per token per site input: for each group of 4 inputs, a 256-entry f32 table of partial sums over every packed-byte trit pattern. Each row is then y[row] = sum over g of Tab[g, PR[row, g]], a take of out x in/4 f32 plus a row sum, with no f32 weight tile and no dot.

This is not a repeat of 2026-11-11_1855 / 2026-11-12_0255 / 2026-11-12_0555. Those tested T-MAC single-threaded on the transposed PT pack (tmac_g8, G 8 column chunks, uint16 offsets, int16 table) against blk32 / int8 controls of that era. This note runs it on the row-major PR rows, the W 8 pool and the 02:55 GEMV baselines (all8_T128 / all8_T256 / all2_T256) with a per-op split.

Answers:
- **Best T-MAC vs best int8:** tm64_T256 / all8_T128 = **2.213x (rounds 2.01–2.35) / 2.115x (1.80–2.43)** for seed 1 / seed 2. Against all8_T256 it is 2.027 / 1.988x. **Bar 1.3x: not met.**
- **Vs the 02:55 u32 path:** tm64_T256 / all2_T256 = **1.019 / 1.006x** (rounds 0.81–1.14), so it is at parity, not a gain. tm32_T256 is 1.055 / 1.100x, tm32_T128 1.17 / 1.17x and tm64_T128 1.06 / 1.08x. The column-chunked G 128 variant (tmc_T128, 128 KiB L2-resident sub-table) is **2.85 / 2.99x slower** than u32: row-strided chunk slices and 5–14 numpy calls per tile swamp any cache gain.
- **Per-op split (tm64_T256, thread-ms per layer summed over 8 workers, seed 1 / seed 2):** build 0.59 / 0.58 ms (main thread, serial, 4 tables per layer, ~5% of wall). Index add (uint8 PR + g*256 -> intp) 17.99 / 17.27 (**31.3 / 31.0%**). Take 34.95 / 34.13 (**60.7 / 61.3%**). Row sum (np.dot with ones) 4.64 / 4.32 (**8.1 / 7.8%**). That totals **57.6 / 55.7 thread-ms vs 63.7 / 65.0 for u32 all2_T256 and 36.4 / 35.1 for all8_T128**.
- **Why it can't reach 1.3x in numpy:** removing the cast and dot (16 + 12 thread-ms) is paid back by the index add (12–23 thread-ms). The gather count is the same out x in/4 as the u32 take, but it now reads from a 640 KiB–1.69 MiB table instead of 1 KiB, so it is no cheaper per element (34–68 vs 36–37 thread-ms). **Take + sum alone (38.5–39.6 thread-ms) already exceeds int8's whole cast + dot (35–36).** int32 indices are worse at T 128 because np.take converts them to intp internally (tm32 take 46–68 vs tm64 34–47 thread-ms).
- **Memory:** weights unchanged at 17,458,176 B (16.65 MiB) for 2-bit vs 69,559,296 B (66.34 MiB) for int8 (**3.984x**, already recorded, not re-claimed). The activation LUT is 655,360 B at in 2560 and 1,769,472 B at in 6912 (one reused buffer). Per-worker scratch at in 6912 is 1.69 MiB for T 128 int32, 2.53 MiB for T 128 intp, 3.38 MiB for T 256 int32 and 5.06 MiB for T 256 intp.
- **Error:** every T-MAC config matches the float64 ternary reference: max abs **6.8e-07 / 6.4e-07** (tmc 5.3e-07 / 3.2e-07), cos 1.00000000. This is slightly tighter than the u32 path (9.6e-07 / 8.5e-07) because partial sums are formed in groups of 4. Int8 is 3.4e-06 / 3.5e-06 vs its own f64 reference.
- **Verdict: not-viable.** The T-MAC LUT removes weight expansion but not the per-weight-byte gather, which is the cost in numpy. int8 T 128 again beat int8 T 256 (1.092 / 1.064x), confirming the 02:55 tile finding.

## Method

- Weights: base.build() / par.prep() / par.prep8() from /workspace/bench1455_1118 (bitnetd layer: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gate|up / down). The 2-bit path stores resident PR rows (one byte = 4 trits, code 0/1/2 -> 0/+1/-1 at bits 2j) with per-row gamma. The int8 control is absmax int8 of the f32 W. Same W = 8 ThreadPoolExecutor, contiguous row range per worker, full-layer structure and GEMV path as the 02:55 note.
- T-MAC: per site call, Tab = x.reshape(in/4, 4) @ LUTF_T (in/4 x 256 f32), built on the main thread before the workers are dispatched, so build cost is inside every timed layer. Each worker tile does idx = PR tile + g*256 (int32 or intp scratch), then np.take(Tab.ravel(), idx, out=f32 scratch), then np.dot(gathered, ones) for the row sum, and gamma is applied after.
- Configs: tm32_T128 / tm32_T256 (int32 idx), tm64_T128 / tm64_T256 (intp idx), tmc_T128 (G 128 column chunks over a 128 KiB sub-table, intp idx, per-chunk partial sums). Baselines: all2_T256 (02:55 u32 take -> cast -> dot), all8_T128 and all8_T256 (int8 cast -> dot). An f16/int16 table was not rerun: 2026-11-12_0555 already found int16 no faster in numpy, and numpy has no BLAS row sum for f16.
- Timing: one full layer (rms, qkv, attention stub = v repeated over GQA groups, o, residual, rms, gate|up, silu*up, down, residual) at B 1, OPENBLAS_NUM_THREADS=1, 9 rounds x 11 reps (3 warm-up) per config, config order rotated each round. Reported value is the median of per-round medians with the min–max of round medians as spread. Ratios use the same per-round pairing.
- Split: a separate instrumented pass. It times main-thread build ms and per-worker perf_counter around idx add / take / cast / dot-or-sum, summed over workers (thread-ms per layer), median of 27 layers after 3 warm-up. Read it as a split, not a timing.
- Two input seeds (XSEED 1, 2), 8-core Xeon box (L2 2 MiB/core). No installs, no network, no model download, nothing bound. This wave's absolute times run ~5–10% slower than 02:55 (all8_T128 5.37 / 5.53 vs 4.83 / 4.88 ms); ratios are paired per round.

Command:

    cd /workspace/bench0555_1124 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 900 python3 tmac.py run$s.json > run$s.txt 2>&1; done

Script (tmac.py):

    #!/usr/bin/env python3
    """B=1 T-MAC-style activation LUT GEMV on the W=8 pool, 05:55 11/24. Full bitnetd-style layer at B = 1 (base: bench0255_1124/split.py, gv path).
    tm32_T{128,256}: per site call, build Tab = x.reshape(in/4, 4) @ LUTF_T (in/4 x 256 f32, once per token per site input, main thread);
      per worker tile: idx = PR tile (uint8) + g*256 (int32) -> np.take(Tab.ravel(), idx) f32 (T x in/4) -> row sum np.dot(., ones).
      No weight expansion to f32 / int8, no dot against x.
    tm64_T{128,256}: same with intp (int64) idx (np.take converts int32 indices to intp internally).
    tmc_T128: column chunks of G = 128 groups (sub-table 128 KiB, L2-resident), intp idx, partial sums accumulated per chunk.
    all2_T256: 02:55 u32-LUT take -> int8 view cast -> np.dot path. all8_T{128,256}: int8 cast -> np.dot (baselines).
    Split: separate instrumented pass; main-thread build ms + per-worker perf_counter around idx-add / take / cast / dot-or-sum, summed thread-time per layer."""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import D, F, KVH, QH, HD, rms, silu
    W, ROUNDS, REPS, WARM, G = 8, 9, 11, 3, 128
    base.build(); par.prep(); par.prep8()
    SITES = ("qkv", "o", "gu", "down")
    def wf64(k, bits):
        if bits == 2: return np.concatenate([base.PW[n]["Wt"].astype(np.float64) for n in par.FUSE[k]])
        return np.concatenate([base.PW[n]["q8"].astype(np.float64) * float(base.PW[n]["m8"]) for n in par.FUSE[k]])
    WR = {(k, b): wf64(k, b) for k in SITES for b in (2, 8)}
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT", "Wt"): base.PW[nm].pop(f, None)
    LUT = par.LUT_U32
    LUTF_T = np.ascontiguousarray(base.LUTF_T)                      # (4, 256) f32 trit patterns: code 0 -> 0, 1 -> +1, 2 -> -1, bits 2j = input 4g+j
    TAB = np.empty((F // 4, 256), np.float32)                        # activation LUT, sized for the widest input (in 6912 -> 1.69 MiB), rebuilt per site call
    OFF = {dt: np.arange(F // 4, dtype=dt) * 256 for dt in (np.int32, np.intp)}
    ONES = np.ones(F // 4, np.float32)
    _ts = {}
    def tscr(w, T, q, dt):
        k = (w, T, np.dtype(dt).char)
        if k not in _ts: _ts[k] = (np.empty(T * F // 4, dt), np.empty(T * F // 4, np.float32), np.empty(T, np.float32))
        a, b, c = _ts[k]; return a[:T * q].reshape(T, q), b[:T * q].reshape(T, q), c
    ACC = np.zeros((W, 4)); BACC = np.zeros(1)   # per-worker idx / take / cast / dot-or-sum seconds; main-thread build seconds
    def rows_tm(dt):
        def fn(PR, tab, y, r0, r1, w, T, inst):
            q = PR.shape[1]; idx, gv, _ = tscr(w, T, q, dt); off = OFF[dt][:q]; one = ONES[:q]; pc = time.perf_counter
            for o0 in range(r0, r1, T):
                n = min(T, r1 - o0)
                if inst:
                    t0 = pc(); np.add(PR[o0:o0+n], off, out=idx[:n]); t1 = pc(); np.take(tab, idx[:n], out=gv[:n]); t2 = pc()
                    np.dot(gv[:n], one, out=y[0, o0:o0+n]); t3 = pc(); ACC[w] += (t1 - t0, t2 - t1, 0.0, t3 - t2)
                else:
                    np.add(PR[o0:o0+n], off, out=idx[:n]); np.take(tab, idx[:n], out=gv[:n]); np.dot(gv[:n], one, out=y[0, o0:o0+n])
        return fn
    def rows_tmc(PR, tab, y, r0, r1, w, T, inst):
        q = PR.shape[1]; idx, gv, acc = tscr(w, T, G, np.intp); off = OFF[np.intp][:G]; one = ONES[:G]; pc = time.perf_counter
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0); yy = y[0, o0:o0+n]; yy[:] = 0
            for c0 in range(0, q, G):
                m = min(G, q - c0)
                if inst:
                    t0 = pc(); np.add(PR[o0:o0+n, c0:c0+m], off[:m], out=idx[:n, :m]); t1 = pc()
                    np.take(tab[c0*256:(c0+m)*256], idx[:n, :m], out=gv[:n, :m]); t2 = pc()
                    np.dot(gv[:n, :m], one[:m], out=acc[:n]); yy += acc[:n]; t3 = pc(); ACC[w] += (t1 - t0, t2 - t1, 0.0, t3 - t2)
                else:
                    np.add(PR[o0:o0+n, c0:c0+m], off[:m], out=idx[:n, :m]); np.take(tab[c0*256:(c0+m)*256], idx[:n, :m], out=gv[:n, :m])
                    np.dot(gv[:n, :m], one[:m], out=acc[:n]); yy += acc[:n]
    def rows2(PR, x, y, r0, r1, w, T, inst):
        q = PR.shape[1]; s8, s = par.scratch(w, T, q); pc = time.perf_counter
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            if inst:
                t0 = pc(); np.take(LUT, PR[o0:o0+n], out=s8[:n]); t1 = pc(); np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe"); t2 = pc()
                np.dot(s[:n], x[0], out=y[0, o0:o0+n]); t3 = pc(); ACC[w] += (0.0, t1 - t0, t2 - t1, t3 - t2)
            else:
                np.take(LUT, PR[o0:o0+n], out=s8[:n]); np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe"); np.dot(s[:n], x[0], out=y[0, o0:o0+n])
    def rows8(Q, x, y, r0, r1, w, T, inst):
        i = Q.shape[1]; _, s = par.scratch(w, T, i // 4); pc = time.perf_counter
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            if inst:
                t0 = pc(); np.copyto(s[:n], Q[o0:o0+n], casting="unsafe"); t1 = pc(); np.dot(s[:n], x[0], out=y[0, o0:o0+n]); t3 = pc(); ACC[w] += (0.0, 0.0, t1 - t0, t3 - t1)
            else:
                np.copyto(s[:n], Q[o0:o0+n], casting="unsafe"); np.dot(s[:n], x[0], out=y[0, o0:o0+n])
    FN = {"2": rows2, "8": rows8, "tm32": rows_tm(np.int32), "tm64": rows_tm(np.intp), "tmc": rows_tmc}
    def fmm(key, x, kind, T, inst):
        bits = 8 if kind == "8" else 2
        P = par.FP[key] if bits == 2 else par.F8[key]; R = P["PR"] if bits == 2 else P["Q"]; out = R.shape[0]
        x = np.ascontiguousarray(x, np.float32); y = np.empty((x.shape[0], out), np.float32)
        arg = x
        if kind.startswith("tm"):
            q = R.shape[1]; t0 = time.perf_counter()
            tab = TAB[:q]; np.matmul(x[0].reshape(q, 4), LUTF_T, out=tab); arg = tab.ravel()   # build once per token per site input
            if inst: BACC[0] += time.perf_counter() - t0
        nt = -(-out // T); per = -(-nt // W); fn = FN[kind]
        futs = [par.pool(W).submit(fn, R, arg, y, w * per * T, min(out, (w + 1) * per * T), w, T, inst) for w in range(W) if w * per * T < out]
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
    CFG = {"all2_T256": ("2", 256), "tm32_T128": ("tm32", 128), "tm32_T256": ("tm32", 256), "tm64_T128": ("tm64", 128), "tm64_T256": ("tm64", 256),
           "tmc_T128": ("tmc", 128), "all8_T128": ("8", 128), "all8_T256": ("8", 256)}
    def main():
        seed = int(os.environ.get("XSEED", "1")); rng = np.random.default_rng(seed)
        res = {"xseed": seed}
        W2 = sum(par.FP[k]["PR"].nbytes + par.FP[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        res["wbytes"] = dict(all2=W2, all8=W8, tab_d=D // 4 * 256 * 4, tab_f=F // 4 * 256 * 4)
        print(f"layer weights bytes all2 {W2} ({W2/2**20:.2f} MiB) all8 {W8} ({W8/2**20:.2f} MiB) ({W8/W2:.3f}x); activation LUT in{D} {D//4*1024:,} B, in{F} {F//4*1024:,} B", flush=True)
        for T in (128, 256):
            for dt, b in (("int32", 4), ("intp", 8)): print(f"tmac scratch per worker T{T} {dt}: in{F} idx {T*F//4*b:,} B + f32 gather {T*F//4*4:,} B = {T*F//4*(b+4)/2**20:.2f} MiB", flush=True)
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
            print(f"B1 {c:10s} {d[c]['med']:7.3f} ms (round medians {d[c]['rlo']:.3f}-{d[c]['rhi']:.3f}; all {d[c]['lo']:.3f}-{d[c]['hi']:.3f}) | {1e3/d[c]['med']:6.1f} tok/s/layer | mab {d[c]['mab']:.1e} cos {d[c]['cos']:.8f}", flush=True)
        rr = lambda a, b: [p / q for p, q in zip(rm[a], rm[b])]
        S = {}
        for a in names:
            if a.startswith("all8"): continue
            for b in ("all8_T128", "all8_T256", "all2_T256"):
                if a == b: continue
                v = rr(a, b); nm = f"{a}/{b}"; S[nm] = dict(r=d[a]["med"] / d[b]["med"], lo=min(v), hi=max(v))
                print(f"ratio {nm:24s} {S[nm]['r']:.3f}x (rounds {S[nm]['lo']:.2f}-{S[nm]['hi']:.2f})", flush=True)
        v = rr("all8_T256", "all8_T128"); S["all8_T256/all8_T128"] = dict(r=d["all8_T256"]["med"] / d["all8_T128"]["med"], lo=min(v), hi=max(v))
        print(f"ratio all8_T256/all8_T128 {S['all8_T256/all8_T128']['r']:.3f}x", flush=True)
        res["B1"] = d; res["ratios"] = S
        sp = {}
        for c in names:
            k, T = CFG[c]
            for _ in range(WARM): layer(x, k, T, True)
            per = []
            for r in range(ROUNDS * 3):
                ACC[:] = 0; BACC[:] = 0; t0 = time.perf_counter(); layer(x, k, T, True); wall = (time.perf_counter() - t0) * 1e3
                per.append([BACC[0] * 1e3] + list(ACC.sum(0) * 1e3) + [wall, float(ACC.sum(1).max() * 1e3)])
            m = np.median(np.array(per), axis=0); tot = m[1:5].sum()
            sp[c] = dict(build=m[0], idx=m[1], take=m[2], cast=m[3], dot=m[4], wall=m[5], maxworker=m[6], **{f"{n}_pct": 100 * m[j] / tot for j, n in ((1, "idx"), (2, "take"), (3, "cast"), (4, "dot"))})
            print(f"split {c:10s} build(main) {m[0]:5.3f} ms | thread-ms/layer idx {m[1]:6.2f} ({sp[c]['idx_pct']:4.1f}%) take {m[2]:6.2f} ({sp[c]['take_pct']:4.1f}%) cast {m[3]:6.2f} ({sp[c]['cast_pct']:4.1f}%) dot/sum {m[4]:6.2f} ({sp[c]['dot_pct']:4.1f}%) | busiest worker {m[6]:.2f} ms, wall {m[5]:.2f} ms", flush=True)
        res["split"] = sp
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    main()

## Results

Full layer, B 1, W 8, GEMV (ms; median of per-round medians, min–max of round medians):

| config | seed 1 ms | seed 1 rounds | seed 2 ms | seed 2 rounds | tok/s/layer (s1/s2) | max abs vs own f64 |
|---|---|---|---|---|---|---|
| all2_T256 (02:55 u32 take + cast + dot) | 11.654 | 11.23–14.38 | 11.618 | 10.75–14.79 | 85.8 / 86.1 | 9.6e-07 / 8.5e-07 |
| tm32_T128 (T-MAC, int32 idx) | 13.634 | 12.50–15.00 | 13.583 | 12.31–15.53 | 73.3 / 73.6 | 6.8e-07 / 6.4e-07 |
| tm32_T256 | 12.296 | 10.90–13.97 | 12.778 | 11.66–13.61 | 81.3 / 78.3 | 6.8e-07 / 6.4e-07 |
| tm64_T128 (T-MAC, intp idx) | 12.339 | 11.63–13.95 | 12.495 | 10.69–16.11 | 81.0 / 80.0 | 6.8e-07 / 6.4e-07 |
| **tm64_T256** | **11.876** | 11.10–13.04 | **11.692** | 10.08–13.42 | 84.2 / 85.5 | 6.8e-07 / 6.4e-07 |
| tmc_T128 (G 128 column chunks) | 33.169 | 32.17–34.55 | 34.713 | 31.24–35.66 | 30.1 / 28.8 | 5.3e-07 / 3.2e-07 |
| all8_T128 | 5.366 | 5.08–6.13 | 5.527 | 5.24–6.50 | 186.4 / 180.9 | 3.4e-06 / 3.5e-06 |
| all8_T256 | 5.858 | 5.67–7.16 | 5.881 | 5.51–6.31 | 170.7 / 170.0 | 3.4e-06 / 3.5e-06 |

Ratios (seed 1 / seed 2, per-round range; bar <= 1.3x vs all8_T128):

| 2-bit config | / all8_T128 (best int8) | / all8_T256 | / all2_T256 (u32) |
|---|---|---|---|
| all2_T256 | 2.172x (1.90–2.59) / 2.102x (1.75–2.68) | 1.989x / 1.975x | — |
| tm32_T128 | 2.541x (2.23–2.76) / 2.458x (2.01–2.86) | 2.327x / 2.309x | 1.170x / 1.169x |
| tm32_T256 | 2.292x (2.05–2.51) / 2.312x (1.85–2.57) | 2.099x / 2.173x | 1.055x / 1.100x |
| tm64_T128 | 2.300x (2.20–2.57) / 2.261x (1.97–2.91) | 2.106x / 2.125x | 1.059x / 1.076x |
| **tm64_T256** | **2.213x (2.01–2.35) / 2.115x (1.80–2.43)** | 2.027x / 1.988x | 1.019x / 1.006x |
| tmc_T128 | 6.182x (5.43–6.56) / 6.281x (5.13–6.66) | 5.662x / 5.902x | 2.846x / 2.988x |
| (all8_T256 / all8_T128) | 1.092x / 1.064x | — | — |

Per-op split (instrumented pass; build = main-thread ms per layer; others thread-ms per layer summed over 8 workers, % of idx+take+cast+dot/sum; seed 1 / seed 2):

| config | build (main) | idx add | take | cast | dot / row sum | total thread-ms | busiest worker ms |
|---|---|---|---|---|---|---|---|
| all2_T256 | — | — | 35.96 (56.5%) / 37.07 (57.0%) | 15.93 / 16.02 | 11.77 / 11.95 | 63.66 / 65.03 | 10.25 / 10.58 |
| tm32_T128 | 0.592 / 0.610 | 16.51 (19.9%) / 19.06 (19.7%) | 57.59 (69.5%) / 68.16 (70.3%) | — | 8.74 / 9.68 | 82.85 / 96.90 | 11.85 / 14.37 |
| tm32_T256 | 0.597 / 0.583 | 12.54 (19.6%) / 12.22 (19.9%) | 45.96 (71.8%) / 44.31 (72.1%) | — | 5.51 / 4.91 | 64.01 / 61.45 | 10.29 / 9.95 |
| tm64_T128 | 0.602 / 0.593 | 22.80 (29.3%) / 21.74 (29.1%) | 46.50 (59.7%) / 44.75 (59.8%) | — | 8.59 / 8.33 | 77.89 / 74.82 | 11.34 / 10.85 |
| **tm64_T256** | 0.591 / 0.577 | 17.99 (31.3%) / 17.27 (31.0%) | 34.95 (60.7%) / 34.13 (61.3%) | — | 4.64 (8.1%) / 4.32 (7.8%) | **57.58 / 55.72** | 9.33 / 8.99 |
| tmc_T128 | 0.658 / 0.640 | 80.56 (32.6%) / 78.71 (32.8%) | 144.12 (58.4%) / 140.27 (58.5%) | — | 22.25 / 20.87 | 246.93 / 239.85 | 34.33 / 33.37 |
| all8_T128 | — | — | — | 21.55 (59.3%) / 20.91 (59.6%) | 14.82 / 14.20 | 36.37 / 35.11 | 5.15 / 5.10 |
| all8_T256 | — | — | — | 16.25 (58.3%) / 17.74 (56.8%) | 11.62 / 13.48 | 27.88 / 31.23 | 4.78 / 5.40 |

Bytes: weights per layer 2-bit 17,458,176 B (16.65 MiB), int8 69,559,296 B (66.34 MiB), 3.984x. Activation LUT 655,360 B (in 2560) / 1,769,472 B (in 6912), rebuilt per site call into one reused buffer. T-MAC per-worker scratch at in 6912 (idx + f32 gather): T 128 int32 1.69 MiB, T 128 intp 2.53 MiB, T 256 int32 3.38 MiB, T 256 intp 5.06 MiB. L2 is 2 MiB per core.

## Pair files this wave

This wave's pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) were not in hand for this lane, and none are linked. The newest JuniorHome ledger with a local copy is docs/ledger/2026-11-24_0205.md (/workspace/ledger/jh, close 56eb9370a4cdd659). It reports an idle wave with no builder reports or shas, ROADMAP.md / BETA_TO_OS.md exhausted, "none open; blobs exhausted (#5 UE5 still held)", and LLC 2673729 (04:15 MT, ROADMAP #2 GGUF operator-sets) touching no ledger paths. The 02:55 note cited the 2026-11-22_1805 ledger.

## Next question

Every numpy 2-bit B 1 path (u32 take, f32-LUT take, T-MAC take) costs about one gather per packed weight byte, out x in/4 gathers at ~35 thread-ms per layer, which alone matches int8's whole cast + dot. Reaching 1.3x therefore needs fewer gathers per weight. At B 1 on the W 8 pool, does a base-3 pack of 5 trits per byte (1.6 bit/weight, 243-entry T-MAC table per group of 5 inputs, in padded to a multiple of 5) cut the gather count and index add by 20%? Does it cut weight bytes to ~4.98x vs int8, and bring T-MAC all-2-bit within 1.3x of all8_T128 GEMV, or at least >= 1.2x faster than tm64_T256? The question is whether the 1.25x fewer gathers survive the padding and the less regular table.

close: 84b16366d2d7128c
