---
local: true
date: 2026-11-24
wave: "2026-11-24 10:55 slot (fired ~10:47 MT, ran ~10:47–11:00 MT; pair reports for this wave were not in hand; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (a base-3 pack of 5 trits per byte, 1.6 bit/weight, cuts gathers to 0.800x and makes the B 1 T-MAC path 1.22-1.25x faster than tm64_T256 at T 256, meeting the >= 1.2x secondary bar, but it is still 1.45-1.77x of all8_T128 GEMV, missing the 1.3x bar; weight memory is 4.973x vs int8, of which only 1.248x is new vs the 2-bit pack, below the 1.5x bar)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, b1, gemv, t-mac, activation-lut, base-3, 1p6-bit, five-trit-pack, tile-size, int8, thread-pool, numpy, not-viable]
---

# B 1 base-3 5-trit/byte T-MAC on the 8-thread pool: 0.80x the gathers gives a 1.22–1.25x speedup over the 2-bit T-MAC, and weights drop to 4.97x vs int8, but all-ternary is still 1.45–1.77x of int8 T 128 GEMV

The 05:55 note ([[2026-11-24_0555_b1-tmac-activation-lut-row-major-take-plus-row-sum-int32-intp-idx-t128-t256-column-chunk-g128-full-bitnetd-layer-8-thread-pool-2-bit-vs-int8-gap-stays-2p1x]]) found that every numpy 2-bit B 1 path costs about one gather per packed weight byte (out x in/4), and that this alone matches int8's whole cast + dot. It asked: at B 1 on the W 8 pool, does a base-3 pack of 5 trits per byte (1.6 bit/weight, a 243-entry T-MAC table per group of 5 inputs, in padded to a multiple of 5) cut the gather count and the index add by about 20%? Does it cut weight bytes to about 4.98x vs int8, and bring T-MAC all-ternary within 1.3x of all8_T128 GEMV, or at least make it >= 1.2x faster than tm64_T256?

This is not a repeat. The 2026-10-26_1455 / 1855 and 2026-10-27_0255 / 0555 / 1455 base-3 notes packed 5 or 10 trits for the ask-index (retrieval top-k scoring against a shared LUT), not for bitnetd layer GEMV. No earlier note ran a base-3 pack through the full bitnetd layer, the W 8 pool or the 02:55 / 05:55 GEMV baselines.

Answers:
- **Gathers:** 13,895,168 vs 17,367,040 per layer (**0.8001x**). Padding costs almost nothing: in 2560 is 512 groups exactly, and in 6912 pads to 6915, which is 1383 groups vs 1728.
- **Vs tm64_T256 (05:55 best 2-bit T-MAC):** t5_64_T256 / tm64_T256 = **0.803x (rounds 0.73–0.97) / 0.817x (0.69–0.90) for seed 1 (two runs) and 0.798x (0.65–1.30) for seed 2**, so it is **1.22–1.25x faster**. **Secondary bar (>= 1.2x) met.** Wall time fell by about the gather reduction. At T 128 the gain is smaller (t5_64_T128 / tm64_T256 0.92–0.97x), and int32 indices are slower again (t5_32_T256 0.85–0.97x) because np.take converts them to intp.
- **Vs all8_T128 (best int8):** t5_64_T256 / all8_T128 = **1.454x (rounds 1.36–1.71) / 1.596x (1.11–1.81) / 1.765x (1.65–2.49)**. **Bar 1.3x: not met.** On the same runs tm64_T256 was 1.81 / 1.95 / 2.21x, so ternary closes about a fifth of the gap. Against all8_T256 it is 1.35 / 1.37 / 1.55x.
- **Per-op split (cleanest run, seed 1b; thread-ms per layer summed over 8 workers):** t5_64_T256 index add 15.57, take 27.63, row sum 3.81, **total 47.0**. tm64_T256 is 17.96 / 31.70 / 4.22, **total 53.9**. That is about −13% each on index add and take, less than the 20% fewer gathers because the 243-stride table is less regular and each gather row is shorter (1383 vs 1728 elements at in 6912). For comparison all8_T128 is 20.80 cast + 11.62 dot = 32.4. Main-thread table build is 0.56–0.59 ms vs 0.54–0.59 for the 4-trit table. Busiest worker is 7.6–8.4 ms vs 4.8–5.5 ms for int8.
- **Memory:** weights are **13,986,304 B (13.34 MiB)** vs 69,559,296 B (66.34 MiB) for int8 (**4.973x**) and 17,458,176 B (16.65 MiB) for the 2-bit pack. **What is new is 1.248x vs the 2-bit pack**; the ~4x vs int8 has long been recorded. Activation table: 497,664 B at in 2560 and 1,344,276 B at in 6912, which is 0.76x the 4-trit table.
- **Error:** every t5 config matches the float64 ternary reference exactly. Max abs is **6.3e-07 (seed 1) / 6.9e-07 (seed 2)**, cos 1.00000000. All trits round-trip through the base-3 pack (asserted at build). The 2-bit T-MAC was 6.8e-07 / 6.4e-07, u32 9.6e-07 / 8.5e-07, and int8 3.4e-06 / 3.5e-06 vs its own f64 reference.
- **Verdict: not-viable.** By the bar of this B 1 series, speed vs int8 within 1.3x is not met (1.45–1.77x). Speed vs the 2-bit T-MAC is a real 1.22–1.25x gain but below 1.5x. Memory vs int8 is 4.97x, but the new part vs the existing 2-bit pack is 1.25x, also below 1.5x. Error is exact. The 1.6-bit pack is the fastest numpy B 1 ternary path so far (0.72–0.80x of u32 all2_T256) and should replace 2-bit rows as the T-MAC layout, but it does not close the int8 gap.

## Method

- Weights: base.build() / par.prep() / par.prep8() from /workspace/bench1455_1118, the same bitnetd layer as 05:55 (hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gate|up / down) with the same per-row gamma. The P5 rows are built from sign(Wt) per fused site. Each byte = c0 + 3c1 + 9c2 + 27c3 + 81c4, where c 0/1/2 means trit 0/+1/−1 for input 5g+j, and the tail is padded with zero trits. Every trit is decoded back and asserted equal, and gamma is asserted to equal |Wt| row max.
- t5: per site call, x is copied into a zero-padded buffer and Tab5 = x_pad.reshape(in5/5, 5) @ LUT5_T (in5/5 x 243 f32) is built on the main thread inside every timed layer. Per worker tile: idx = P5 tile (uint8) + g*243 (intp, or int32 for t5_32), then np.take(Tab5.ravel(), idx, out=f32 scratch), then np.dot(., ones), then gamma. It uses the same tscr scratch, ONES and dispatch as the 05:55 tm path.
- Configs: t5_64_T128, t5_64_T256, t5_32_T256. Baselines rerun in the same process and rounds: all2_T256 (02:55 u32 take, cast, dot), tm64_T128 / tm64_T256 (05:55 2-bit T-MAC) and all8_T128 / all8_T256 (int8 cast, dot).
- Timing is as at 05:55: one full layer (rms, qkv, attention stub, o, residual, rms, gate|up, silu*up, down, residual) at B 1, OPENBLAS_NUM_THREADS=1, W 8 ThreadPoolExecutor, 9 rounds x 11 reps (3 warm-up) per config with config order rotated each round. The value is the median of per-round medians, with the min–max of round medians as spread. Ratios use per-round pairing. A separate instrumented pass gives the split (median of 27 layers after 3 warm-up).
- Runs: seed 1 twice (run1, run1b) and seed 2 once (run2) on the 8-core Xeon box (L2 2 MiB/core). The box was noisier than at 05:55: all8_T128 ran 5.47–6.95 ms vs 5.37 / 5.53, and run1 had a 1.3–1.6x spread across round medians. Paired ratios are steadier than absolute ms. No installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench1047_1124 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 900 python3 tm5.py run$s.json > run$s.txt 2>&1; done
    OPENBLAS_NUM_THREADS=1 XSEED=1 timeout 900 python3 tm5.py run1b.json > run1b.txt 2>&1

Script (tm5.py):

    #!/usr/bin/env python3
    """B=1 base-3 5-trits-per-byte (1.6 bit/weight) T-MAC GEMV on the W=8 pool, 10:47 MT 11/24. Base: bench0555_1124/tmac.py (same layer, pool, timing, split).
    t5_64_T{128,256} / t5_32_T256: resident P5 rows, byte = c0 + 3 c1 + 9 c2 + 27 c3 + 81 c4 (c 0/1/2 -> trit 0/+1/-1, input 5g+j), in padded to a
      multiple of 5 with zero trits (2560 -> 512 groups, 6912 -> 1383 groups = 6915). Per site call build Tab5 = x_pad.reshape(in5/5, 5) @ LUT5_T
      (in5/5 x 243 f32, main thread, inside the timed layer). Per worker tile: idx = P5 tile (uint8) + g*243 (intp / int32) -> np.take(Tab5.ravel(), idx)
      -> row sum np.dot(., ones) -> gamma. Gathers per row in5/5 vs in/4 for the 2-bit T-MAC (0.800x at in 2560, 0.800x at in 6912 incl. padding).
    Baselines rerun: all2_T256 (02:55 u32 take/cast/dot), tm64_T128 / tm64_T256 (05:55 2-bit T-MAC), all8_T128 / all8_T256 (int8 cast/dot)."""
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
    def p5(k):
        t = np.concatenate([np.sign(base.PW[n]["Wt"]) for n in par.FUSE[k]]).astype(np.int8); o, i = t.shape; i5 = -(-i // 5) * 5
        c = np.zeros((o, i5), np.uint8); c[:, :i] = np.where(t < 0, 2, t); c = c.reshape(o, i5 // 5, 5).astype(np.uint16)
        p = np.ascontiguousarray((c * np.array([1, 3, 9, 27, 81], np.uint16)).sum(-1).astype(np.uint8))
        dec = np.stack([(p.astype(np.int16) // 3 ** j) % 3 for j in range(5)], -1).reshape(o, i5)[:, :i]; dec[dec == 2] = -1
        assert np.array_equal(dec.astype(np.int8), t), k                       # exact round-trip of every trit
        g = par.FP[k]["g"]; assert np.allclose(np.abs(np.concatenate([base.PW[n]["Wt"] for n in par.FUSE[k]])).max(1)[np.abs(t).max(1) > 0], g[np.abs(t).max(1) > 0])
        return dict(PR=p, g=g, split=par.FP[k]["split"], inp=i, inp5=i5)
    P5 = {k: p5(k) for k in SITES}
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT", "Wt"): base.PW[nm].pop(f, None)
    LUT = par.LUT_U32
    LUTF_T = np.ascontiguousarray(base.LUTF_T)                      # (4, 256) f32 trit patterns: code 0 -> 0, 1 -> +1, 2 -> -1, bits 2j = input 4g+j
    TAB = np.empty((F // 4, 256), np.float32)                        # activation LUT, sized for the widest input (in 6912 -> 1.69 MiB), rebuilt per site call
    OFF = {dt: np.arange(F // 4, dtype=dt) * 256 for dt in (np.int32, np.intp)}
    ONES = np.ones(F // 4, np.float32)
    F5 = -(-F // 5)
    LUT5_T = np.array([[((c // 3 ** j) % 3) for j in range(5)] for c in range(243)], np.float32); LUT5_T[LUT5_T == 2] = -1; LUT5_T = np.ascontiguousarray(LUT5_T.T)  # (5, 243)
    TAB5 = np.empty((F5, 243), np.float32); XP = np.zeros(F5 * 5, np.float32)            # 243-entry table per group of 5 inputs, in 6912 -> 1,344,276 B
    OFF5 = {dt: np.arange(F5, dtype=dt) * 243 for dt in (np.int32, np.intp)}
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
    def rows_t5(dt):
        def fn(PR, tab, y, r0, r1, w, T, inst):
            q = PR.shape[1]; idx, gv, _ = tscr(w, T, q, dt); off = OFF5[dt][:q]; one = ONES[:q]; pc = time.perf_counter
            for o0 in range(r0, r1, T):
                n = min(T, r1 - o0)
                if inst:
                    t0 = pc(); np.add(PR[o0:o0+n], off, out=idx[:n]); t1 = pc(); np.take(tab, idx[:n], out=gv[:n]); t2 = pc()
                    np.dot(gv[:n], one, out=y[0, o0:o0+n]); t3 = pc(); ACC[w] += (t1 - t0, t2 - t1, 0.0, t3 - t2)
                else:
                    np.add(PR[o0:o0+n], off, out=idx[:n]); np.take(tab, idx[:n], out=gv[:n]); np.dot(gv[:n], one, out=y[0, o0:o0+n])
        return fn
    FN = {"t5_32": rows_t5(np.int32), "t5_64": rows_t5(np.intp), "2": rows2, "8": rows8, "tm32": rows_tm(np.int32), "tm64": rows_tm(np.intp), "tmc": rows_tmc}
    def fmm(key, x, kind, T, inst):
        bits = 8 if kind == "8" else 2
        P = P5[key] if kind.startswith("t5") else par.FP[key] if bits == 2 else par.F8[key]; R = P["PR"] if bits == 2 else P["Q"]; out = R.shape[0]
        x = np.ascontiguousarray(x, np.float32); y = np.empty((x.shape[0], out), np.float32)
        arg = x
        if kind.startswith("t5"):
            q = R.shape[1]; t0 = time.perf_counter(); i = x.shape[1]
            xp = XP[:q * 5]; xp[:i] = x[0]; xp[i:] = 0
            tab = TAB5[:q]; np.matmul(xp.reshape(q, 5), LUT5_T, out=tab); arg = tab.ravel()   # build once per token per site input
            if inst: BACC[0] += time.perf_counter() - t0
        elif kind.startswith("tm"):
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
    CFG = {"all2_T256": ("2", 256), "tm64_T128": ("tm64", 128), "tm64_T256": ("tm64", 256), "t5_64_T128": ("t5_64", 128), "t5_64_T256": ("t5_64", 256),
           "t5_32_T256": ("t5_32", 256), "all8_T128": ("8", 128), "all8_T256": ("8", 256)}
    def main():
        seed = int(os.environ.get("XSEED", "1")); rng = np.random.default_rng(seed)
        res = {"xseed": seed}
        W2 = sum(par.FP[k]["PR"].nbytes + par.FP[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES)
        res["wbytes"] = dict(all2=W2, all8=W8, t5=W5, tab_d=D // 4 * 256 * 4, tab_f=F // 4 * 256 * 4, tab5_d=D // 5 * 243 * 4, tab5_f=F5 * 243 * 4)
        G2 = sum(par.FP[k]["PR"].size for k in SITES); G5 = sum(P5[k]["PR"].size for k in SITES); res["gathers"] = dict(tm=G2, t5=G5)
        print(f"t5 weights bytes {W5} ({W5/2**20:.2f} MiB): int8/t5 {W8/W5:.3f}x, 2-bit/t5 {W2/W5:.3f}x; gathers per layer tm {G2:,} t5 {G5:,} ({G5/G2:.4f}x); tab5 in{D} {D//5*243*4:,} B in{F} {F5*243*4:,} B", flush=True)
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
            for b in ("all8_T128", "all8_T256", "all2_T256", "tm64_T256"):
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

Full layer, B 1, W 8, GEMV (ms; median of per-round medians, min–max of round medians in brackets):

| config | seed 1 (run1) | seed 1 (run1b) | seed 2 (run2) | tok/s/layer (1 / 1b / 2) | max abs vs own f64 (s1 / s2) |
|---|---|---|---|---|---|
| all2_T256 (u32 take + cast + dot) | 14.080 [11.58–16.50] | 13.632 [11.86–15.65] | 12.062 [11.95–18.16] | 71.0 / 73.4 / 82.9 | 9.6e-07 / 8.5e-07 |
| tm64_T128 (2-bit T-MAC) | 13.078 [10.75–14.31] | 13.070 [11.03–18.74] | 11.729 [10.76–15.95] | 76.5 / 76.5 / 85.3 | 6.8e-07 / 6.4e-07 |
| tm64_T256 (2-bit T-MAC, 05:55 best) | 12.577 [10.51–17.37] | 11.947 [10.91–15.30] | 12.098 [10.76–14.73] | 79.5 / 83.7 / 82.7 | 6.8e-07 / 6.4e-07 |
| t5_64_T128 (base-3, intp idx) | 12.194 [9.79–14.55] | 11.265 [10.35–15.09] | 11.085 [10.37–13.27] | 82.0 / 88.8 / 90.2 | 6.3e-07 / 6.9e-07 |
| **t5_64_T256** | **10.102** [8.78–13.16] | **9.765** [9.17–12.00] | **9.659** [9.16–14.37] | 99.0 / 102.4 / 103.5 | 6.3e-07 / 6.9e-07 |
| t5_32_T256 (int32 idx) | 10.828 [9.31–12.60] | 11.593 [9.87–17.00] | 10.307 [9.56–15.65] | 92.3 / 86.3 / 97.0 | 6.3e-07 / 6.9e-07 |
| all8_T128 | 6.948 [5.39–7.75] | 6.117 [5.56–8.70] | 5.473 [5.31–6.10] | 143.9 / 163.5 / 182.7 | 3.4e-06 / 3.5e-06 |
| all8_T256 | 7.463 [5.88–8.71] | 7.127 [6.57–8.03] | 6.235 [6.05–9.01] | 134.0 / 140.3 / 160.4 | 3.4e-06 / 3.5e-06 |

Paired ratios (run1 / run1b / run2; per-round range for run1 and run2; bar <= 1.3x vs all8_T128, secondary bar <= 0.833x vs tm64_T256, i.e. >= 1.2x faster):

| config | / all8_T128 | / all8_T256 | / all2_T256 | / tm64_T256 |
|---|---|---|---|---|
| tm64_T256 | 1.810x (1.51–2.24) / 1.953x / 2.211x (1.92–2.54) | 1.685x / — / 1.940x | 0.893x / — / 1.003x | — |
| t5_64_T128 | 1.755x / — / 2.025x | 1.634x / — / 1.778x | 0.866x / — / 0.919x | 0.970x / — / 0.916x |
| **t5_64_T256** | **1.454x (1.36–1.71) / 1.596x / 1.765x (1.65–2.49)** | 1.354x / 1.370x / 1.549x | 0.718x / 0.716x / 0.801x | **0.803x (0.73–0.97) / 0.817x / 0.798x (0.65–1.30)** |
| t5_32_T256 | 1.559x / — / 1.883x | 1.451x / — / 1.653x | 0.769x / — / 0.854x | 0.861x / 0.970x / 0.852x |
| (all8_T256 / all8_T128) | 1.074x / 1.165x / 1.139x | — | — | — |

Per-op split (instrumented pass; build = main-thread ms per layer; others are thread-ms per layer summed over 8 workers; run1 / run1b / run2):

| config | build (main) | idx add | take | row sum / dot | total thread-ms | busiest worker ms |
|---|---|---|---|---|---|---|
| tm64_T256 | 0.582 / 0.543 / 0.589 | 17.61 / 17.96 / 22.64 | 33.26 / 31.70 / 43.76 | 4.17 / 4.22 / 4.66 | 55.0 / 53.9 / 71.1 | 9.26 / 8.87 / 13.27 |
| t5_64_T128 | 0.560 / 0.588 / 0.574 | 19.72 / 19.37 / 20.71 | 40.67 / 36.57 / 42.43 | 8.27 / 8.36 / 8.52 | 68.7 / 64.3 / 71.7 | 10.03 / 9.23 / 10.65 |
| **t5_64_T256** | 0.560 / 0.577 / 0.573 | 15.66 / 15.57 / 15.42 | 29.14 / 27.63 / 27.67 | 4.05 / 3.81 / 3.82 | **48.9 / 47.0 / 46.9** | 8.41 / 7.65 / 7.61 |
| t5_32_T256 | 0.545 / 0.569 / 0.570 | 10.63 / 10.80 / 10.87 | 38.86 / 35.90 / 37.02 | 4.29 / 4.34 / 4.40 | 53.8 / 51.0 / 52.3 | 9.16 / 8.17 / 8.42 |
| all8_T128 (cast / dot) | — | — | — | 22.34 + 14.20 / 20.80 + 11.62 / 22.79 + 13.07 | 36.5 / 32.4 / 35.9 | 5.51 / 4.80 / 5.21 |

Bytes: weights per layer are 13,986,304 B (13.34 MiB) for t5, 17,458,176 B (16.65 MiB) for 2-bit and 69,559,296 B (66.34 MiB) for int8. That makes int8/t5 **4.973x**, 2-bit/t5 **1.248x** and int8/2-bit 3.984x (gamma included). Gathers per layer: 13,895,168 for t5 vs 17,367,040 for the 2-bit T-MAC (0.8001x). Activation table: 497,664 B at in 2560 and 1,344,276 B at in 6912 for t5, vs 655,360 / 1,769,472 B. Per-worker scratch is at most 0.80x the 05:55 figures, because the same buffers are used with in5/5 columns.

## Pair files this wave

This wave's pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) were not in hand for this lane, and none are linked. The newest local JuniorHome ledger is docs/ledger/2026-11-24_0505.md (/workspace/JuniorHome-ledger-20261124-0505, commit 3e69f97, Close 7d55a6a39effc71c). It reports an idle wave with no builder reports or shas, ROADMAP.md / BETA_TO_OS.md exhausted, and "none open; blobs exhausted (#5 UE5 still held)". It says LLC 46f089f (ROADMAP i2sd userspace named item) touched no ledger paths. The 05:55 note cited the 2026-11-24_0205 ledger. No new pair outputs exist for this lane.

## Next question

The 1.6-bit pack removes about 13% of the index add and take thread-ms and about 20% of wall time. T 256 beat T 128 by 13–17% for both T-MAC layouts, and the row sum halves from T 128 to T 256, so per-tile numpy call overhead is still a visible cost. At B 1 on the W 8 pool, does t5_64 at T 384 / T 512, with the Tab5 build split per site across workers (removing the 0.56 ms serial build), bring the base-3 T-MAC within 1.3x of all8_T128? Or does the per-worker intp scratch (6.3–8.5 MiB at in 6912, past the 2 MiB L2) cancel the call savings? Keep the 5-trit pack fixed and compare against t5_64_T256 and all8_T128 in the same rounds.

close: 463b53d2d32390af
