---
local: true
date: 2026-11-24
wave: "2026-11-24 14:55 slot (fired 14:44 MT, ran ~14:44–15:00 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (larger tiles help the base-3 T-MAC a little: t5_64_T384 is 1.05-1.07x faster than t5_64_T256, but it is still 1.54-1.75x of all8_T128 GEMV, missing the 1.3x bar; T 512 gives nothing (0.99-1.00x of T 256); building Tab5 in worker slices costs an extra pool dispatch that makes the build wall 1.29-1.71 ms vs 0.53-0.56 ms serial, so it is no gain; weight memory unchanged at 4.973x vs int8, 1.248x vs the 2-bit pack)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, b1, gemv, t-mac, activation-lut, base-3, 1p6-bit, five-trit-pack, tile-size, load-balance, thread-pool, int8, numpy, not-viable]
---

# B 1 base-3 T-MAC: T 384 is the best tile (1.05–1.07x over T 256), T 512 and a worker-split table build add nothing, and all-ternary stays 1.54–1.75x of int8 T 128 GEMV

The 10:51 note ([[2026-11-24_1051_b1-base3-5-trit-per-byte-1p6-bit-tmac-243-entry-table-in-padded-to-5-full-bitnetd-layer-8-thread-pool-1p25x-faster-than-2-bit-tmac-gap-to-int8-stays-1p6x]]) asked: at B 1 on the W 8 pool, does t5_64 at T 384 / T 512, with the Tab5 build split per site across workers (removing the 0.56 ms serial build), bring the base-3 T-MAC within 1.3x of all8_T128? Or does the per-worker intp scratch (6.3–8.5 MiB at in 6912, past the 2 MiB L2) cancel the call savings?

This is not a repeat. No earlier note ran the 5-trit pack above T 256, and none split the T-MAC table build across workers.

Answers:
- **Tile size:** t5_64_T384 / t5_64_T256 = **0.950x (rounds 0.87–1.05) seed 1, 0.937x (0.86–2.04) seed 2**, so T 384 is 1.05–1.07x faster. T 512 is 0.985x / 0.996x, no gain. Thread-ms per layer keeps falling with T (seed 2: 42.5 → 39.6 → 36.4 at T 256 / 384 / 512, mostly the row sum and index add), but at T 512 the o and down sites (out 2560) have only 5 tiles for 8 workers, and qkv at T 384 only 10 tiles on 5 workers, so the busiest worker does not shrink (seed 1: 7.06 / 6.49 / 8.20 ms). Per-worker intp + f32 scratch is 5.06 / 7.59 / 10.12 MiB at T 256 / 384 / 512 (in 6912), all past L2 already at T 256, so the scratch size is not the limit; worker balance is.
- **Worker-split Tab5 build (t5p):** the build wall per layer rises from **0.53–0.56 ms (serial, main thread) to 1.29–1.71 ms**, because the extra pool round-trip per site costs more than the 4-site matmul it splits. Same-tile t5p / t5: T 256 1.020x / 1.093x (slower); T 384 0.959x / 1.068x (mixed); T 512 1.015x / 1.053x (slower). **No gain.**
- **Vs all8_T128 (best int8):** best ternary t5_64_T384 = **1.750x (rounds 1.48–1.92) seed 1, 1.544x (1.45–3.05) seed 2**; t5p_64_T384 1.677x / 1.649x. **Bar 1.3x: not met.** t5_64_T256 on the same rounds was 1.843x / 1.647x and the 2-bit tm64_T256 2.162x / 1.901x.
- **Vs 2-bit T-MAC:** t5_64_T384 / tm64_T256 = 0.809x / 0.812x (1.23x faster), only a small step on the 10:51 0.80–0.82x.
- **Memory:** unchanged: t5 weights 13,986,304 B (13.34 MiB) vs 69,559,296 B (66.34 MiB) int8 (**4.973x**) and 17,458,176 B 2-bit (1.248x). Nothing new on memory.
- **Error:** every t5 / t5p config matches the float64 ternary reference exactly: max abs **6.3e-07 (seed 1) / 6.9e-07 (seed 2)**, cos 1.00000000. int8 3.4e-06 / 3.5e-06 vs its own f64 reference.
- **Verdict: not-viable.** T 384 is a real but small 5–7% gain and should become the t5 default tile. A worker-split build and T 512 are not worth it in numpy. The B 1 numpy ternary GEMV path has now plateaued at about 1.5–2.2x of int8 over five B 1 notes (11-23 18:55 through this one).

## Method

- Same layer, weights, P5 base-3 pack, pool, timing and split instrumentation as 10:51 (tm5.py from /workspace/bench1047_1124, which builds on /workspace/bench1455_1118 base.py / par.py: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gate|up / down, per-row gamma). Every trit round-trips through the base-3 pack (asserted at build).
- New: t5p_64_T{256,384,512} builds Tab5 (in5/5 x 243 f32) in W row slices on the pool (np.matmul(xp[g0*5:g1*5].reshape(g1-g0, 5), LUT5_T, out=tab[g0:g1])), waits, then dispatches the GEMV tiles. t5_64_T{256,384,512} keeps the serial main-thread build. Padding the input to x5 stays on the main thread in both.
- Configs in the same rounds: tm64_T256 (05:55 2-bit T-MAC), t5_64_T256 / T384 / T512, t5p_64_T256 / T384 / T512, all8_T128, all8_T256.
- Full layer at B 1 (rms, qkv, attention stub, o, residual, rms, gate|up, silu*up, down, residual), OPENBLAS_NUM_THREADS=1, W 8 ThreadPoolExecutor, 9 rounds x 11 reps (3 warm-up) per config, config order rotated each round. Value = median of per-round medians, spread = min–max of round medians; ratios are per-round paired. Instrumented split = median of 27 layers after 3 warm-up. Two input seeds. python3 + numpy only; no installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench1444_1124 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 1200 python3 tm5t.py run$s.json > run$s.txt 2>&1; done

Script (tm5t.py; imports base / par from /workspace/bench1455_1118):

    #!/usr/bin/env python3
    """B=1 base-3 5-trit/byte T-MAC tile size T256/T384/T512 x serial vs worker-split Tab5 build, W=8 pool, 14:44 MT 11/24. Base: bench1047_1124/tm5.py.
    t5p_*: the in5/5 x 243 table is built in W row slices on the pool (one extra dispatch) instead of serially on the main thread.
    Previous header follows. B=1 base-3 5-trits-per-byte (1.6 bit/weight) T-MAC GEMV on the W=8 pool, 10:47 MT 11/24. Base: bench0555_1124/tmac.py (same layer, pool, timing, split).
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
    def bld5(xp, tab, g0, g1): np.matmul(xp[g0 * 5:g1 * 5].reshape(g1 - g0, 5), LUT5_T, out=tab[g0:g1])
    FN = {"t5p_64": rows_t5(np.intp), "t5_32": rows_t5(np.int32), "t5_64": rows_t5(np.intp), "2": rows2, "8": rows8, "tm32": rows_tm(np.int32), "tm64": rows_tm(np.intp), "tmc": rows_tmc}
    def fmm(key, x, kind, T, inst):
        bits = 8 if kind == "8" else 2
        P = P5[key] if kind.startswith("t5") else par.FP[key] if bits == 2 else par.F8[key]; R = P["PR"] if bits == 2 else P["Q"]; out = R.shape[0]
        x = np.ascontiguousarray(x, np.float32); y = np.empty((x.shape[0], out), np.float32)
        arg = x
        if kind.startswith("t5"):
            q = R.shape[1]; t0 = time.perf_counter(); i = x.shape[1]
            xp = XP[:q * 5]; xp[:i] = x[0]; xp[i:] = 0
            tab = TAB5[:q]
            if kind.startswith("t5p"):
                gper = -(-q // W); bf = [par.pool(W).submit(bld5, xp, tab, w * gper, min(q, (w + 1) * gper)) for w in range(W) if w * gper < q]
                for f in bf: f.result()
            else: np.matmul(xp.reshape(q, 5), LUT5_T, out=tab)
            arg = tab.ravel()
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
    CFG = {"tm64_T256": ("tm64", 256), "t5_64_T256": ("t5_64", 256), "t5_64_T384": ("t5_64", 384), "t5_64_T512": ("t5_64", 512),
           "t5p_64_T256": ("t5p_64", 256), "t5p_64_T384": ("t5p_64", 384), "t5p_64_T512": ("t5p_64", 512), "all8_T128": ("8", 128), "all8_T256": ("8", 256)}
    def main():
        seed = int(os.environ.get("XSEED", "1")); rng = np.random.default_rng(seed)
        res = {"xseed": seed}
        W2 = sum(par.FP[k]["PR"].nbytes + par.FP[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES)
        res["wbytes"] = dict(all2=W2, all8=W8, t5=W5, tab_d=D // 4 * 256 * 4, tab_f=F // 4 * 256 * 4, tab5_d=D // 5 * 243 * 4, tab5_f=F5 * 243 * 4)
        G2 = sum(par.FP[k]["PR"].size for k in SITES); G5 = sum(P5[k]["PR"].size for k in SITES); res["gathers"] = dict(tm=G2, t5=G5)
        print(f"t5 weights bytes {W5} ({W5/2**20:.2f} MiB): int8/t5 {W8/W5:.3f}x, 2-bit/t5 {W2/W5:.3f}x; gathers per layer tm {G2:,} t5 {G5:,} ({G5/G2:.4f}x); tab5 in{D} {D//5*243*4:,} B in{F} {F5*243*4:,} B", flush=True)
        print(f"layer weights bytes all2 {W2} ({W2/2**20:.2f} MiB) all8 {W8} ({W8/2**20:.2f} MiB) ({W8/W2:.3f}x); activation LUT in{D} {D//4*1024:,} B, in{F} {F//4*1024:,} B", flush=True)
        for k in SITES:
            o = P5[k]["PR"].shape[0]; print(f"site {k} out {o}: tiles/busy workers " + " ".join(f"T{T} {-(-o // T)}/{len(range(0, -(-o // T), -(-(-(-o // T)) // W)))}" for T in (128, 256, 384, 512)), flush=True)
        for T in (256, 384, 512):
            for dt, b in (("intp", 8),): print(f"tmac scratch per worker T{T} {dt}: in{F} idx {T*F//4*b:,} B + f32 gather {T*F//4*4:,} B = {T*F//4*(b+4)/2**20:.2f} MiB", flush=True)
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
            for b in ("all8_T128", "all8_T256", "t5_64_T256", "tm64_T256"):
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

| config | seed 1 | seed 2 | tok/s/layer (1 / 2) | max abs vs own f64 (1 / 2) |
|---|---|---|---|---|
| tm64_T256 (2-bit T-MAC) | 11.905 [10.95–14.34] | 10.145 [9.49–11.93] | 84.0 / 98.6 | 6.8e-07 / 6.4e-07 |
| t5_64_T256 (10:51 best) | 10.146 [8.64–10.81] | 8.791 [7.96–10.33] | 98.6 / 113.7 | 6.3e-07 / 6.9e-07 |
| **t5_64_T384** | **9.634** [8.13–11.02] | **8.241** [7.21–18.36] | 103.8 / 121.4 | 6.3e-07 / 6.9e-07 |
| t5_64_T512 | 9.995 [8.44–10.61] | 8.757 [7.81–11.17] | 100.0 / 114.2 | 6.3e-07 / 6.9e-07 |
| t5p_64_T256 (split build) | 10.350 [9.17–11.95] | 9.605 [8.23–11.74] | 96.6 / 104.1 | 6.3e-07 / 6.9e-07 |
| t5p_64_T384 | 9.236 [8.23–11.88] | 8.800 [7.58–9.64] | 108.3 / 113.6 | 6.3e-07 / 6.9e-07 |
| t5p_64_T512 | 10.143 [8.84–13.12] | 9.225 [8.33–10.81] | 98.6 / 108.4 | 6.3e-07 / 6.9e-07 |
| all8_T128 | 5.506 [5.19–7.12] | 5.336 [4.80–6.03] | 181.6 / 187.4 | 3.4e-06 / 3.5e-06 |
| all8_T256 | 6.129 [5.76–6.98] | 5.542 [5.24–10.38] | 163.2 / 180.4 | 3.4e-06 / 3.5e-06 |

Seed 2 t5_64_T384 had one slow round (18.36 ms); its median is not affected.

Paired ratios (seed 1 / seed 2, per-round range in brackets; bar <= 1.3x vs all8_T128):

| config | / all8_T128 | / all8_T256 | / t5_64_T256 | / tm64_T256 |
|---|---|---|---|---|
| tm64_T256 | 2.162x (1.71–2.70) / 1.901x (1.74–2.35) | 1.943x / 1.830x | 1.173x / 1.154x | — |
| t5_64_T256 | 1.843x (1.45–2.03) / 1.647x (1.50–1.91) | 1.655x / 1.586x | — | 0.852x / 0.867x |
| **t5_64_T384** | **1.750x (1.48–1.92) / 1.544x (1.45–3.05)** | 1.572x / 1.487x | **0.950x / 0.937x** | 0.809x / 0.812x |
| t5_64_T512 | 1.815x (1.49–1.90) / 1.641x (1.45–2.06) | 1.631x / 1.580x | 0.985x / 0.996x | 0.840x / 0.863x |
| t5p_64_T256 | 1.880x (1.67–2.06) / 1.800x (1.45–2.20) | 1.689x / 1.733x | 1.020x / 1.093x | 0.869x / 0.947x |
| t5p_64_T384 | 1.677x (1.57–2.16) / 1.649x (1.50–1.84) | 1.507x / 1.588x | 0.910x / 1.001x | 0.776x / 0.867x |
| t5p_64_T512 | 1.842x (1.64–2.38) / 1.729x (1.60–1.96) | 1.655x / 1.665x | 1.000x / 1.049x | 0.852x / 0.909x |
| (all8_T256 / all8_T128) | 1.113x / 1.039x | — | — | — |

Per-op split (instrumented; build = main-thread wall ms per layer, which for t5p includes the pool round-trips; others are thread-ms per layer summed over 8 workers; seed 1 / seed 2):

| config | build | idx add | take | row sum / dot | total thread-ms | busiest worker ms |
|---|---|---|---|---|---|---|
| tm64_T256 | 0.576 / 0.540 | 17.70 / 15.47 | 34.34 / 29.90 | 4.01 / 4.05 | 56.1 / 49.4 | 9.34 / 7.94 |
| t5_64_T256 | 0.547 / 0.533 | 14.38 / 13.59 | 25.55 / 25.18 | 3.63 / 3.77 | 43.6 / 42.5 | 7.06 / 6.97 |
| t5_64_T384 | 0.555 / 0.552 | 13.09 / 12.92 | 24.09 / 23.67 | 3.15 / 3.05 | 40.3 / 39.6 | 6.49 / 6.34 |
| t5_64_T512 | 0.558 / 0.528 | 13.78 / 11.44 | 26.08 / 22.40 | 2.79 / 2.59 | 42.7 / 36.4 | 8.20 / 6.38 |
| t5p_64_T256 | 1.614 / 1.294 | 16.34 / 13.23 | 32.01 / 24.45 | 4.82 / 3.84 | 53.2 / 41.5 | 8.92 / 6.68 |
| t5p_64_T384 | 1.624 / 1.473 | 14.59 / 11.99 | 28.29 / 24.12 | 3.21 / 2.86 | 46.1 / 39.0 | 7.35 / 6.43 |
| t5p_64_T512 | 1.708 / 1.439 | 14.26 / 11.17 | 27.33 / 22.78 | 2.78 / 2.45 | 44.4 / 36.4 | 8.31 / 6.41 |
| all8_T128 (cast / dot) | — | — | — | 20.89 + 14.07 / 18.21 + 12.72 | 35.0 / 30.9 | 5.05 / 4.51 |

Tiles / busy workers per site (out rows): qkv 3840: T256 15/8, T384 10/5, T512 8/8; o and down 2560: T256 10/5, T384 7/7, T512 5/5; gate|up 13824: T256 54/8, T384 36/8, T512 27/7.

Bytes: weights per layer 13,986,304 B t5 vs 17,458,176 B 2-bit vs 69,559,296 B int8 (int8/t5 4.973x, 2-bit/t5 1.248x). Gathers per layer 13,895,168 (t5) vs 17,367,040 (2-bit T-MAC). Tab5 497,664 B at in 2560 and 1,344,276 B at in 6912. Per-worker scratch at in 6912 (intp idx + f32 gather): 5.06 / 7.59 / 10.12 MiB at T 256 / 384 / 512.

## Pair files this wave

No pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) for this wave reached this lane, so none are linked. The newest JuniorHome ledger on GitHub is the 2026-11-24 10:05 MT wave (01e84df); this lane did not read its body. JuniorHome commits since 05:05 MT are LLC named-item advances (ROADMAP #1 skill-pin, #2 GGUF, #3 i2sd userspace) and the 06:00 hwsw seam 56a9748; none touched JuniorLLM junior_osai/vault/findings. The only JuniorLLM commit in the last 6 hours is this lane's own abb4414 (10:51 note).

## Next question

Load balance, not tile size or table build, now limits the base-3 T-MAC: at T 384 qkv uses only 5 of 8 workers, and at T 512 o and down use 5. At B 1 on the W 8 pool, does a balanced row partition (each site split into exactly W contiguous row ranges of out/W rows, sub-tiled at 384 inside each worker, serial Tab5 build) bring t5_64 within 1.3x of all8_T128 given the same partition? If it still misses 1.3x, retire the B 1 numpy ternary GEMV series (now five notes at 1.5–2.2x of int8) and rotate to a different question.

close: 91de21e9f808e58d
