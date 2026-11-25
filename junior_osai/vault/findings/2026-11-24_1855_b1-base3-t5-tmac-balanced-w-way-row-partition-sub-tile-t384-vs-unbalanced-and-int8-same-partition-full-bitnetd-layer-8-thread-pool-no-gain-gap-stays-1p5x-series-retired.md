---
local: true
date: 2026-11-24
wave: "2026-11-24 18:55 slot (fired 18:47 MT, ran ~18:47–19:00 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (a balanced row partition, W contiguous ranges of out/W rows sub-tiled at 384, does not close the gap: t5b_64_T384 / all8b_T128 (int8 on the same partition) = 1.571x seed 1, 1.534x seed 2, missing the 1.3x bar; vs the unbalanced t5_64_T384 it is 0.962x / 1.020x, i.e. noise; int8 gains about as much from balancing (all8b_T128 / all8_T128 0.986x / 0.989x, all8b_T256 0.974x / 0.912x); weight memory unchanged at 4.973x vs int8, not new; B 1 numpy ternary GEMV series retired)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, b1, gemv, t-mac, activation-lut, base-3, 1p6-bit, five-trit-pack, load-balance, row-partition, thread-pool, int8, numpy, not-viable, series-retired]
---

# B 1 base-3 T-MAC with a balanced W-way row partition: no gain (0.96–1.02x of unbalanced T 384), still 1.53–1.57x of int8 on the same partition; B 1 numpy ternary GEMV series retired

The 14:55 note ([[2026-11-24_1455_b1-base3-t5-tmac-tile-t384-t512-and-worker-split-tab5-build-full-bitnetd-layer-8-thread-pool-t384-best-1p05x-gap-to-int8-stays-1p5x-1p75x]]) asked: load balance, not tile size or table build, now limits the base-3 T-MAC (at T 384 qkv uses only 5 of 8 workers, at T 512 o and down use 5). At B 1 on the W 8 pool, does a balanced row partition (each site split into exactly W contiguous row ranges of out/W rows, sub-tiled at 384 inside each worker, serial Tab5 build) bring t5_64 within 1.3x of all8_T128 given the same partition? If it still misses 1.3x, retire the B 1 numpy ternary GEMV series and rotate.

This is not a repeat. Every earlier B 1 note dispatched whole T-row tiles, ceil(tiles/W) per worker. None split each site into exactly W equal row ranges.

Answers:
- **Balanced vs unbalanced t5:** t5b_64_T384 / t5_64_T384 = **0.962x (rounds 0.89–1.04) seed 1, 1.020x (0.83–1.23) seed 2**. That is within noise: no consistent gain. The busiest-worker time barely moves (seed 1 5.91 → 5.53 ms; seed 2 6.00 → 7.77 ms, a noisy instrumented run). The idle workers at T 384 were idle on the small qkv / o / down sites, but gate|up (13824 rows, already 8 of 8 busy) dominates the layer, so balancing the small sites saves little. Sub-tiling at 256 inside the balanced ranges is worse (t5b_64_T256 / t5_64_T384 1.019x / 1.063x).
- **Vs int8 on the same partition (the bar):** t5b_64_T384 / all8b_T128 = **1.571x (rounds 1.52–1.91) seed 1, 1.534x (1.16–1.69) seed 2**. **Bar 1.3x: not met.** Vs the unbalanced all8_T128 it is 1.548x / 1.517x. Vs the best int8 this wave (all8b_T256, 5.000 / 4.745 ms) it is 1.590x / 1.663x.
- **int8 gains from balance too:** all8b_T128 / all8_T128 0.986x / 0.989x, all8b_T256 / all8_T128 0.974x / 0.912x. A fair partition helps both formats about equally, so the gap does not close.
- **Memory:** unchanged: t5 weights 13,986,304 B (13.34 MiB) vs 69,559,296 B (66.34 MiB) int8 (**4.973x**, already known) and 17,458,176 B 2-bit (1.248x). Nothing new on memory.
- **Error:** every t5 config matches the float64 ternary reference exactly: max abs **6.3e-07 (seed 1) / 6.9e-07 (seed 2)**, cos 1.00000000. int8 3.4e-06 / 3.5e-06 vs its own f64 reference.
- **Verdict: not-viable. The series is retired.** Six B 1 notes (11-23 18:55 through this one) have tried tiling, per-op splits, the 2-bit T-MAC, the base-3 pack, tile size, a split table build and now load balance. In numpy the ternary GEMV stays at 1.5–2.2x of int8 on speed. The only real win is the weight memory (4.97x vs int8), which was already known. Further B 1 numpy GEMV tuning is not worth a wave. Closing that gap would need native code, which this routine does not allow.

## Method

- Same layer, weights, P5 base-3 pack, pool, timing and split instrumentation as 14:55 (tm5t.py from /workspace/bench1444_1124 → bench1047_1124/tm5.py → /workspace/bench1455_1118 base.py / par.py: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gate|up / down, per-row gamma). Every trit round-trips through the base-3 pack (asserted at build).
- New: the `*b_*` configs (t5b_64_T384, t5b_64_T256, all8b_T128, all8b_T256) split each site into exactly W = 8 contiguous row ranges of out/W rows (qkv 480, o and down 320, gate|up 1728). Each worker walks its range in sub-tiles of T rows (e.g. qkv 384 + 96, o / down one 320 tile at T 384, gate|up 4 x 384 + 192). The t5 Tab5 build stays serial on the main thread. The unbalanced configs (t5_64_T384, all8_T128) keep the old dispatch: ceil(out/T) tiles, ceil(tiles/W) whole tiles per worker.
- Configs in the same rounds: t5_64_T384 (14:55 best, unbalanced), t5b_64_T384, t5b_64_T256, all8_T128 (unbalanced), all8b_T128, all8b_T256.
- Full layer at B 1 (rms, qkv, attention stub, o, residual, rms, gate|up, silu*up, down, residual), OPENBLAS_NUM_THREADS=1, W 8 ThreadPoolExecutor on 8 cores, 9 rounds x 11 reps (3 warm-up) per config, config order rotated each round. Value = median of per-round medians, spread = min–max of round medians; ratios are per-round paired. Instrumented split = median of 27 layers after 3 warm-up. Two input seeds. python3 + numpy only; no installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench1855_1124 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 1200 python3 tm5b.py run$s.json > run$s.txt 2>&1; done

Script (tm5b.py; imports base / par from /workspace/bench1455_1118):

    #!/usr/bin/env python3
    """B=1 balanced row partition, W=8 pool, 18:47 MT 11/24. Base: bench1444_1124/tm5t.py.
    *b_* configs: each site is split into exactly W contiguous row ranges of out/W rows (qkv 480, o/down 320, gate|up 1728), sub-tiled at T inside
      each worker (serial Tab5 build for t5). Unbalanced (old) dispatch gives each worker ceil(ntiles/W) whole tiles, leaving workers idle.
    Previous header follows. B=1 base-3 5-trit/byte T-MAC tile size T256/T384/T512 x serial vs worker-split Tab5 build, W=8 pool, 14:44 MT 11/24. Base: bench1047_1124/tm5.py.
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
    FN = {"t5b_64": rows_t5(np.intp), "8b": rows8, "t5p_64": rows_t5(np.intp), "t5_32": rows_t5(np.int32), "t5_64": rows_t5(np.intp), "2": rows2, "8": rows8, "tm32": rows_tm(np.int32), "tm64": rows_tm(np.intp), "tmc": rows_tmc}
    def fmm(key, x, kind, T, inst):
        bits = 8 if kind in ("8", "8b") else 2
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
        fn = FN[kind]
        if kind.endswith("b") or kind.startswith("t5b"):   # balanced: exactly W contiguous ranges of out/W rows, sub-tiled at T inside each worker
            rw = -(-out // W); futs = [par.pool(W).submit(fn, R, arg, y, w * rw, min(out, (w + 1) * rw), w, T, inst) for w in range(W) if w * rw < out]
        else:
            nt = -(-out // T); per = -(-nt // W)
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
    CFG = {"t5_64_T384": ("t5_64", 384), "t5b_64_T384": ("t5b_64", 384), "t5b_64_T256": ("t5b_64", 256), "all8_T128": ("8", 128), "all8b_T128": ("8b", 128), "all8b_T256": ("8b", 256)}
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
            o = P5[k]["PR"].shape[0]; print(f"site {k} out {o}: tiles/busy workers " + " ".join(f"T{T} {-(-o // T)}/{len(range(0, -(-o // T), -(-(-(-o // T)) // W)))}" for T in (128, 256, 384, 512)) + f" | balanced {W} x {-(-o // W)} rows", flush=True)
        for T in (384,):
            for dt, b in (("intp", 8),): print(f"tmac scratch per worker T{T} {dt}: in{F} idx {T*F//4*b:,} B + f32 gather {T*F//4*4:,} B = {T*F//4*(b+4)/2**20:.2f} MiB", flush=True)
        x = (rng.standard_normal((1, D)) * 0.5).astype(np.float32); R = {b: layer_ref(x, b) for b in (2, 8)}
        d = {}
        for c, (k, T) in CFG.items():
            y = layer(x, k, T); b = 8 if k in ("8", "8b") else 2; d[c] = dict(t=[], mab=float(np.abs(y - R[b]).max()), cos=cos(y, R[b]))
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
            for b in ("all8b_T128", "all8_T128", "t5_64_T384"):
                if a == b: continue
                v = rr(a, b); nm = f"{a}/{b}"; S[nm] = dict(r=d[a]["med"] / d[b]["med"], lo=min(v), hi=max(v))
                print(f"ratio {nm:24s} {S[nm]['r']:.3f}x (rounds {S[nm]['lo']:.2f}-{S[nm]['hi']:.2f})", flush=True)
        for a in ("all8b_T128", "all8b_T256"):
            v = rr(a, "all8_T128"); nm = f"{a}/all8_T128"; S[nm] = dict(r=d[a]["med"] / d["all8_T128"]["med"], lo=min(v), hi=max(v))
            print(f"ratio {nm:24s} {S[nm]['r']:.3f}x (rounds {S[nm]['lo']:.2f}-{S[nm]['hi']:.2f})", flush=True)
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
| t5_64_T384 (14:55 best, unbalanced) | 8.263 [7.89–9.27] | 7.738 [7.35–9.63] | 121.0 / 129.2 | 6.3e-07 / 6.9e-07 |
| **t5b_64_T384 (balanced)** | **7.949** [7.76–9.42] | **7.891** [7.38–9.63] | 125.8 / 126.7 | 6.3e-07 / 6.9e-07 |
| t5b_64_T256 (balanced) | 8.423 [7.90–9.06] | 8.229 [7.44–9.58] | 118.7 / 121.5 | 6.3e-07 / 6.9e-07 |
| all8_T128 (unbalanced) | 5.134 [4.96–5.58] | 5.203 [4.65–5.98] | 194.8 / 192.2 | 3.4e-06 / 3.5e-06 |
| all8b_T128 (balanced) | 5.060 [4.75–5.43] | 5.143 [4.77–6.48] | 197.6 / 194.4 | 3.4e-06 / 3.5e-06 |
| all8b_T256 (balanced) | 5.000 [4.38–5.26] | 4.745 [4.35–5.29] | 200.0 / 210.7 | 3.4e-06 / 3.5e-06 |

Paired ratios (seed 1 / seed 2, per-round range in brackets; bar <= 1.3x vs int8 on the same partition):

| config | / all8b_T128 (same partition) | / all8_T128 (unbalanced) | / t5_64_T384 (unbalanced) |
|---|---|---|---|
| t5_64_T384 | 1.633x (1.49–1.88) / 1.505x (1.19–1.85) | 1.609x (1.57–1.80) / 1.487x (1.29–1.81) | — |
| **t5b_64_T384** | **1.571x (1.52–1.91) / 1.534x (1.16–1.69)** | 1.548x (1.47–1.83) / 1.517x (1.26–1.82) | **0.962x (0.89–1.04) / 1.020x (0.83–1.23)** |
| t5b_64_T256 | 1.665x (1.55–1.79) / 1.600x (1.48–1.71) | 1.640x (1.49–1.74) / 1.582x (1.49–1.76) | 1.019x (0.93–1.10) / 1.063x (0.87–1.24) |
| all8b_T128 | — | 0.986x (0.94–1.10) / 0.989x (0.96–1.12) | — |
| all8b_T256 | — | 0.974x (0.86–1.01) / 0.912x (0.82–1.00) | — |

Per-op split (instrumented; build = main-thread wall ms per layer; others are thread-ms per layer summed over 8 workers; seed 1 / seed 2):

| config | build | idx add | take | cast | row sum / dot | busiest worker ms | wall ms |
|---|---|---|---|---|---|---|---|
| t5_64_T384 | 0.525 / 0.531 | 11.40 / 11.70 | 22.40 / 22.92 | — | 3.04 / 3.20 | 5.91 / 6.00 | 7.75 / 8.01 |
| t5b_64_T384 | 0.534 / 0.527 | 12.68 / 16.07 | 24.45 / 34.15 | — | 4.21 / 4.61 | 5.53 / 7.77 | 7.69 / 9.92 |
| t5b_64_T256 | 0.545 / 0.563 | 14.96 / 15.81 | 29.62 / 31.59 | — | 4.85 / 5.31 | 6.70 / 7.54 | 9.43 / 9.97 |
| all8_T128 | — | — | — | 17.60 / 18.89 | 12.37 / 13.38 | 4.51 / 4.70 | 5.66 / 5.96 |
| all8b_T128 | — | — | — | 19.12 / 20.42 | 14.79 / 15.64 | 4.54 / 4.81 | 5.91 / 6.15 |
| all8b_T256 | — | — | — | 16.06 / 15.65 | 12.56 / 12.50 | 3.98 / 3.85 | 5.50 / 5.32 |

The seed 2 t5b_64_T384 instrumented split ran noisily (thread-ms 54.8 vs 41.3 on seed 1); its timed median (7.891 ms) is in line with seed 1.

Tiles / busy workers per site (out rows), unbalanced: qkv 3840: T128 30/8, T384 10/5; o and down 2560: T128 20/7, T384 7/7; gate|up 13824: T128 108/8, T384 36/8. Balanced: every site 8/8 (480 / 320 / 1728 rows per worker).

Bytes: weights per layer 13,986,304 B t5 vs 17,458,176 B 2-bit vs 69,559,296 B int8 (int8/t5 4.973x, 2-bit/t5 1.248x). Gathers per layer 13,895,168 (t5). Tab5 497,664 B at in 2560 and 1,344,276 B at in 6912. Per-worker scratch at in 6912, T 384 (intp idx + f32 gather): 7.59 MiB.

## Pair files this wave

No pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) for this wave reached this lane, so none are linked. The fresh depth-1 clone of JuniorLLM shows only this lane's own e20685e (14:47 MT, the 14:55 note) as the newest commit. No Trit Pack, Handshake or LLC commit touching junior_osai/vault/findings shows up in the last 6 hours. This lane did not read JuniorHome this wave.

## Next question

The B 1 numpy ternary GEMV series is retired (six notes at 1.5–2.2x of int8). Rotate to prefill with the base-3 pack, which no earlier note covered: the 11-17 prefill notes used only the 2-bit u32-LUT expand. At prefill P 256 / 1024 / 2048 (chunk 512, W 8, full bitnetd layer), expand each P5 row tile to an int8 / f32 tile with a 243 x 5 LUT take (np.take(LUT5, P5 tile) → reshape to (T, in5) → drop padding), then run np.matmul against the chunk. Is that within 1.1x of the 2-bit u32-LUT expand prefill and of int8 prefill? If so, t5 can be the single resident weight format for both decode and prefill at 4.97x less weight memory than int8 and 1.25x less than 2-bit.

close: ac7b5192132560c0
