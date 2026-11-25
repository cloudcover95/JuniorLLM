---
local: true
date: 2026-11-25
wave: "2026-11-25 05:55 slot (fired 05:45 MT, ran ~05:45–06:00 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (base-3 t5 small-batch crossover, full bitnetd layer, W 8, ctx-free: the T-MAC gather with B 243-entry tables (tmL_T384) is the faster t5 path at B 1 and B 2 (tmL / best expand 0.66x / 0.61x at B 1, 0.73x / 0.72x at B 2, seed 1 / seed 2); the void5 LUT expand + weight-major matmul (t5vg_T256) wins from B 4 on (tmL / t5vg 1.25x / 1.37x at B 4, 2.2–2.5x at B 8, 6.4–6.8x at B 32), so the crossover lies between B 2 and B 4 (~B 3 by linear fit); the faster t5 path is 1.71x–2.68x slower than the best int8 (T 128) path at every B, so the 1.3x int8 bar is missed everywhere; no speed win, the 4.973x weight-memory win vs int8 is already known)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, multi-stream, speculative-verify, small-batch, crossover, base-3, 1p6-bit, five-trit-pack, tmac, lut-expand, int8, matmul, gemv, thread-pool, numpy, not-viable]
---

# Small-batch crossover for the base-3 t5 pack: T-MAC gather wins only at B 1–2, the void5 expand + matmul wins from B 4; the best t5 path is 1.7–2.7x slower than int8 at every B 1–32

The 02:47 note ([[2026-11-25_0247_prefill-base3-t5-243x5-lut-expand-void5-take-plus-matmul-t256-p256-p1024-p2048-c512-vs-2-bit-u32-lut-expand-and-int8-full-bitnetd-layer-8-thread-pool-parity-with-2-bit-1p06-1p26x-int8]]) showed the t5 expand is at parity with the 2-bit expand at prefill and asked: at B 2 / 4 / 8 / 16 / 32 tokens per weight pass (speculative-decode verify widths or multi-stream decode, full bitnetd layer, W 8, ctx-free), which t5 path is faster, the T-MAC gather (one 243-entry table per input row, B tables) or the void5 LUT expand + matmul? Where is the crossover B, and does the faster path stay within 1.3x of int8 (T 128) at each B?

This is not a repeat. The 11-24 base-3 notes ran only B 1 T-MAC; the 02:47 note ran only prefill (P ≥ 256); the 11-18 multi-stream and 11-23 batch-amortization notes used only the 2-bit u32 expand.

Answers:
- **Which t5 path, per B:** T-MAC (tmL_T384, B tables, one shared index build per tile, one take + row-sum per row) is fastest at **B 1 (11.52 / 10.34 ms, seed 1 / seed 2)** and **B 2 (14.31 / 14.19 ms)**. From **B 4** the void5 expand + weight-major matmul (t5vg_T256) is fastest: **18.23 / 17.42 ms at B 4, 17.61 / 17.23 at B 8, 20.02 / 21.51 at B 16, 23.42 / 23.87 at B 32**. T-MAC cost grows ~linearly in B (~4.3 ms per extra row: 22.8 / 23.8 ms at B 4, 150 / 161 ms at B 32). The expand is nearly flat in B (17–24 ms) because the weight-tile expand (~15 ms of the layer) is paid once per pass.
- **Crossover B:** paired tmL / t5vg = **0.48x / 0.45x at B 1, 0.81x / 0.81x at B 2, 1.25x / 1.37x at B 4**, then 2.23x / 2.45x (B 8), 3.65x / 4.13x (B 16), 6.40x / 6.75x (B 32). **The crossover is between B 2 and B 4** (~B 3 by linear fit: tmL ≈ 14.3 + 4.2·(B − 2) ms vs t5vg ≈ 17.6 ms). B 3 was not measured.
- **vs int8 (bar 1.3x):** best t5 / best int8 (paired) = **1.853x / 1.799x at B 1, 1.882x / 2.201x at B 2, 2.389x / 2.663x at B 4, 2.181x / 2.679x at B 8, 2.061x / 2.531x at B 16, 1.710x / 1.980x at B 32**. **Missed at every B.** The best int8 path is per-row np.dot at B 1–2 (int8d: 6.22 / 5.75 ms, 7.60 / 6.45 ms) and the weight-major matmul from B 4 on (int8g: 7.63 / 6.54 ms at B 4 up to 13.70 / 12.06 ms at B 32). int8 needs only a cast, not a LUT expand, so its cost per pass is ~7 ms vs ~17 ms for the t5 expand.
- **Side finding (affects any small-B matmul in this pool):** np.matmul with 1–2 activation rows is pathological on the 8-thread pool. int8_T128 (x @ tile.T, the prefill orientation) takes 12.74 / 11.66 ms at B 1 and 17.61 / 17.14 ms at B 2, but only 9.46 / 7.58 ms at B 4. Per-row np.dot is 2x faster at B 1–2 (int8d / int8g 0.52x, 0.40–0.47x). A micro-check (chk2.py: 8 workers x 14 tiles of 128 x 6912 f32) shows the same: matmul 13.97 ms (B 1) / 35.84 ms (B 2) vs np.dot 2.07 / 3.49 ms, while at B 4 matmul 5.34 ms beats np.dot 6.38 ms. For the same reason the expand path at B 1 needs per-row np.dot (t5vd 17.46 / 17.03 ms vs t5vg 24.00 / 22.91 ms). The weight-major orientation (tile @ x.T) beats x @ tile.T at every B ≥ 2 (t5vg / t5v 0.85–0.99x, int8g / int8 0.73–0.95x).
- **Interleaved T-MAC table (tmV, (in5/5·243, B) layout, axis-0 take of B contiguous floats + ones-matmul):** slower than tmL at every B (tmV / tmL 1.14x–2.01x). Dropped.
- **Correctness:** all t5 paths vs the float64 ternary reference on the first B of 32 fixed rows: max abs ≤ 1.4e-06, cos ≥ 0.99999999999997 (both seeds, all B). All t5 paths vs t5v_T256 at the same B: max abs ≤ 1.7e-06 (summation order only). int8 vs its own f64 reference: max abs ≤ 4.7e-06, cos 0.99999999999982.
- **Memory:** t5 weights 13,986,304 B vs 69,559,296 B int8 (4.973x) vs 17,458,176 B 2-bit (1.248x). Already known, not new. The T-MAC tables cost 1,344,276 B per input row (in 6912, the widest input). At B 32 that is 43,016,832 B, 3.08x the t5 weights themselves, so the T-MAC path's live bytes stop being small past B ~10. Scratch, all W 8 workers: tmL_T384 50,982,912 B; tmV 136–272 MB; t5v / t5vg / t5vd_T256 70,809,600 B; int8_T128 35,389,440 B.
- **Verdict: not-viable** (no new ≥ 1.5x gain; t5 is 1.7–2.7x slower than int8 at every B 1–32). For a t5-resident decode server: use T-MAC for B ≤ 2 and the expand + weight-major matmul for B ≥ 4. The B 1 gap to int8 (1.80–1.85x) is a bit wider than the 11-24 18:55 note's 1.55–1.61x. On this box, at this hour, int8d B 1 measured 5.75–6.22 ms vs 5.13 ms then, and tmL 10.3–11.5 vs 8.26 ms.

## Method

- Same layer, weights and P5 base-3 pack as the 02:47 note (pf5.py → /workspace/bench1455_1118 base.py / par.py: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gate|up / down, per-row gamma, attention stub (row-local, ctx-free)). Every trit round-trips through the base-3 pack (asserted at build). B rows go through the threaded layer as one weight pass per site.
- t5 paths: tmL_T384: B tables (B, in5/5, 243) f32 built per site call as x_pad (g, 5) @ LUT5_T, split over the pool by group slices. Per worker tile, idx = P5 tile + g·243 (intp, once per tile, shared by all B rows), then for each row b: np.take(Tab_b, idx) → np.dot(., ones) → y[b] (the B 1 t5_64_T384 kernel of 11-24, looped over b). tmV_Tb: interleaved table (in5/5·243, B) = LUT5F @ x_pad (g, 5, B), axis-0 take → (n, q, B) → np.matmul(ones, .); tile T = clamp(4096 // B, 32, 1024). t5v_T256: void5 take + int8→f32 cast + np.matmul(x, tile.T) (the 02:47 prefill path). t5vg_T256: same expand, np.matmul(tile, x.T) → y.T. t5vd_T256: same expand, per-row np.dot(tile, x[b]).
- int8 baselines in the same process: int8_T128 (par.layer8: cast + x @ tile.T), int8g_T128 (cast + tile @ x.T), int8d_T128 (cast + per-row np.dot). All use the same dispatch (ceil(out/T) tiles, ceil(tiles/W) per worker), OPENBLAS_NUM_THREADS=1, W 8 ThreadPoolExecutor on 8 cores, numpy 2.5.4.
- "Best t5" / "best int8" = the lowest median per B among the five t5 / three int8 configs. Ratios are per-round paired.
- Timing: per B, 1 warm-up, 7 rounds x 5 reps per config, config order rotated each round. Value = median of round medians; spread = min–max of round medians. Two input seeds (weights fixed by base.SEED). Total run time ~2.4 min for both seeds. python3 + numpy only; no installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench0545_1125 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 900 python3 xb5.py run$s.json > run$s.txt 2>&1; done

Script (xb5.py; imports base / par from /workspace/bench1455_1118):

    #!/usr/bin/env python3
    """Small-batch crossover for the base-3 t5 pack (5 trits/byte, 243-entry table), W=8 pool, full bitnetd layer, ctx-free, 05:45 MT 11/25.
    Base: bench0255_1125/pf5.py (layer, weights, P5 pack + exact round-trip assert, void5 LUT expand) and bench1855_1124/tm5b.py (B1 T-MAC t5).
    B = 1/2/4/8/16/32 token rows share one weight pass per site.
    tmL_T384 : T-MAC gather, B tables (B, in5/5, 243) f32 built per site call (split over the pool by group slices), per worker tile
               idx = P5 tile + g*243 (intp, built once per tile, shared by all B) -> for each b: np.take(Tab_b, idx) -> np.dot(., ones) -> y[b].
    tmV_T*   : T-MAC gather, interleaved table (in5/5*243, B) (= LUT5 @ x_pad (q,5,B), pool-split), per tile np.take(tab, idx, axis=0)
               -> (n, q, B) -> np.matmul(ones(q), .) -> y[:, tile]. Tile rows T = clamp(4096 // B, 32, 1024) (scratch n*q*B constant).
    t5v_T256 : void5 LUT expand (243 x 5-byte take) -> int8 view -> f32 tile -> np.matmul(x (B, in), tile.T) (pf5 best).
    t5vg_T256: same expand, weight-major matmul tile (n, in) @ x.T (in, B) -> y.T (out, B), transposed once per site.
    t5vd_T256: same expand, per-row gemv np.dot(tile, x[b]) for each b (np.matmul with B 1-2 rows is 7-10x slower than np.dot in the 8-thread pool).
    Baselines rerun: int8d_T128 (cast + per-row np.dot), int8_T128 (par.layer8 cast + x @ tile.T, the prefill orientation) and int8g_T128 (cast + tile @ x.T, weight-major).
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 xb5.py out.json"""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import D, F, KVH, QH, HD, rms, silu
    W, ROUNDS, REPS, WARM = 8, 7, 5, 1
    BS = (1, 2, 4, 8, 16, 32)
    base.build(); par.prep(); par.prep8()
    SITES = ("qkv", "o", "gu", "down")
    def p5(k):
        t = np.concatenate([np.sign(base.PW[n]["Wt"]) for n in par.FUSE[k]]).astype(np.int8); o, i = t.shape; i5 = -(-i // 5) * 5
        c = np.zeros((o, i5), np.uint8); c[:, :i] = np.where(t < 0, 2, t); c = c.reshape(o, i5 // 5, 5).astype(np.uint16)
        p = np.ascontiguousarray((c * np.array([1, 3, 9, 27, 81], np.uint16)).sum(-1).astype(np.uint8))
        dec = np.stack([(p.astype(np.int16) // 3 ** j) % 3 for j in range(5)], -1).reshape(o, i5)[:, :i]; dec[dec == 2] = -1
        assert np.array_equal(dec.astype(np.int8), t), k                       # exact round-trip of every trit
        g = par.FP[k]["g"]; nz = np.abs(t).max(1) > 0
        assert np.allclose(np.abs(np.concatenate([base.PW[n]["Wt"] for n in par.FUSE[k]])).max(1)[nz], g[nz])
        return dict(PR=p, g=g, split=par.FP[k]["split"], inp=i, inp5=i5)
    P5 = {k: p5(k) for k in SITES}
    def wf64(k, bits):
        if bits == 2: return np.concatenate([base.PW[n]["Wt"].astype(np.float64) for n in par.FUSE[k]])
        return np.concatenate([base.PW[n]["q8"].astype(np.float64) * float(base.PW[n]["m8"]) for n in par.FUSE[k]])
    def layer_ref(x, bits):
        x = x.astype(np.float64); r = lambda a: a / np.sqrt(np.mean(a ** 2, axis=-1, keepdims=True) + 1e-6); sg = lambda a: a / (1 + np.exp(-a))
        mm = lambda a, k: a @ wf64(k, bits).T
        qkv = mm(r(x), "qkv"); v = qkv[:, D + KVH * HD:]
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        x2 = x + mm(att, "o"); gu = mm(r(x2), "gu"); z = sg(gu[:, :F]) * gu[:, F:]
        return x2 + mm(z, "down")
    XREF = np.random.default_rng(int(os.environ.get("XSEED", "1")) + 1000).standard_normal((32, D)).astype(np.float32)
    REF = {b: layer_ref(XREF, b) for b in (2, 8)}
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT", "Wt"): base.PW[nm].pop(f, None)
    L5 = np.array([[((c // 3 ** j) % 3) for j in range(5)] for c in range(243)], np.int8); L5[L5 == 2] = -1
    LUT5V = np.ascontiguousarray(L5).view("V5").ravel()             # 243 x 5-byte rows (1,215 B)
    LUT5F = np.ascontiguousarray(L5.astype(np.float32))             # (243, 5) f32
    LUT5_T = np.ascontiguousarray(LUT5F.T)                          # (5, 243)
    F5 = -(-F // 5); BM = max(BS)
    TABL = np.empty((BM, F5, 243), np.float32)                     # per-row tables, B 32 x in 6912 -> 43,016,832 B
    TABV = np.empty(F5 * 243 * BM, np.float32)                     # interleaved (q*243, B) table, same bytes
    XPL = np.zeros((BM, F5 * 5), np.float32); XPV = np.zeros(F5 * 5 * BM, np.float32)
    OFF5 = np.arange(F5, dtype=np.intp) * 243; ONES = np.ones(F5, np.float32)
    def TV(B): return int(min(1024, max(32, 4096 // B)))
    _s = {}
    def scr(key, shape, dt):
        k = key
        if k not in _s: _s[k] = np.empty(int(np.prod(shape)), dt)
        return _s[k]
    def rows_t5v(PR, x, y, r0, r1, T, w, i):
        q = PR.shape[1]; sv = scr(("v5", w, T), (T * F5,), "V5")[:T * q].reshape(T, q); s = scr(("f5", w, T), (T * F5 * 5,), np.float32)[:T * q * 5].reshape(T, q * 5)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.take(LUT5V, PR[o0:o0+n], out=sv[:n])
            np.copyto(s[:n], sv[:n].view(np.int8).reshape(n, q * 5), casting="unsafe")
            np.matmul(x, s[:n, :i].T, out=y[:, o0:o0+n])
    def rows_t5vg(PR, xT, yT, r0, r1, T, w, i):   # weight-major orientation: tile (n, in) @ x.T (in, B) -> yT (out, B)
        q = PR.shape[1]; sv = scr(("v5", w, T), (T * F5,), "V5")[:T * q].reshape(T, q); s = scr(("f5", w, T), (T * F5 * 5,), np.float32)[:T * q * 5].reshape(T, q * 5)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.take(LUT5V, PR[o0:o0+n], out=sv[:n])
            np.copyto(s[:n], sv[:n].view(np.int8).reshape(n, q * 5), casting="unsafe")
            np.matmul(s[:n, :i], xT, out=yT[o0:o0+n])
    def rows8g(Q, xT, yT, r0, r1, T, w, i):
        _, s = par.scratch(w, T, i // 4)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.copyto(s[:n], Q[o0:o0+n], casting="unsafe")
            np.matmul(s[:n], xT, out=yT[o0:o0+n])
    def rows_t5vd(PR, x, y, r0, r1, T, w, i):     # per-row gemv: np.dot(tile (n, in), x[b]) for each of the B rows
        q = PR.shape[1]; sv = scr(("v5", w, T), (T * F5,), "V5")[:T * q].reshape(T, q); s = scr(("f5", w, T), (T * F5 * 5,), np.float32)[:T * q * 5].reshape(T, q * 5)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.take(LUT5V, PR[o0:o0+n], out=sv[:n])
            np.copyto(s[:n], sv[:n].view(np.int8).reshape(n, q * 5), casting="unsafe")
            for b in range(x.shape[0]): np.dot(s[:n, :i], x[b], out=y[b, o0:o0+n])
    def rows8d(Q, x, y, r0, r1, T, w, i):
        _, s = par.scratch(w, T, i // 4)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.copyto(s[:n], Q[o0:o0+n], casting="unsafe")
            for b in range(x.shape[0]): np.dot(s[:n], x[b], out=y[b, o0:o0+n])
    def fmmd(key, x, kind, T):
        P = P5[key] if kind == "t5vd" else par.F8[key]; R = P["PR"] if kind == "t5vd" else P["Q"]; out = R.shape[0]; B = x.shape[0]
        i = P["inp"] if kind == "t5vd" else R.shape[1]; fn = rows_t5vd if kind == "t5vd" else rows8d
        x = np.ascontiguousarray(x, np.float32); y = np.empty((B, out), np.float32); nt = -(-out // T); per = -(-nt // W)
        futs = [par.pool(W).submit(fn, R, x, y, w * per * T, min(out, (w + 1) * per * T), T, w, i) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
        y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
    def fmmg(key, x, kind, T):
        P = P5[key] if kind == "t5vg" else par.F8[key]; R = P["PR"] if kind == "t5vg" else P["Q"]; out = R.shape[0]; B = x.shape[0]
        i = P["inp"] if kind == "t5vg" else R.shape[1]; fn = rows_t5vg if kind == "t5vg" else rows8g
        xT = np.ascontiguousarray(np.asarray(x, np.float32).T); yT = np.empty((out, B), np.float32); nt = -(-out // T); per = -(-nt // W)
        futs = [par.pool(W).submit(fn, R, xT, yT, w * per * T, min(out, (w + 1) * per * T), T, w, i) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
        y = np.ascontiguousarray(yT.T); y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
    def layerg(x, kind, T):
        mm = lambda k, a: (fmmd if kind.endswith("d") else fmmg)(k, a, kind, T)
        h1 = rms(x); q, kk, v = mm("qkv", h1)
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        o = mm("o", att)[0]; x2 = x + o; h2 = rms(x2)
        g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
        return x2 + mm("down", z)[0]
    def rows_tmL(PR, tab, y, r0, r1, T, w, B):
        q = PR.shape[1]; idx = scr(("iL", w, T), (T * F5,), np.intp)[:T * q].reshape(T, q); gv = scr(("gL", w, T), (T * F5,), np.float32)[:T * q].reshape(T, q)
        off = OFF5[:q]; one = ONES[:q]
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0); np.add(PR[o0:o0+n], off, out=idx[:n])
            for b in range(B):
                np.take(tab[b], idx[:n], out=gv[:n]); np.dot(gv[:n], one, out=y[b, o0:o0+n])
    def rows_tmV(PR, tab, y, r0, r1, T, w, B):
        q = PR.shape[1]; idx = scr(("iV", w, T), (T * F5,), np.intp)[:T * q].reshape(T, q); gv = scr(("gV", w, T, B), (T * F5 * B,), np.float32)[:T * q * B].reshape(T, q, B)
        acc = scr(("aV", w, T, B), (T * B,), np.float32)[:T * B].reshape(T, B); off = OFF5[:q]; one = ONES[:q]
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0); np.add(PR[o0:o0+n], off, out=idx[:n])
            np.take(tab, idx[:n], axis=0, out=gv[:n]); np.matmul(one, gv[:n], out=acc[:n]); y[:, o0:o0+n] = acc[:n].T
    def bldL(xp, tab, B, g0, g1, q):   # tab[b, g, :] = x_pad[b, 5g:5g+5] @ LUT5_T
        for b in range(B): np.matmul(xp[b, g0 * 5:g1 * 5].reshape(g1 - g0, 5), LUT5_T, out=tab[b, g0:g1])
    def bldV(xv, tv, g0, g1):          # tv[g] = LUT5F (243,5) @ xv[g] (5,B)
        np.matmul(LUT5F, xv[g0:g1], out=tv[g0:g1])
    def fmm5(key, x, kind, T):
        P = P5[key]; PR = P["PR"]; out = PR.shape[0]; i = P["inp"]; q = PR.shape[1]; B = x.shape[0]
        x = np.ascontiguousarray(x, np.float32); y = np.empty((B, out), np.float32); pool = par.pool(W)
        gper = -(-q // W)
        if kind == "tmL":
            xp = XPL[:B, :q * 5]; xp[:, :i] = x; xp[:, i:] = 0; tab = TABL[:B, :q]
            bf = [pool.submit(bldL, xp, tab, B, w * gper, min(q, (w + 1) * gper), q) for w in range(W) if w * gper < q]
            for f in bf: f.result()
            arg = TABL[:B, :q].reshape(B, q * 243); fn = rows_tmL; extra = B
        elif kind == "tmV":
            xv = XPV[:q * 5 * B].reshape(q, 5, B); xv[:, :, :] = 0; xv.reshape(q * 5, B)[:i] = x.T
            tv = TABV[:q * 243 * B].reshape(q, 243, B)
            bf = [pool.submit(bldV, xv, tv, w * gper, min(q, (w + 1) * gper)) for w in range(W) if w * gper < q]
            for f in bf: f.result()
            arg = tv.reshape(q * 243, B); fn = rows_tmV; extra = B
        else: arg = x; fn = rows_t5v; extra = i
        nt = -(-out // T); per = -(-nt // W)
        futs = [pool.submit(fn, PR, arg, y, w * per * T, min(out, (w + 1) * per * T), T, w, extra) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
        y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
    def layer5(x, kind, T):
        mm = lambda k, a: fmm5(k, a, kind, T)
        h1 = rms(x); q, kk, v = mm("qkv", h1)
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        o = mm("o", att)[0]; x2 = x + o; h2 = rms(x2)
        g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
        return x2 + mm("down", z)[0]
    CFG = {"tmL_T384": lambda x: layer5(x, "tmL", 384), "tmV_Tb": lambda x: layer5(x, "tmV", TV(x.shape[0])),
           "t5v_T256": lambda x: layer5(x, "t5v", 256), "t5vg_T256": lambda x: layerg(x, "t5vg", 256), "t5vd_T256": lambda x: layerg(x, "t5vd", 256),
           "int8_T128": lambda x: par.layer8(x, 128, W), "int8g_T128": lambda x: layerg(x, "8g", 128), "int8d_T128": lambda x: layerg(x, "8d", 128)}
    BITS = {c: 8 if c.startswith("int8") else 2 for c in CFG}
    T5 = ("tmL_T384", "tmV_Tb", "t5v_T256", "t5vg_T256", "t5vd_T256"); I8 = ("int8_T128", "int8g_T128", "int8d_T128")
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    def main():
        seed = int(os.environ.get("XSEED", "1")); res = {"xseed": seed, "numpy": np.__version__, "cpus": os.cpu_count(), "W": W, "BS": BS}
        W2 = sum(par.FP[k]["PR"].nbytes + par.FP[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES)
        res["bytes"] = dict(t5=W5, u2=W2, int8=W8, int8_over_t5=W8 / W5, u2_over_t5=W2 / W5, tab_per_row=F5 * 243 * 4, tab_B32=BM * F5 * 243 * 4,
                            tmL_scr=W * 384 * F5 * 12, tmV_scr={B: W * TV(B) * F5 * (8 + 4 * B) for B in BS}, t5v_scr=W * 256 * F5 * 25, t5vg_scr=W * 256 * F5 * 25, int8_scr=W * 128 * F * 4 + W * 128 * (F // 4) * 4)
        print("bytes", res["bytes"], flush=True)
        err = {}
        for B in BS:
            xb = XREF[:B]; e = {}
            for c, f in CFG.items():
                y = f(xb); r = REF[BITS[c]][:B]; e[c] = dict(mab=float(np.abs(y - r).max()), cos=cos(y, r))
            yv = CFG["t5v_T256"](xb)
            for c in ("tmL_T384", "tmV_Tb", "t5vg_T256", "t5vd_T256"): e[c]["vs_t5v_mab"] = float(np.abs(CFG[c](xb) - yv).max())
            err[f"B{B}"] = e
            print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.15f}", f"{v.get('vs_t5v_mab', 0):.2e}") for c, v in e.items()}, flush=True)
        res["err"] = err; names = list(CFG); res["time"] = {}
        for B in BS:
            xb = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32); rm = {c: [] for c in names}
            for c in names:
                for _ in range(WARM): CFG[c](xb)
            for rd in range(ROUNDS):
                for c in names[rd % len(names):] + names[:rd % len(names)]:
                    s = []
                    for _ in range(REPS):
                        t0 = time.perf_counter(); CFG[c](xb); s.append((time.perf_counter() - t0) * 1e3)
                    rm[c].append(float(np.median(s)))
            o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), us_tok=float(np.median(v)) * 1e3 / B, rm=v) for c, v in rm.items()}
            rat = {}
            pairs = [(a, b) for a in T5 for b in I8] + [("tmL_T384", "t5vg_T256"), ("tmL_T384", "t5vd_T256"), ("tmL_T384", "t5v_T256"), ("t5vd_T256", "t5vg_T256"), ("int8d_T128", "int8g_T128"), ("tmV_Tb", "tmL_T384"), ("t5vg_T256", "t5v_T256"), ("int8g_T128", "int8_T128")]
            for a, b in pairs:
                v = [p / q for p, q in zip(rm[a], rm[b])]; rat[f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v))
            best = min(T5, key=lambda c: o[c]["med"]); b8 = min(I8, key=lambda c: o[c]["med"])
            v = [p / q for p, q in zip(rm[best], rm[b8])]; rat["best_t5/best_int8"] = dict(r=o[best]["med"] / o[b8]["med"], lo=min(v), hi=max(v), a=best, b=b8)
            res["time"][f"B{B}"] = dict(cfg=o, ratios=rat, best_t5=best, best_int8=b8, TV=TV(B))
            for c in names: print(f"B{B} {c:10s} {o[c]['med']:8.2f} ms [{o[c]['lo']:.2f}-{o[c]['hi']:.2f}] {o[c]['us_tok']:.1f} us/tok", flush=True)
            for k, v in rat.items(): print(f"B{B} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.2f}-{v['hi']:.2f})", flush=True)
            print(f"B{B} best t5 path {best}, best int8 {b8}", flush=True)
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    main()

Micro-check (chk2.py, the small-B matmul pathology; same pool, OPENBLAS_NUM_THREADS=1):

    import numpy as np, time
    from concurrent.futures import ThreadPoolExecutor
    P=ThreadPoolExecutor(8); S=[np.random.rand(128,6912).astype(np.float32) for _ in range(8)]
    def run(B, mode):
        x=np.random.rand(B,6912).astype(np.float32); xT=np.ascontiguousarray(x.T); Y=[np.empty((B,128),np.float32) for _ in range(8)]; YT=[np.empty((128,B),np.float32) for _ in range(8)]
        def w(k):
            for _ in range(14):
                s=S[k]
                if mode=="mm": np.matmul(x, s.T, out=Y[k])
                elif mode=="mmg": np.matmul(s, xT, out=YT[k])
                else:
                    for b in range(B): np.dot(s, x[b], out=Y[k][b])
        ts=[]
        for _ in range(15):
            t0=time.perf_counter(); [f.result() for f in [P.submit(w,k) for k in range(8)]]; ts.append(time.perf_counter()-t0)
        return np.median(ts)*1e3
    for B in (1,2,4,8):
        print(B, {m: round(run(B,m),2) for m in ("mm","mmg","dot")})

## Results

Layer time in ms, median of 7 round medians [min–max round median], seed 1 / seed 2 (bold = fastest t5 and fastest int8 per B):

| config | B 1 | B 2 | B 4 | B 8 | B 16 | B 32 |
|---|---|---|---|---|---|---|
| tmL_T384 (T-MAC, B tables) | **11.52 [8.2–12.0] / 10.34 [9.2–11.9]** | **14.31 [12.6–16.6] / 14.19 [14.0–18.1]** | 22.77 [22.0–24.4] / 23.80 [23.3–25.8] | 39.20 [36.8–42.7] / 42.28 [41.7–51.9] | 73.04 [70.0–82.8] / 88.94 [80.5–110.4] | 149.88 [132.2–160.0] / 161.23 [152.1–195.3] |
| tmV_Tb (T-MAC, interleaved) | 15.78 [12.9–18.0] / 13.95 [12.6–16.8] | 28.58 [27.9–29.5] / 28.58 [27.5–30.9] | 36.66 [35.5–40.5] / 36.56 [35.8–37.8] | 75.61 [69.0–85.2] / 76.89 [72.3–82.4] | 114.68 [103.6–129.2] / 128.22 [109.7–149.7] | 170.51 [160.2–191.0] / 189.00 [148.8–210.0] |
| t5v_T256 (expand, x @ tile.T) | 23.47 [22.6–25.1] / 24.17 [23.3–28.0] | 20.00 [17.4–22.3] / 17.69 [17.0–22.3] | 19.27 [17.7–20.4] / 18.23 [17.8–20.3] | 19.62 [18.8–20.0] / 19.01 [18.2–21.4] | 23.60 [21.0–24.7] / 21.70 [20.1–26.3] | 26.86 [23.0–28.6] / 26.02 [22.8–28.5] |
| t5vg_T256 (expand, tile @ x.T) | 24.00 [22.7–25.6] / 22.91 [22.5–28.9] | 17.62 [16.1–19.6] / 17.56 [16.3–19.5] | **18.23 [16.6–19.5] / 17.42 [16.7–18.4]** | **17.61 [17.1–18.9] / 17.23 [16.4–18.3]** | **20.02 [18.2–21.8] / 21.51 [19.0–22.5]** | **23.42 [21.9–25.4] / 23.87 [22.4–28.3]** |
| t5vd_T256 (expand, per-row dot) | 17.46 [16.3–20.4] / 17.03 [16.4–23.3] | 19.59 [19.2–23.3] / 19.83 [18.7–20.6] | 25.49 [24.5–52.2] / 24.96 [24.3–25.8] | 34.81 [33.8–36.6] / 33.26 [32.6–34.6] | 57.77 [55.2–76.2] / 57.27 [56.5–66.8] | 99.26 [93.1–124.8] / 95.81 [89.8–112.7] |
| int8_T128 (x @ tile.T) | 12.74 [11.5–13.4] / 11.66 [10.9–13.5] | 17.61 [16.9–18.5] / 17.14 [16.2–19.4] | 9.46 [8.7–16.7] / 7.58 [7.3–8.4] | 10.13 [9.4–10.4] / 8.76 [8.3–9.1] | 11.67 [10.8–17.5] / 10.50 [9.7–11.5] | 14.67 [13.6–21.4] / 13.46 [11.7–14.8] |
| int8g_T128 (tile @ x.T) | 11.94 [10.8–13.4] / 11.05 [9.9–12.3] | 16.28 [15.5–17.3] / 16.22 [16.0–17.7] | **7.63 [7.4–10.4] / 6.54 [6.1–7.0]** | **8.07 [7.3–9.7] / 6.43 [6.2–6.8]** | **9.71 [9.1–11.9] / 8.50 [7.8–9.4]** | **13.70 [12.7–21.8] / 12.06 [10.7–14.5]** |
| int8d_T128 (per-row dot) | **6.22 [5.6–6.5] / 5.75 [5.0–6.3]** | **7.60 [6.5–8.5] / 6.45 [6.4–7.4]** | 10.80 [10.6–11.7] / 9.53 [8.9–9.9] | 16.74 [15.8–18.2] / 14.71 [14.5–15.5] | 29.92 [28.5–37.2] / 26.71 [24.9–36.3] | 53.28 [51.8–70.0] / 49.61 [46.7–55.8] |

Per-token cost of the best t5 path: 11,522 / 10,341 µs at B 1, 7,155 / 7,097 at B 2, 4,558 / 4,354 at B 4, 2,201 / 2,154 at B 8, 1,251 / 1,345 at B 16, 732 / 746 at B 32. Best int8: 6,217 / 5,747, 3,802 / 3,224, 1,908 / 1,635, 1,009 / 804, 607 / 531, 428 / 377.

Paired ratios (seed 1 / seed 2, per-round range in brackets):

| ratio | B 1 | B 2 | B 4 | B 8 | B 16 | B 32 |
|---|---|---|---|---|---|---|
| **tmL / t5vg (crossover)** | **0.480x (0.36–0.53) / 0.451x (0.38–0.47)** | **0.812x (0.65–0.98) / 0.808x (0.72–1.07)** | **1.249x (1.16–1.37) / 1.367x (1.35–1.47)** | **2.227x (1.98–2.43) / 2.453x (2.31–3.01)** | **3.649x (3.47–4.11) / 4.134x (3.74–5.21)** | **6.399x (5.94–6.59) / 6.753x (6.10–7.50)** |
| tmL / t5vd | 0.660x (0.49–0.70) / 0.607x (0.46–0.63) | 0.731x (0.57–0.87) / 0.716x (0.70–0.88) | 0.893x / 0.954x | 1.126x / 1.271x | 1.264x / 1.553x | 1.510x / 1.683x |
| **best t5 / best int8 (bar ≤ 1.3x)** | **1.853x (1.46–2.01) / 1.799x (1.63–2.07)** | **1.882x (1.71–2.31) / 2.201x (1.99–2.44)** | **2.389x (1.87–2.63) / 2.663x (2.47–2.85)** | **2.181x (1.81–2.48) / 2.679x (2.42–2.88)** | **2.061x (1.53–2.39) / 2.531x (2.20–2.76)** | **1.710x (1.12–1.81) / 1.980x (1.64–2.35)** |
| tmV / tmL | 1.369x / 1.349x | 1.997x / 2.014x | 1.610x / 1.536x | 1.929x / 1.819x | 1.570x / 1.442x | 1.138x / 1.172x |
| t5vg / t5v | 1.023x / 0.948x | 0.881x / 0.993x | 0.946x / 0.955x | 0.897x / 0.907x | 0.848x / 0.991x | 0.872x / 0.918x |
| t5vd / t5vg | 0.727x / 0.743x | 1.111x / 1.129x | 1.398x / 1.433x | 1.977x / 1.930x | 2.886x / 2.662x | 4.238x / 4.013x |
| int8g / int8 | 0.937x / 0.948x | 0.925x / 0.947x | 0.807x / 0.863x | 0.797x / 0.734x | 0.832x / 0.810x | 0.934x / 0.896x |
| int8d / int8g | 0.521x / 0.520x | 0.467x / 0.398x | 1.415x / 1.457x | 2.074x / 2.286x | 3.081x / 3.143x | 3.890x / 4.114x |

Best paths (both seeds agree at every B): t5 tmL_T384 at B 1–2 and t5vg_T256 at B 4–32. int8 int8d_T128 at B 1–2 and int8g_T128 at B 4–32. tmV tile T: 1024 (B 1–4), 512 (B 8), 256 (B 16), 128 (B 32).

Error (first B of 32 fixed rows vs own float64 reference, seed 1 / seed 2): tmL max abs 5.6e-07–8.0e-07, tmV 3.6e-07–1.4e-06, t5v / t5vg / t5vd 8.7e-07–1.3e-06, all cos ≥ 0.99999999999997. int8 2.6e-06–4.7e-06, cos ≥ 0.99999999999982. t5 paths vs t5v_T256: max abs ≤ 1.7e-06 (t5vg is bit-identical to t5v).

Micro-check chk2.py (8 workers x 14 tiles of 128 x 6912 f32, median of 15, ms): B 1 matmul x @ s.T 13.97, s @ x.T 13.76, np.dot 2.07. B 2: 35.84 / 28.52 / 3.49. B 4: 5.34 / 5.05 / 6.38. B 8: 5.57 / 4.09 / 12.27. On one thread the same B 1 tile is 0.23 ms for all three, so the B 1–2 matmul slowdown appears only under concurrency.

Bytes: weights per layer 13,986,304 B t5 vs 17,458,176 B 2-bit vs 69,559,296 B int8 (int8/t5 4.973x, 2-bit/t5 1.248x). T-MAC tables 1,344,276 B per row at in 6912; B 32: 43,016,832 B. Scratch, all 8 workers: tmL_T384 50,982,912 B; tmV_Tb 135,954,432 (B 1), 181,272,576 (B 2), 271,908,864 (B 4), 226,590,720 (B 8), 203,931,648 (B 16), 192,602,112 B (B 32); t5v / t5vg / t5vd_T256 70,809,600 B; int8_T128 35,389,440 B.

## Pair files this wave

No pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) for this wave reached this lane, so none are linked. The depth-1 clone of JuniorLLM (05:53 MT) showed this lane's own 5b049f0 (02:55 MT, the 02:47 note) as the newest commit. The only other commit in the last 6 hours is unattributed cloudcover95 23ed5da "rails/cleanclone: shallow clone guard" (02:54 MT, rails/cleanclone only). Nothing in the last 6 hours touches junior_osai/vault/findings, and no Trit Pack, Handshake or LLC commit appears. This lane did not read JuniorHome this wave.

## Next question

Rotate away from kernel choice to pool overhead. Decode-layer dispatch and sync overhead on the W 8 pool: today a layer makes four submit/result rounds (one per site), plus main-thread rms, silu, table builds and splits in between. Compare that with one persistent task per worker per layer that walks all four sites and syncs on a threading.Barrier between them, with worker 0 doing rms / silu / the residual adds. Run at B 1 and B 4 on the best paths found here (t5 tmL / t5vg, int8d / int8g), full bitnetd layer, ctx-free. First measure an empty-work layer (same dispatch, no math) to size the fixed overhead. Then ask: does the barrier layer cut layer time by ≥ 1.1x for int8 or t5, and does it narrow t5's 1.8–2.7x gap to int8? No earlier note measured pool dispatch / sync overhead or a barrier-synchronized layer.

close: 97b49f071bbfd469
