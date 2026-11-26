---
local: true
date: 2026-11-25
wave: "2026-11-25 18:55 slot (fired 18:43 MT, ran ~18:43–19:00 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (guided self-scheduling on the W 8 decode layer, full bitnetd layer, ctx-free. A precomputed per-site chunk list claimed by index from a shared itertools.count, with big chunks Tb (T, or 2T for int8 and t5vg gu) while at least W·T rows remain and T/2 after that (gd2), clears both bars only for t5 at B 4: t5vg_T256 gd2 / dy_T 0.929x (1.08x) and gd2 / st_T 0.869x (1.15x), bit-identical to st_T. At B 1 tmL_T384 gd2 is 1.03x slower than dy_T (0.973x of st_T). Three-level lists (gd3, T/4 tail) and gu-only 2T (gdL) lose to dy_T everywhere for t5 (1.07–1.13x slower). int8 gets no 1.1x from any guided list (best gd2 0.981x B 1, gd3 1.023x B 4 vs st_T), while plain dy_T gave int8 0.844x at B 4 in this wave. Off-CPU share grows with claim count (t5 B 1 0.10 st -> 0.33 gd3). t5 / int8 gap, best vs best paired per round: B 1 1.543x, B 4 2.214x (14:50: 1.49x / 2.12x), far above the 1.3x bar. No 1.5x speed or memory win for ternary; the 4.973x t5 weight-memory win vs int8 is already known)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, load-imbalance, guided-self-scheduling, chunk-schedule, dynamic-scheduling, tile-size, per-worker-busy, thread-time, off-cpu, base-3, 1p6-bit, five-trit-pack, tmac, lut-expand, int8, numpy, not-viable]
---

# Guided self-scheduling on the W 8 decode layer: a two-level chunk list gives t5 1.15x over static at B 4 (only), three-level lists lose, int8 gains nothing, and t5's gap to int8 stays 1.5–2.2x

The 14:50 note ([[2026-11-25_1450_decode-layer-per-worker-load-imbalance-dynamic-tile-claiming-shared-counter-t-t2-t4-w8-pool-b1-b4-t5-tmac-expand-vs-int8-full-bitnetd-layer-t5-1p12-1p14x-int8-none-gap-1p5-2p1x]]) found that dynamic claiming at the native tile T speeds t5 up 1.12–1.14x (not int8), that smaller tiles everywhere lose to per-tile Python / GIL cost, and asked: guided self-scheduling on the W 8 decode layer. Workers claim large tiles first (T, or 2T for int8 / t5vg gu) from the shared counter and switch to T/2 and T/4 tiles only when fewer than W·T rows remain (a two- or three-level chunk schedule precomputed per site, claimed by index), plus longest-site-first ordering for the fused gu site. Does guided claiming beat dy_T by ≥ 1.05x and st_T by ≥ 1.15x for t5 tmL_T384 (B 1) and t5vg_T256 (B 4)? Does it finally help int8 (≥ 1.1x vs st_T)? What is left of t5's 1.49x / 2.12x gap to int8? Record per-worker wall vs thread_time to check whether int8's ~40% off-CPU share grows with chunk count.

This is not a repeat. No earlier note tested guided or multi-level chunk schedules, or chunk sizes larger than the native tile (2T). The 14:50 note used one tile size per mode (T, T/2 or T/4) for the whole site.

Answers:
- **t5, does guided beat dy_T by ≥ 1.05x and st_T by ≥ 1.15x?** **Only at B 4, and only the two-level list.** t5vg_T256 B 4 gd2 / dy_T 0.929 / 0.957 / 0.930 / 0.923 (4 runs), median **0.929x (1.08x)**; gd2 / st_T 0.880 / 0.900 / 0.858 / 0.832, median **0.869x (1.151x)**, just over the bar. It is bit-identical to st_T. tmL_T384 B 1 gd2 / dy_T **1.033x** (slower) and gd2 / st_T 0.973x (1.03x), so B 1 fails both bars. The three-level gd3 and gu-only gdL lose to dy_T at both B: B 1 1.125x / 1.106x slower, B 4 1.080x / 1.069x slower. They are no better than static (0.99–1.04x of st_T).
- **Why gd2 wins at B 4 but not at B 1:** at B 4 the fused gu site (13,824 rows) runs as 24 chunks of 2T = 512 plus 12 of 128. That roughly halves the claims on the longest site, and its span falls from 6,691 µs (dy_T) to 6,133 µs. The o / down sites (2,560 rows) get 3 × 256 + 14 × 128 and reach 8 of 8 active workers (span o 1,855 -> 1,727, down 4,010 -> 3,663 µs). At B 1 tmL has Tb = T = 384 everywhere. o / down (2,560 rows < W·T = 3,072) therefore run entirely in 192-row chunks, and qkv / o spans grow (qkv 1,076 -> 1,196, o 765 -> 906 µs) because the per-chunk table gather plus dot costs more than the idle time it removes. Imbalance does fall (o 1.40 -> 1.24).
- **Longest-site-first (gdL):** the layer's four sites are data-dependent (qkv -> o -> gu -> down), so sites cannot be reordered. "Longest-site-first" was implemented as giving only the longest site (gu) 2T chunks, with qkv / o / down at Tb = T and the gd3 tail. At B 1 for tmL this cuts gu claims from 50 to 34 and gu span from 3,161 to 2,866 µs vs gd3. The T/4 tail on the other sites still loses (gdL 1.106x of dy_T). For t5vg B 4 gdL is the same schedule as gd3 (gu already had 2T), so it serves as a noise check: 0.988x vs 0.994x of st_T. For int8, gdL (qkv / o / down at T 128 + tail) is the worst guided mode: 1.049x B 1, 1.149x B 4 vs st_T.
- **int8, ≥ 1.1x vs st_T?** **No.** Best guided: B 1 gd2 0.981x (1.02x), B 4 gd3 1.023x (slower), gd2 1.073x, gdL 1.149x. The 2T = 256 chunks hurt the weight-major 8g matmul at B 4: the gu site span is 2,592 µs for dy_T vs 2,594 µs for gd2 with fewer claims, while the down span is 1,413 -> 2,577 µs and off-CPU share rises. In this wave plain dy_T gave int8 B 4 **0.844x** (1.18x; 14:50 had 0.941x, and one round reached 1.63x, so the int8 B 4 dy_T number is noisy). int8 B 1 dy_T is 1.035x.
- **Wall vs thread_time (off-CPU share = 1 − Σ thread_time / Σ busy wall over the four sites, median of 4 runs):** for t5 it grows with claim count. B 1: st_T 0.10 (60 tiles), dy_T 0.15, gd2 0.24 (88 claims), gdL 0.31 (100), gd3 0.33 (116). B 4: st 0.14, dy 0.19, gd2 0.18 (92), gd3 / gdL 0.30 (120). For int8 at B 1 it tracks claims: gd2 0.27 (125) < st 0.34 < gd3 0.35 (153) < gdL 0.40 (182) < dy 0.45 (178). At B 4 it also grows with chunk *size*: gd2 0.36 at 125 claims vs dy_T 0.20 at 178, because 256-row int8 chunks spend longer off-CPU (down site slowest worker 2,454 µs wall vs 1,119 µs CPU). So int8's off-CPU share does grow with chunk count, and also with 2T chunk size, and it does not drop below the static split except for B 1 gd2.
- **What is left of the gap?** Nothing closes. Best t5 / best int8, paired per round, median of 4: **B 1 1.543x** (t5 dy_T or gd2 vs int8 gd2 / gd3) and **B 4 2.214x** (t5 gd2 vs int8 dy_T). At 14:50 these were 1.49x / 2.12x. The same-scheme ratio t5_gd2 / i8_gd2 at B 4 (1.749x) is lower only because gd2 slows int8 down. Both stay far above the 1.3x bar.
- **Error / bit identity:** at B 1 every mode is bit-identical to its st_T layer in all 4 runs (rows are independent np.dot / gather-dot calls). At B 4 t5 gd2 is bit-identical (chunk sizes 512 / 256 / 128 do not change the t5vg matmul result here), while gd3 / gdL differ by max |Δ| 1.4e-06 (64-row chunks). All int8 guided modes differ by 4.2–4.4e-06 (matmul blocking at 256 / 64 / 32 rows). vs float64 refs: t5 max |err| 5.57e-07 (B 1) / 1.23e-06 (B 4 st / dy / gd2), 1.06e-06 (gd3 / gdL), cos 0.99999999999998–0.99999999999999; int8 3.19e-06 / 4.42e-06 (guided 3.5–3.7e-06), cos 0.9999999999998.
- **Bytes:** weights per layer 13,986,304 B t5 vs 69,559,296 B int8 (4.973x), unchanged. Scratch for all 8 workers as implemented (one buffer per worker per chunk size): tmL st / dy 50,982,912 B, gd2 76,474,368, gd3 89,220,096, gdL 191,185,920 (+ B 1 table 1,344,276 B). t5vg st / dy 70,809,600, gd2 247,833,600, gd3 / gdL 265,536,000. int8 st / dy 35,389,440, gd2 88,473,600, gd3 97,320,960, gdL 132,710,400. If one max-size buffer per worker were shared, that would be 50.98 / 101.97 MB tmL, 141.62 MB t5vg and 70.78 MB int8. The t5vg gd2 win costs 2x scratch (2T gu chunks).

Bar used: a scheduling change counts if guided beats dy_T by ≥ 1.05x (ratio ≤ 0.952) and st_T by ≥ 1.15x (ratio ≤ 0.870) for t5, and ≥ 1.1x vs st_T (≤ 0.909) for int8, as the question set. Ternary is "viable" only if t5 gets within this series' 1.3x gap-to-int8 bar, or wins ≥ 1.5x on speed or memory vs int8 at acceptable error. The known 4.973x weight-memory win does not count as new. gd2 clears the t5 bars only at B 4 (0.929 / 0.869), not at B 1. int8 gets nothing. t5 stays 1.54–2.21x slower than int8, so the verdict is **not-viable**.

Box note: this wave's absolute layer times are lower than at 14:50 (t5 B 1 st_T 8.75 vs 11.77 ms, int8 B 1 5.52 vs 6.96 ms; load avg 1.8–3.4 at start, other agents' jobs varying). All comparisons are paired per round within a run.

## Method

- Same layer, weights, P5 base-3 pack, kernels and float64 references as the 05:45 / 10:41 / 14:50 notes. gd.py execs /workspace/bench0545_1125/xb5.py without its trailing main() call (-> /workspace/bench1455_1118 base.py / par.py: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gu / down; output rows per site 3840 / 2560 / 13824 / 2560). Kernels rows_tmL, rows_t5vg, rows8d and rows8g are unchanged. Each guided chunk calls the kernel with its tile argument = chunk size, so each claim is exactly one kernel tile (scratch is keyed per chunk size).
- Paths (the 05:45 best, as at 14:50): B 1: t5 tmL (T-MAC 243-entry B tables, T 384), int8 8d (cast + per-row np.dot, T 128). B 4: t5 t5vg (void5 LUT expand + tile @ x.T, T 256), int8 8g (cast + tile @ x.T, T 128).
- Modes, one submit/result round per site (tmL keeps its static table-build round before it; glue on the main thread as in xb5):
  - st_T: xb5's static split, nt = ceil(out / T), per = ceil(nt / W) contiguous tiles per worker (baseline).
  - dy_T: 14:50's dynamic claiming, 8 tasks pulling j = next(cnt), rows [j·T, (j+1)·T).
  - gd2: per-site chunk list precomputed once: chunks of Tb while remaining rows ≥ W·T, then T/2 chunks to the end. Tb = 2T for int8 (all sites) and for t5vg's gu site, else T. 8 tasks claim list indices j = next(cnt) until j ≥ len(list).
  - gd3: Tb while remaining ≥ W·T, T/2 while remaining ≥ W·T/2, then T/4.
  - gdL ("longest-site-first"): the gd3 schedule, but only the longest site (fused gu) gets Tb = 2T, on every path; qkv / o / down use Tb = T. Sites cannot be reordered: each site's input depends on the previous one.
- Chunk lists (claims per site qkv / o / gu / down): tmL B 1 gd2 17 / 14 / 43 / 14, gd3 24 / 21 / 50 / 21, gdL 24 / 21 / 34 / 21. t5vg B 4 gd2 22 / 17 / 36 / 17, gd3 = gdL 29 / 24 / 43 / 24. int8 gd2 24 / 19 / 63 / 19, gd3 31 / 26 / 70 / 26, gdL 44 / 34 / 70 / 34. Static / dynamic at T: tmL 10 / 7 / 36 / 7, t5vg 15 / 10 / 54 / 10, int8 30 / 20 / 108 / 20 tiles. Full size mix in each run's JSON "chunks".
- Every task appends (site, worker, start, end relative to the site's submit, time.thread_time delta, chunks) in all modes. Per site: span = last end, slowest = max worker busy, mean = mean busy over all 8 workers, imbalance = slowest / mean, cpu of slowest = thread_time of the slowest worker, off-CPU share = 1 − Σ thread_time / Σ busy. Profile = 25 extra layers per config after the timing rounds; per-run value = median over the 25.
- Timing per B: 1 warm-up, then 7 rounds x 5 reps per config, config order rotated each round. Value = median of round medians. Spread = min–max of round medians. Ratios are paired per round. Two input seeds, each run twice (run1 / run2 = seeds 1 / 2, rep1 / rep2 = seeds 1 / 2 again). Tables give the median of the 4 run medians, with per-run values and the range of round ratios across all runs.
- Box: 8 CPUs, Python 3.13.5, numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 ThreadPoolExecutor, load avg ~1.8–3.4. No network, nothing bound, no installs.

Command:

    cd /workspace/bench1855_1125 && for r in "1 run1" "2 run2" "1 rep1" "2 rep2"; do set -- $r; OPENBLAS_NUM_THREADS=1 XSEED=$1 python3 gd.py $2.json > $2.txt 2>&1; done && python3 agg.py > agg.txt && python3 tab.py > tables.md

Script (/workspace/bench1855_1125/gd.py; execs /workspace/bench0545_1125/xb5.py, whose full text is in the 05:45 note):

    #!/usr/bin/env python3
    """Guided self-scheduling (multi-level chunk lists) on the W=8 decode layer, full bitnetd layer, ctx-free, 18:4x MT 11/25.
    Base: /workspace/bench0545_1125/xb5.py (weights, P5 base-3 pack, kernels rows_tmL / rows_t5vg / rows8d / rows8g, float64 refs);
    its source is exec'd here without its trailing main() call (same kernels/weights as the 14:50 dyn.py).
    Modes per site (one submit/result round per site; tmL keeps its static table-build round):
      st_T : static ceil split as xb5 (baseline)          dy_T : shared itertools.count over T-row tiles (14:50 winner for t5)
      gd2  : precomputed chunk list per site: big chunks Tb while remaining >= W*T, then T/2 chunks (two-level)
      gd3  : Tb while rem >= W*T, T/2 while rem >= W*T/2, then T/4 (three-level)
      gdL  : gd3 schedule, but only the longest site (fused gu) gets Tb = 2T on every path; qkv/o/down use Tb = T
      Tb = 2T for int8 (all sites) and for t5vg's gu site, else T. Workers claim chunk index j = next(cnt) from the list.
      Each chunk calls the kernel with tile = chunk size (scratch keyed per chunk size), so 1 kernel tile per claim.
    Every task records wall start/end, time.thread_time delta and chunk count -> per-site slowest, mean, imbalance, off-CPU share.
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 gd.py out.json"""
    import json, os, sys, time, itertools
    import numpy as np
    SRC = open("/workspace/bench0545_1125/xb5.py").read(); SRC = SRC[:SRC.rstrip().rfind("main()")]
    _argv = sys.argv; exec(compile(SRC, "xb5.py", "exec")); sys.argv = _argv
    ROUNDS, REPS, PROF = 7, 5, 25
    pc, tt = time.perf_counter, time.thread_time
    def chunks(out, T, Tb, levels):
        lv = [(Tb, W * T)] + ([(T // 2, W * T // 2)] if levels >= 3 else []) + [(T // 2 if levels == 2 else T // 4, 0)]
        L = []; o = 0
        for sz, thr in lv:
            while o < out and out - o >= thr:
                c = min(sz, out - o); L.append((o, o + c, sz)); o += c
        assert o == out
        return L
    _CL = {}
    def chunk_list(site, kind, out, T, mode):
        k = (site, kind, out, T, mode)
        if k not in _CL:
            if mode == "gdL": Tb = 2 * T if site == "gu" else T
            else: Tb = 2 * T if (kind in ("8d", "8g") or (kind == "t5vg" and site == "gu")) else T
            _CL[k] = chunks(out, T, Tb, 2 if mode == "gd2" else 3)
        return _CL[k]
    def run_site(fn, R, arg, y, out, T, extra, mode, rec, site, kind):
        pool = par.pool(W); t_sub = pc()
        def task(w, r0, r1, dyn, cnt, CL):
            t0 = pc(); c0 = tt(); n = 0
            if dyn == 1:
                while True:
                    o0 = next(cnt) * T
                    if o0 >= out: break
                    fn(R, arg, y, o0, min(out, o0 + T), T, w, extra); n += 1
            elif dyn == 2:
                nc = len(CL)
                while True:
                    j = next(cnt)
                    if j >= nc: break
                    a, b, sz = CL[j]; fn(R, arg, y, a, b, sz, w, extra); n += 1
            elif r0 < r1:
                fn(R, arg, y, r0, r1, T, w, extra); n = -(-(r1 - r0) // T)
            rec.append((site, w, t0 - t_sub, pc() - t_sub, tt() - c0, n))
        if mode == "st":
            nt = -(-out // T); per = -(-nt // W)
            futs = [pool.submit(task, w, min(out, w * per * T), min(out, (w + 1) * per * T), 0, None, None) for w in range(W)]
        elif mode == "dy":
            cnt = itertools.count(); futs = [pool.submit(task, w, 0, 0, 1, cnt, None) for w in range(W)]
        else:
            CL = chunk_list(site, kind, out, T, mode); cnt = itertools.count(); futs = [pool.submit(task, w, 0, 0, 2, cnt, CL) for w in range(W)]
        for f in futs: f.result()
    def mm(k, a, kind, T, mode, rec):
        B = a.shape[0]; pool = par.pool(W)
        if kind in ("tmL", "t5vg"): P = P5[k]; R = P["PR"]; i = P["inp"]
        else: P = par.F8[k]; R = P["Q"]; i = R.shape[1]
        out = R.shape[0]
        if kind == "tmL":
            a = np.ascontiguousarray(a, np.float32); y = np.empty((B, out), np.float32); q = R.shape[1]
            xp = XPL[:B, :q * 5]; xp[:, :i] = a; xp[:, i:] = 0; tab = TABL[:B, :q]; gper = -(-q // W); t_sub = pc()
            def bt(w, g0, g1):
                t0 = pc(); c0 = tt(); bldL(xp, tab, B, g0, g1, q); rec.append((k + ":bld", w, t0 - t_sub, pc() - t_sub, tt() - c0, 1))
            bf = [pool.submit(bt, w, w * gper, min(q, (w + 1) * gper)) for w in range(W) if w * gper < q]
            for f in bf: f.result()
            run_site(rows_tmL, R, TABL[:B, :q].reshape(B, q * 243), y, out, T, B, mode, rec, k, kind)
        elif kind == "8d":
            a = np.ascontiguousarray(a, np.float32); y = np.empty((B, out), np.float32)
            run_site(rows8d, R, a, y, out, T, i, mode, rec, k, kind)
        else:
            fn = rows_t5vg if kind == "t5vg" else rows8g
            xT = np.ascontiguousarray(np.asarray(a, np.float32).T); yT = np.empty((out, B), np.float32)
            run_site(fn, R, xT, yT, out, T, i, mode, rec, k, kind); y = np.ascontiguousarray(yT.T)
        y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
    def layer(x, kind, T, mode, rec):
        f = lambda k, a: mm(k, a, kind, T, mode, rec)
        h1 = rms(x); q, kk, v = f("qkv", h1)
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        o = f("o", att)[0]; x2 = x + o; h2 = rms(x2)
        g, u = f("gu", h2); z = (silu(g) * u).astype(np.float32)
        return x2 + f("down", z)[0]
    PATHS = {1: (("t5", "tmL", 384), ("i8", "8d", 128)), 4: (("t5", "t5vg", 256), ("i8", "8g", 128))}
    MODES = ("st_T", "dy_T", "gd2", "gd3", "gdL")
    MK = {"st_T": "st", "dy_T": "dy", "gd2": "gd2", "gd3": "gd3", "gdL": "gdL"}
    def cfgs(B): return {f"{tag}_{mn}": (kind, T, MK[mn]) for tag, kind, T in PATHS[B] for mn in MODES}
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    def main():
        seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W)
        W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        res["bytes"] = dict(t5=W5, int8=W8, int8_over_t5=W8 / W5)
        res["chunks"] = {}
        for B in (1, 4):
            for tag, kind, T in PATHS[B]:
                for mn in ("gd2", "gd3", "gdL"):
                    for k in SITES:
                        out = P5[k]["PR"].shape[0]; CL = chunk_list(k, kind, out, T, mn); sz = {}
                        for _, _, s in CL: sz[s] = sz.get(s, 0) + 1
                        res["chunks"][f"B{B}_{tag}_{mn}_{k}"] = dict(n=len(CL), sizes=sz, out=out)
        print("bytes", res["bytes"], flush=True); print("chunks", res["chunks"], flush=True)
        res["err"] = {}; res["time"] = {}; res["prof"] = {}
        for B in (1, 4):
            C = cfgs(B); xr = XREF[:B]; e = {}; outs = {}
            for c, (kind, T, mode) in C.items():
                y = layer(xr, kind, T, mode, []); outs[c] = y; r = REF[2 if c.startswith("t5") else 8][:B]
                e[c] = dict(mab=float(np.abs(y - r).max()), cos=cos(y, r))
            for c in C:
                b = c.split("_")[0] + "_st_T"; e[c]["vs_st_T_mab"] = float(np.abs(outs[c] - outs[b]).max()); e[c]["identical_st_T"] = bool(np.array_equal(outs[c], outs[b]))
            res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.12f}", v["identical_st_T"], f"{v['vs_st_T_mab']:.1e}") for c, v in e.items()}, flush=True)
            xb = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32); names = list(C); rm = {c: [] for c in names}
            for c in names: layer(xb, *C[c], [])
            for rd in range(ROUNDS):
                for c in names[rd % len(names):] + names[:rd % len(names)]:
                    s = []
                    for _ in range(REPS):
                        rec = []; t0 = pc(); layer(xb, *C[c], rec); s.append((pc() - t0) * 1e3)
                    rm[c].append(float(np.median(s)))
            o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), rm=v) for c, v in rm.items()}
            rat = {}
            def pr_(a, b): v = [p / q for p, q in zip(rm[a], rm[b])]; rat[f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v))
            for tag in ("t5", "i8"):
                for mn in MODES[1:]: pr_(f"{tag}_{mn}", f"{tag}_st_T")
                for mn in MODES[2:]: pr_(f"{tag}_{mn}", f"{tag}_dy_T")
            for mn in MODES: pr_(f"t5_{mn}", f"i8_{mn}")
            bt = min((c for c in names if c.startswith("t5")), key=lambda c: o[c]["med"]); bi = min((c for c in names if c.startswith("i8")), key=lambda c: o[c]["med"])
            v = [p / q for p, q in zip(rm[bt], rm[bi])]; rat["best_t5/best_i8"] = dict(r=o[bt]["med"] / o[bi]["med"], lo=min(v), hi=max(v), a=bt, b=bi)
            res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
            for c in names: print(f"B{B} {c:9s} {C[c][0]:5s}_T{C[c][1]:<4d} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
            for k, v in rat.items(): print(f"B{B} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f})", flush=True)
            pr = {}
            for c in names:
                acc = {}
                for _ in range(PROF):
                    rec = []; layer(xb, *C[c], rec); bys = {}
                    for site, w, t0, t1, cpu, n in rec: bys.setdefault(site, []).append((w, t0, t1, cpu, n))
                    for site, L in bys.items():
                        busy = np.zeros(W); cpuv = np.zeros(W); nt = np.zeros(W)
                        for w, t0, t1, cpu, n in L: busy[w] += t1 - t0; cpuv[w] += cpu; nt[w] += n
                        span = max(t1 for _, _, t1, _, _ in L); a = acc.setdefault(site, dict(span=[], mx=[], mean=[], cpu_mx=[], cpu_mean=[], off=[], act=[], tiles=[]))
                        a["span"].append(span * 1e6); a["mx"].append(busy.max() * 1e6); a["mean"].append(busy.mean() * 1e6)
                        a["cpu_mx"].append(cpuv[busy.argmax()] * 1e6); a["cpu_mean"].append(cpuv.mean() * 1e6); a["off"].append(1 - cpuv.sum() / max(busy.sum(), 1e-12))
                        a["act"].append(int((nt > 0).sum())); a["tiles"].append(sorted(nt.astype(int).tolist()))
                pr[c] = {s: dict(span=float(np.median(a["span"])), mx=float(np.median(a["mx"])), mean=float(np.median(a["mean"])), imb=float(np.median(np.array(a["mx"]) / np.array(a["mean"]))),
                                 cpu_mx=float(np.median(a["cpu_mx"])), cpu_mean=float(np.median(a["cpu_mean"])), off=float(np.median(a["off"])), act=int(np.median(a["act"])), tiles=a["tiles"][-1]) for s, a in acc.items()}
                for s, d in pr[c].items(): print(f"B{B} prof {c:9s} {s:9s} span {d['span']:8.1f} us  slowest {d['mx']:8.1f}  mean {d['mean']:8.1f}  imb {d['imb']:.2f}  cpu slowest/mean {d['cpu_mx']:.0f}/{d['cpu_mean']:.0f}  offcpu {d['off']:.2f}  active {d['act']}  chunks {d['tiles']}", flush=True)
            res["prof"][f"B{B}"] = pr
        res["scratch_live"] = dict(t5=sum(v.nbytes for v in _s.values()), int8=par.live_scratch_bytes())
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    main()

Aggregator (/workspace/bench1855_1125/agg.py):

    import json, numpy as np
    R = [json.load(open(f"{n}.json")) for n in ("run1", "run2", "rep1", "rep2")]
    for B in ("B1", "B4"):
        print(f"## {B}")
        for c in R[0]["time"][B]["cfg"]:
            m = [r["time"][B]["cfg"][c]["med"] for r in R]; lo = min(r["time"][B]["cfg"][c]["lo"] for r in R); hi = max(r["time"][B]["cfg"][c]["hi"] for r in R)
            print(f"{c:10s} meds " + " / ".join(f"{x:.3f}" for x in m) + f"  median {np.median(m):.3f}  rounds [{lo:.3f}-{hi:.3f}]")
        for k in R[0]["time"][B]["ratios"]:
            m = [r["time"][B]["ratios"][k]["r"] for r in R]
            lo = min(r["time"][B]["ratios"][k]["lo"] for r in R); hi = max(r["time"][B]["ratios"][k]["hi"] for r in R)
            extra = [r["time"][B]["ratios"][k].get("a", "") + "/" + r["time"][B]["ratios"][k].get("b", "") for r in R] if k.startswith("best") else ""
            print(f"{k:22s} " + " / ".join(f"{x:.3f}" for x in m) + f"  median {np.median(m):.3f}  rounds [{lo:.3f}-{hi:.3f}] {extra}")
        for c in R[0]["prof"][B]:
            for s in ("qkv", "o", "gu", "down"):
                g = lambda f: np.median([r["prof"][B][c][s][f] for r in R])
                print(f"prof {c:9s} {s:5s} span {g('span'):7.0f} slowest {g('mx'):7.0f} mean {g('mean'):7.0f} imb {g('imb'):.2f} cpu_slowest {g('cpu_mx'):6.0f} offcpu {g('off'):.2f} active {[r['prof'][B][c][s]['act'] for r in R]} chunks {R[0]['prof'][B][c][s]['tiles']}")
    for B in ("B1", "B4"):
        e = R[0]["err"][B]; print(B, {c: (f"{v['mab']:.2e}", round(v['cos'], 14), v["identical_st_T"], f"{v['vs_st_T_mab']:.1e}") for c, v in e.items()})
        print(B, "identical per cfg over runs:", {c: all(r["err"][B][c]["identical_st_T"] for r in R) for c in e})
    print("scratch", [r["scratch_live"] for r in R])

Table generator (/workspace/bench1855_1125/tab.py):

    import json, numpy as np
    R = [json.load(open(f"{n}.json")) for n in ("run1", "run2", "rep1", "rep2")]
    KT = {"B1": {"t5": "tmL", "i8": "8d"}, "B4": {"t5": "t5vg", "i8": "8g"}}
    TT = {"B1": {"t5": 384, "i8": 128}, "B4": {"t5": 256, "i8": 128}}
    DESC = {"st_T": "static ceil split, T", "dy_T": "dynamic count, T", "gd2": "guided Tb -> T/2", "gd3": "guided Tb -> T/2 -> T/4", "gdL": "gd3, Tb=2T on gu only"}
    print("| B | config | kernel / schedule | median ms | run medians | vs own st_T (round range) | vs own dy_T (round range) |")
    print("|---|---|---|---|---|---|---|")
    for B in ("B1", "B4"):
        for c in R[0]["time"][B]["cfg"]:
            tag, mn = c.split("_", 1); m = [r["time"][B]["cfg"][c]["med"] for r in R]
            def rr(k):
                if k not in R[0]["time"][B]["ratios"]: return "1" if k.endswith(c) else "—"
                v = [r["time"][B]["ratios"][k]["r"] for r in R]; lo = min(r["time"][B]["ratios"][k]["lo"] for r in R); hi = max(r["time"][B]["ratios"][k]["hi"] for r in R)
                return f"{np.median(v):.3f}x ({lo:.3f}–{hi:.3f})"
            a = rr(f"{c}/{tag}_st_T") if mn != "st_T" else "1"
            b = rr(f"{c}/{tag}_dy_T") if mn not in ("st_T", "dy_T") else ("1" if mn == "dy_T" else "—")
            print(f"| {B[1]} | {c} | {KT[B][tag]} T{TT[B][tag]}, {DESC[mn]} | {np.median(m):.3f} | " + " / ".join(f"{x:.3f}" for x in m) + f" | {a} | {b} |")
    print()
    print("| B | pair | run ratios | median | rounds |"); print("|---|---|---|---|---|")
    for B in ("B1", "B4"):
        for k in [k for k in R[0]["time"][B]["ratios"] if k.startswith("t5_") and "/i8_" in k] + ["best_t5/best_i8"]:
            v = [r["time"][B]["ratios"][k]["r"] for r in R]; lo = min(r["time"][B]["ratios"][k]["lo"] for r in R); hi = max(r["time"][B]["ratios"][k]["hi"] for r in R)
            lab = k if not k.startswith("best") else "best t5 / best int8 (" + ", ".join(r["time"][B]["ratios"][k]["a"] + " vs " + r["time"][B]["ratios"][k]["b"] for r in R) + ")"
            print(f"| {B[1]} | {lab} | " + " / ".join(f"{x:.3f}" for x in v) + f" | {np.median(v):.3f}x | {lo:.3f}–{hi:.3f} |")
    print()
    print("| B | config | site | span | slowest | mean | imbalance | cpu of slowest | off-CPU share | chunks per worker (run 1) |"); print("|---|---|---|---|---|---|---|---|---|---|")
    for B in ("B1", "B4"):
        for c in R[0]["prof"][B]:
            for s in ("qkv", "o", "gu", "down"):
                g = lambda f: np.median([r["prof"][B][c][s][f] for r in R])
                print(f"| {B[1]} | {c} | {s} | {g('span'):.0f} | {g('mx'):.0f} | {g('mean'):.0f} | {g('imb'):.2f} | {g('cpu_mx'):.0f} | {g('off'):.2f} | {','.join(map(str, R[0]['prof'][B][c][s]['tiles']))} |")
    print()
    print("| B | config | claims per layer | sum of site spans µs | layer off-CPU share (1 − Σcpu/Σbusy) |"); print("|---|---|---|---|---|")
    for B in ("B1", "B4"):
        for c in R[0]["prof"][B]:
            S = ("qkv", "o", "gu", "down"); offs = [1 - sum(r["prof"][B][c][s]["cpu_mean"] for s in S) / sum(r["prof"][B][c][s]["mean"] for s in S) for r in R]
            sp = np.median([sum(r["prof"][B][c][s]["span"] for s in S) for r in R]); n = sum(sum(R[0]["prof"][B][c][s]["tiles"]) for s in S)
            print(f"| {B[1]} | {c} | {n} | {sp:.0f} | {np.median(offs):.2f} |")

## Results

Layer time, ms (median of 4 run medians; per-run medians run1 / run2 / rep1 / rep2; ratios paired per round, median of 4 runs, range of round ratios):

| B | config | kernel / schedule | median ms | run medians | vs own st_T (round range) | vs own dy_T (round range) |
|---|---|---|---|---|---|---|
| 1 | t5_st_T | tmL T384, static ceil split, T | 8.752 | 8.461 / 9.044 / 9.180 / 8.258 | 1 | — |
| 1 | t5_dy_T | tmL T384, dynamic count, T | 8.178 | 8.318 / 8.486 / 8.038 / 7.801 | 0.942x (0.783–1.258) | 1 |
| 1 | t5_gd2 | tmL T384, guided Tb -> T/2 | 8.326 | 8.279 / 9.089 / 8.372 / 7.985 | 0.973x (0.802–1.461) | 1.033x (0.914–1.161) |
| 1 | t5_gd3 | tmL T384, guided Tb -> T/2 -> T/4 | 9.001 | 8.771 / 9.767 / 9.232 / 8.596 | 1.039x (0.853–1.216) | 1.125x (0.812–1.297) |
| 1 | t5_gdL | tmL T384, gd3, Tb=2T on gu only | 8.864 | 8.752 / 10.029 / 8.976 / 8.537 | 1.034x (0.810–1.695) | 1.106x (0.747–1.880) |
| 1 | i8_st_T | 8d T128, static ceil split, T | 5.521 | 5.246 / 5.795 / 6.023 / 5.151 | 1 | — |
| 1 | i8_dy_T | 8d T128, dynamic count, T | 5.428 | 5.418 / 6.193 / 5.438 / 5.344 | 1.035x (0.793–1.115) | 1 |
| 1 | i8_gd2 | 8d T128, guided Tb -> T/2 | 5.134 | 5.084 / 5.797 / 5.149 / 5.118 | 0.981x (0.833–1.466) | 0.943x (0.785–1.850) |
| 1 | i8_gd3 | 8d T128, guided Tb -> T/2 -> T/4 | 5.319 | 5.177 / 5.792 / 5.385 / 5.253 | 0.993x (0.797–1.241) | 0.969x (0.814–1.220) |
| 1 | i8_gdL | 8d T128, gd3, Tb=2T on gu only | 5.536 | 5.454 / 6.132 / 5.496 / 5.576 | 1.049x (0.801–1.450) | 1.009x (0.902–1.425) |
| 4 | t5_st_T | t5vg T256, static ceil split, T | 16.910 | 17.339 / 17.247 / 15.846 / 16.573 | 1 | — |
| 4 | t5_dy_T | t5vg T256, dynamic count, T | 15.578 | 16.428 / 16.211 / 14.618 / 14.944 | 0.931x (0.861–1.016) | 1 |
| 4 | t5_gd2 | t5vg T256, guided Tb -> T/2 | 14.527 | 15.263 / 15.517 / 13.592 / 13.791 | 0.869x (0.794–1.153) | 0.929x (0.872–1.212) |
| 4 | t5_gd3 | t5vg T256, guided Tb -> T/2 -> T/4 | 16.825 | 18.053 / 17.583 / 15.313 / 16.067 | 0.994x (0.909–1.102) | 1.080x (0.931–1.141) |
| 4 | t5_gdL | t5vg T256, gd3, Tb=2T on gu only | 16.661 | 18.325 / 17.459 / 15.260 / 15.863 | 0.988x (0.903–1.104) | 1.069x (0.913–1.172) |
| 4 | i8_st_T | 8g T128, static ceil split, T | 7.755 | 8.578 / 8.316 / 6.643 / 7.195 | 1 | — |
| 4 | i8_dy_T | 8g T128, dynamic count, T | 6.577 | 7.117 / 7.049 / 6.105 / 6.043 | 0.844x (0.746–1.629) | 1 |
| 4 | i8_gd2 | 8g T128, guided Tb -> T/2 | 8.313 | 8.771 / 8.765 / 7.800 / 7.860 | 1.073x (0.937–1.549) | 1.260x (0.951–1.369) |
| 4 | i8_gd3 | 8g T128, guided Tb -> T/2 -> T/4 | 7.922 | 8.327 / 8.324 / 7.432 / 7.521 | 1.023x (0.900–1.305) | 1.199x (0.793–1.298) |
| 4 | i8_gdL | 8g T128, gd3, Tb=2T on gu only | 8.801 | 9.681 / 9.187 / 8.037 / 8.415 | 1.149x (0.969–1.393) | 1.338x (0.843–1.428) |

t5 / int8 gap (paired per round; median of 4 runs, round range):

| B | pair | run ratios | median | rounds |
|---|---|---|---|---|
| 1 | t5_st_T/i8_st_T | 1.613 / 1.561 / 1.524 / 1.603 | 1.582x | 1.290–1.972 |
| 1 | t5_dy_T/i8_dy_T | 1.535 / 1.370 / 1.478 / 1.460 | 1.469x | 1.266–2.134 |
| 1 | t5_gd2/i8_gd2 | 1.628 / 1.568 / 1.626 / 1.560 | 1.597x | 0.801–2.657 |
| 1 | t5_gd3/i8_gd3 | 1.694 / 1.687 / 1.714 / 1.636 | 1.690x | 1.381–1.981 |
| 1 | t5_gdL/i8_gdL | 1.605 / 1.635 / 1.633 / 1.531 | 1.619x | 1.154–2.732 |
| 1 | best t5 / best int8 (t5_gd2 vs i8_gd2, t5_dy_T vs i8_gd3, t5_dy_T vs i8_gd2, t5_dy_T vs i8_gd2) | 1.628 / 1.465 / 1.561 / 1.524 | 1.543x | 0.794–2.288 |
| 4 | t5_st_T/i8_st_T | 2.021 / 2.074 / 2.385 / 2.303 | 2.189x | 1.928–2.659 |
| 4 | t5_dy_T/i8_dy_T | 2.308 / 2.300 / 2.394 / 2.473 | 2.351x | 1.255–2.595 |
| 4 | t5_gd2/i8_gd2 | 1.740 / 1.770 / 1.743 / 1.755 | 1.749x | 1.219–2.212 |
| 4 | t5_gd3/i8_gd3 | 2.168 / 2.112 / 2.060 / 2.136 | 2.124x | 1.717–2.244 |
| 4 | t5_gdL/i8_gdL | 1.893 / 1.900 / 1.899 / 1.885 | 1.896x | 1.599–2.128 |
| 4 | best t5 / best int8 (t5_gd2 vs i8_dy_T, t5_gd2 vs i8_dy_T, t5_gd2 vs i8_dy_T, t5_gd2 vs i8_dy_T) | 2.145 / 2.201 / 2.226 / 2.282 | 2.214x | 1.159–2.817 |

Per-site per-worker profile, µs (median of 4 runs; each run = median of 25 profiled layers). Slowest = max worker wall busy, mean = over all 8 workers, cpu of slowest = that worker's thread_time, off-CPU = 1 − Σcpu / Σbusy for the site; chunks = per-worker claims (st_T: T-tiles per worker), run 1, sorted:

| B | config | site | span | slowest | mean | imbalance | cpu of slowest | off-CPU share | chunks per worker (run 1) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | t5_st_T | qkv | 1098 | 982 | 576 | 1.71 | 880 | 0.08 | 0,0,0,2,2,2,2,2 |
| 1 | t5_st_T | o | 750 | 616 | 449 | 1.38 | 464 | 0.16 | 0,1,1,1,1,1,1,1 |
| 1 | t5_st_T | gu | 2691 | 2577 | 2081 | 1.21 | 2102 | 0.11 | 1,5,5,5,5,5,5,5 |
| 1 | t5_st_T | down | 1757 | 1632 | 1258 | 1.28 | 1512 | 0.03 | 0,1,1,1,1,1,1,1 |
| 1 | t5_dy_T | qkv | 1076 | 1018 | 689 | 1.46 | 875 | 0.19 | 1,1,1,1,1,1,2,2 |
| 1 | t5_dy_T | o | 765 | 659 | 471 | 1.40 | 463 | 0.20 | 0,1,1,1,1,1,1,1 |
| 1 | t5_dy_T | gu | 2750 | 2652 | 2396 | 1.11 | 2165 | 0.19 | 3,3,4,4,5,5,6,6 |
| 1 | t5_dy_T | down | 1876 | 1726 | 1286 | 1.33 | 1545 | 0.04 | 0,1,1,1,1,1,1,1 |
| 1 | t5_gd2 | qkv | 1196 | 1098 | 924 | 1.18 | 708 | 0.35 | 1,2,2,2,2,2,3,3 |
| 1 | t5_gd2 | o | 906 | 837 | 676 | 1.24 | 502 | 0.38 | 1,1,2,2,2,2,2,2 |
| 1 | t5_gd2 | gu | 2886 | 2780 | 2611 | 1.07 | 2144 | 0.24 | 5,5,5,5,5,6,6,6 |
| 1 | t5_gd2 | down | 1779 | 1673 | 1374 | 1.22 | 1491 | 0.11 | 1,1,2,2,2,2,2,2 |
| 1 | t5_gd3 | qkv | 1451 | 1355 | 1205 | 1.13 | 755 | 0.46 | 2,3,3,3,3,3,3,4 |
| 1 | t5_gd3 | o | 1190 | 1086 | 955 | 1.15 | 542 | 0.51 | 1,2,3,3,3,3,3,3 |
| 1 | t5_gd3 | gu | 3161 | 3062 | 2907 | 1.05 | 2160 | 0.30 | 4,5,6,6,7,7,7,8 |
| 1 | t5_gd3 | down | 1888 | 1785 | 1557 | 1.15 | 1500 | 0.18 | 2,2,2,3,3,3,3,3 |
| 1 | t5_gdL | qkv | 1406 | 1317 | 1170 | 1.13 | 737 | 0.45 | 2,2,3,3,3,3,4,4 |
| 1 | t5_gdL | o | 1164 | 1071 | 931 | 1.15 | 539 | 0.51 | 2,2,2,3,3,3,3,3 |
| 1 | t5_gdL | gu | 2866 | 2785 | 2613 | 1.06 | 2102 | 0.25 | 3,4,4,4,4,5,5,5 |
| 1 | t5_gdL | down | 1883 | 1786 | 1551 | 1.15 | 1485 | 0.18 | 2,2,2,3,3,3,3,3 |
| 1 | i8_st_T | qkv | 932 | 812 | 627 | 1.26 | 400 | 0.47 | 2,4,4,4,4,4,4,4 |
| 1 | i8_st_T | o | 645 | 523 | 364 | 1.46 | 313 | 0.35 | 0,2,3,3,3,3,3,3 |
| 1 | i8_st_T | gu | 2254 | 2139 | 1873 | 1.14 | 1239 | 0.44 | 10,14,14,14,14,14,14,14 |
| 1 | i8_st_T | down | 1166 | 983 | 760 | 1.30 | 922 | 0.04 | 0,2,3,3,3,3,3,3 |
| 1 | i8_dy_T | qkv | 984 | 889 | 749 | 1.19 | 395 | 0.54 | 1,1,2,2,5,6,6,7 |
| 1 | i8_dy_T | o | 690 | 608 | 482 | 1.25 | 291 | 0.50 | 2,2,2,2,2,3,3,4 |
| 1 | i8_dy_T | gu | 2570 | 2489 | 2365 | 1.06 | 1205 | 0.52 | 11,11,13,13,13,13,15,19 |
| 1 | i8_dy_T | down | 1049 | 944 | 772 | 1.23 | 864 | 0.08 | 1,2,2,3,3,3,3,3 |
| 1 | i8_gd2 | qkv | 941 | 847 | 704 | 1.19 | 501 | 0.40 | 2,2,3,3,3,3,4,4 |
| 1 | i8_gd2 | o | 702 | 626 | 493 | 1.25 | 334 | 0.44 | 1,2,2,2,3,3,3,3 |
| 1 | i8_gd2 | gu | 1996 | 1914 | 1788 | 1.07 | 1380 | 0.23 | 5,6,8,8,8,8,10,10 |
| 1 | i8_gd2 | down | 1188 | 1073 | 889 | 1.18 | 855 | 0.15 | 1,2,2,2,3,3,3,3 |
| 1 | i8_gd3 | qkv | 1022 | 924 | 803 | 1.17 | 499 | 0.47 | 2,3,4,4,4,4,5,5 |
| 1 | i8_gd3 | o | 837 | 757 | 628 | 1.18 | 346 | 0.53 | 2,2,2,3,4,4,4,5 |
| 1 | i8_gd3 | gu | 2102 | 2033 | 1903 | 1.06 | 1465 | 0.28 | 6,8,9,9,9,9,9,11 |
| 1 | i8_gd3 | down | 1245 | 1157 | 992 | 1.15 | 800 | 0.26 | 2,2,2,3,4,4,4,5 |
| 1 | i8_gdL | qkv | 1198 | 1105 | 989 | 1.13 | 430 | 0.61 | 3,4,5,5,6,6,7,8 |
| 1 | i8_gdL | o | 923 | 840 | 731 | 1.14 | 308 | 0.61 | 3,4,4,4,4,5,5,5 |
| 1 | i8_gdL | gu | 2090 | 2024 | 1877 | 1.07 | 1454 | 0.27 | 7,7,8,9,9,10,10,10 |
| 1 | i8_gdL | down | 1271 | 1180 | 1039 | 1.14 | 853 | 0.28 | 3,3,4,4,4,5,5,6 |
| 4 | t5_st_T | qkv | 2507 | 2333 | 1797 | 1.28 | 1519 | 0.21 | 1,2,2,2,2,2,2,2 |
| 4 | t5_st_T | o | 2022 | 1881 | 1023 | 1.81 | 1728 | 0.09 | 0,0,0,2,2,2,2,2 |
| 4 | t5_st_T | gu | 7534 | 7295 | 5959 | 1.18 | 5293 | 0.16 | 5,7,7,7,7,7,7,7 |
| 4 | t5_st_T | down | 4339 | 4131 | 2403 | 1.72 | 4055 | 0.02 | 0,0,0,2,2,2,2,2 |
| 4 | t5_dy_T | qkv | 2312 | 2159 | 1773 | 1.19 | 1483 | 0.23 | 1,2,2,2,2,2,2,2 |
| 4 | t5_dy_T | o | 1855 | 1720 | 1240 | 1.42 | 1440 | 0.25 | 1,1,1,1,1,1,2,2 |
| 4 | t5_dy_T | gu | 6691 | 6512 | 6173 | 1.06 | 5118 | 0.19 | 6,6,6,7,7,7,7,8 |
| 4 | t5_dy_T | down | 4010 | 3894 | 2585 | 1.48 | 3682 | 0.10 | 1,1,1,1,1,1,2,2 |
| 4 | t5_gd2 | qkv | 2381 | 2248 | 1931 | 1.14 | 1544 | 0.25 | 2,2,3,3,3,3,3,3 |
| 4 | t5_gd2 | o | 1727 | 1594 | 1333 | 1.20 | 1108 | 0.29 | 2,2,2,2,2,2,2,3 |
| 4 | t5_gd2 | gu | 6133 | 5960 | 5678 | 1.05 | 4990 | 0.14 | 3,3,4,4,5,5,6,6 |
| 4 | t5_gd2 | down | 3663 | 3496 | 2937 | 1.19 | 3045 | 0.13 | 2,2,2,2,2,2,2,3 |
| 4 | t5_gd3 | qkv | 2769 | 2650 | 2393 | 1.11 | 1562 | 0.40 | 2,3,3,4,4,4,4,5 |
| 4 | t5_gd3 | o | 2050 | 1929 | 1676 | 1.14 | 1057 | 0.43 | 1,3,3,3,3,3,4,4 |
| 4 | t5_gd3 | gu | 6335 | 6197 | 5909 | 1.05 | 4946 | 0.19 | 4,5,5,5,6,6,6,6 |
| 4 | t5_gd3 | down | 4937 | 4772 | 4067 | 1.16 | 2877 | 0.35 | 1,2,3,3,3,4,4,4 |
| 4 | t5_gdL | qkv | 2777 | 2648 | 2374 | 1.11 | 1536 | 0.39 | 3,3,3,3,4,4,4,5 |
| 4 | t5_gdL | o | 2035 | 1933 | 1685 | 1.14 | 1098 | 0.43 | 2,2,3,3,3,3,4,4 |
| 4 | t5_gdL | gu | 6415 | 6279 | 5980 | 1.05 | 4931 | 0.19 | 5,5,5,5,5,6,6,6 |
| 4 | t5_gdL | down | 4913 | 4749 | 4041 | 1.17 | 2920 | 0.35 | 2,2,3,3,3,3,4,4 |
| 4 | i8_st_T | qkv | 1100 | 923 | 715 | 1.29 | 603 | 0.21 | 2,4,4,4,4,4,4,4 |
| 4 | i8_st_T | o | 784 | 638 | 447 | 1.41 | 495 | 0.13 | 0,2,3,3,3,3,3,3 |
| 4 | i8_st_T | gu | 2712 | 2578 | 2107 | 1.21 | 1875 | 0.13 | 10,14,14,14,14,14,14,14 |
| 4 | i8_st_T | down | 1611 | 1429 | 1058 | 1.31 | 1300 | 0.03 | 0,2,3,3,3,3,3,3 |
| 4 | i8_dy_T | qkv | 1036 | 927 | 771 | 1.19 | 646 | 0.29 | 2,3,4,4,4,4,4,5 |
| 4 | i8_dy_T | o | 748 | 649 | 513 | 1.24 | 461 | 0.25 | 2,2,2,2,3,3,3,3 |
| 4 | i8_dy_T | gu | 2592 | 2511 | 2373 | 1.06 | 1928 | 0.22 | 12,13,13,13,13,14,14,16 |
| 4 | i8_dy_T | down | 1413 | 1308 | 1057 | 1.23 | 1221 | 0.07 | 1,2,2,3,3,3,3,3 |
| 4 | i8_gd2 | qkv | 1246 | 1116 | 930 | 1.22 | 654 | 0.42 | 1,2,2,2,3,4,4,6 |
| 4 | i8_gd2 | o | 949 | 845 | 653 | 1.29 | 404 | 0.46 | 1,1,1,2,3,3,4,4 |
| 4 | i8_gd2 | gu | 2594 | 2484 | 2287 | 1.09 | 1932 | 0.20 | 5,7,8,8,8,9,9,9 |
| 4 | i8_gd2 | down | 2577 | 2454 | 1899 | 1.28 | 1119 | 0.45 | 1,1,2,2,2,3,3,5 |
| 4 | i8_gd3 | qkv | 1283 | 1189 | 1010 | 1.17 | 627 | 0.47 | 2,2,3,3,4,5,6,6 |
| 4 | i8_gd3 | o | 1018 | 925 | 771 | 1.19 | 407 | 0.53 | 1,2,2,2,3,3,5,8 |
| 4 | i8_gd3 | gu | 2641 | 2553 | 2377 | 1.06 | 1890 | 0.23 | 5,8,8,9,9,9,10,12 |
| 4 | i8_gd3 | down | 2228 | 2112 | 1857 | 1.14 | 1050 | 0.49 | 2,3,3,3,3,3,4,5 |
| 4 | i8_gdL | qkv | 1515 | 1379 | 1241 | 1.13 | 616 | 0.55 | 3,4,5,5,6,7,7,7 |
| 4 | i8_gdL | o | 1143 | 1050 | 907 | 1.15 | 421 | 0.59 | 3,3,4,4,4,5,5,6 |
| 4 | i8_gdL | gu | 2661 | 2547 | 2402 | 1.07 | 1914 | 0.24 | 6,7,9,9,9,9,10,11 |
| 4 | i8_gdL | down | 2675 | 2559 | 2303 | 1.11 | 1073 | 0.58 | 3,3,3,5,5,5,5,5 |

Whole-layer claims and off-CPU share (sum over the 4 sites; median of 4 runs):

| B | config | kernel tiles per layer (= claims for dy / gd; st_T runs 8 tasks per site) | sum of site spans µs | layer off-CPU share (1 − Σcpu/Σbusy) |
|---|---|---|---|---|
| 1 | t5_st_T | 60 | 6298 | 0.10 |
| 1 | t5_dy_T | 60 | 6473 | 0.15 |
| 1 | t5_gd2 | 88 | 6785 | 0.24 |
| 1 | t5_gd3 | 116 | 7691 | 0.33 |
| 1 | t5_gdL | 100 | 7320 | 0.31 |
| 1 | i8_st_T | 178 | 4997 | 0.34 |
| 1 | i8_dy_T | 178 | 5292 | 0.45 |
| 1 | i8_gd2 | 125 | 4864 | 0.27 |
| 1 | i8_gd3 | 153 | 5206 | 0.35 |
| 1 | i8_gdL | 182 | 5483 | 0.40 |
| 4 | t5_st_T | 89 | 16326 | 0.14 |
| 4 | t5_dy_T | 89 | 14869 | 0.19 |
| 4 | t5_gd2 | 92 | 13903 | 0.18 |
| 4 | t5_gd3 | 120 | 16091 | 0.30 |
| 4 | t5_gdL | 120 | 16139 | 0.30 |
| 4 | i8_st_T | 178 | 6204 | 0.14 |
| 4 | i8_dy_T | 178 | 5789 | 0.20 |
| 4 | i8_gd2 | 125 | 7393 | 0.36 |
| 4 | i8_gd3 | 153 | 7149 | 0.39 |
| 4 | i8_gdL | 182 | 7993 | 0.46 |

Error vs float64 refs (same in all 4 runs per seed): B 1 t5 every mode max |err| 5.57e-07, cos 0.99999999999999; int8 every mode 3.19e-06, cos 0.99999999999986. B 4 t5 st / dy / gd2 1.23e-06 (cos 0.99999999999997), gd3 / gdL 1.06e-06; int8 st / dy 4.42e-06 (cos 0.99999999999983), gd2 / gd3 3.67e-06, gdL 3.52e-06. Bit-identical to st_T in all 4 runs: every B 1 mode; B 4 t5 dy_T and gd2, int8 dy_T. Not identical: B 4 t5 gd3 / gdL (max |Δ| 1.4e-06), int8 gd2 / gd3 (4.2e-06), gdL (4.4e-06).

Bytes: weights per layer 13,986,304 B t5 vs 69,559,296 B int8 (4.973x), unchanged. Scratch for 8 workers, one buffer per chunk size as run: tmL 50,982,912 (st / dy) / 76,474,368 (gd2) / 89,220,096 (gd3) / 191,185,920 B (gdL) + B 1 table 1,344,276 B; t5vg 70,809,600 / 247,833,600 / 265,536,000 / 265,536,000 B; int8 35,389,440 / 88,473,600 / 97,320,960 / 132,710,400 B. With one max-size buffer per worker: tmL 50,982,912 (gdL 101,965,824), t5vg 141,619,200, int8 70,778,880 B. A guided schedule adds one precomputed chunk list and one itertools.count per site.

## Pair files this wave

No pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) for this wave reached this lane, so none are linked; pairs idle. The depth-1 clone of JuniorLLM (18:49 MT) shows this lane's own 33bcc47 (14:56 MT, the 14:50 note) as HEAD on main. Nothing is newer, so no Trit Pack, Handshake, LLC or unattributed commit in the last 6 hours (since ~12:45 MT) touches junior_osai/vault/findings. This lane did not read JuniorHome this wave.

## Next question

Scheduling is now worth at most 1.08–1.15x for t5, and the gap to int8 (1.54x B 1, 2.21x B 4) sits inside the kernels, not between workers. Decompose the t5 kernel per op inside the best schedule. For tmL_T384 at B 1 (dy_T) and t5vg_T256 at B 4 (gd2), time each stage per chunk on the W 8 pool with perf_counter and thread_time around each numpy call: tmL np.add index build / np.take gather from the B tables / np.dot reduce, plus the static table-build round; t5vg V5 np.take LUT expand / int8 -> f32 np.copyto / np.matmul. Compare with int8d's and int8g's copyto cast / np.dot or matmul stages on the same rows. Which stage holds the t5 excess (in µs per site and as a share of the gap), and is that stage on-CPU (compute or memory) or off-CPU (GIL wait)? The 11-24 02:55 note split the 2-bit u32-LUT path at B 1 only. No earlier note split the base-3 t5 tmL or t5vg stages, or did it at B 4.

close: c9b17e4a8bea2e53
