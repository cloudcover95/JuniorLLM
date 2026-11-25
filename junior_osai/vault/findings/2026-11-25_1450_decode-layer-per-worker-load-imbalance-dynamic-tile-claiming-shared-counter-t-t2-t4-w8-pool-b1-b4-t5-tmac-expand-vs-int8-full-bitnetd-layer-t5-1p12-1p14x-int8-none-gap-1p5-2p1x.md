---
local: true
date: 2026-11-25
wave: "2026-11-25 14:55 slot (fired 14:49 MT, ran ~14:50–15:05 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (per-worker load imbalance on the W 8 decode layer, full bitnetd layer, ctx-free. Dynamic tile claiming from a shared itertools.count at the native tile T gives the same output bit for bit and speeds up t5 by 1.12x at B 1 (tmL_T384, dy_T / st_T median 0.896x of 4 runs) and 1.14x at B 4 (t5vg_T256, 0.878x), clearing the 1.1x bar, because the static ceil(tiles/W) split leaves 1–3 of 8 workers idle on t5's few large tiles (qkv B 1 5 of 8 active, o / down B 4 5 of 8). int8 gains nothing (0.995x B 1, 0.941x B 4). Smaller tiles T/2 and T/4 lose everywhere (t5 0.91–1.67x, int8 1.40–2.61x slower) from per-tile Python/GIL cost. The t5 / int8 gap narrows but stays far above the 1.3x bar: best/best B 1 1.65x -> 1.49x, B 4 2.20x -> 2.12x. No 1.5x speed or memory win for ternary; the 4.973x t5 weight-memory win vs int8 is already known)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, load-imbalance, dynamic-scheduling, work-stealing, tile-size, per-worker-busy, thread-time, base-3, 1p6-bit, five-trit-pack, tmac, lut-expand, int8, numpy, not-viable]
---

# Load imbalance on the W 8 decode layer: shared-counter tile claiming speeds t5 up 1.12–1.14x (int8 not at all), smaller tiles lose, and t5's gap to int8 stays 1.5–2.1x

The 10:41 note ([[2026-11-25_1041_decode-layer-dispatch-overhead-persistent-barrier-workers-vs-four-submit-rounds-w8-pool-b1-b4-t5-tmac-expand-vs-int8-full-bitnetd-layer-bit-identical-no-1p1x-gain-gap-unchanged]]) found that a persistent-worker Barrier layer does not beat four submit/result rounds, and asked: record per-worker wall start / end and time.thread_time per site for tmL_T384 / int8d_T128 at B 1 and t5vg_T256 / int8g_T128 at B 4 (full bitnetd layer, ctx-free), report each site's critical path (slowest worker) vs mean worker busy time, then compare the static ceil(tiles / W) split with dynamic tile claiming (a shared itertools.count that workers pull T-row tiles from until exhausted) at tile sizes T, T/2 and T/4. Does dynamic claiming cut the slowest-worker time per site by ≥ 1.1x, and does it narrow t5's 1.6–2.3x gap to int8?

This is not a repeat. No earlier note measured per-worker busy / idle / thread_time inside a site, or dynamic tile claiming. The 11-24 18:55 note tested a *static balanced* partition (W contiguous ranges of out/W rows, sub-tiled at 384) and found noise-level change (0.962x / 1.020x) for t5. Here the claiming is dynamic and whole-tile, and the static split is xb5's ceil split as-is.

Answers:
- **Where the imbalance is (static split, 4-run medians of the per-site medians over 25 profiled layers):** t5 has few large tiles per site, so ceil(tiles / W) leaves workers idle: tmL_T384 B 1 qkv 10 tiles -> 5 of 8 workers active (slowest 1,255 µs vs mean over 8 workers 683 µs, imbalance 1.78), o / down 7 tiles -> 7 active (1.45 / 1.33). t5vg_T256 B 4 o / down 10 tiles -> 5 active (imbalance 1.83 / 1.73), gu 54 tiles split [5,7,7,7,7,7,7,7] (1.19). int8_T128 has 2–4x more tiles and is milder: o / down 20 tiles -> 7 active split [0,2,3,...] (1.40–1.49), gu 108 tiles (1.14–1.16).
- **Wall vs CPU:** t5 workers are on-CPU for most of their wall time (tmL B 1 gu slowest worker 2,990 µs wall vs 2,501 µs thread_time; t5vg B 4 gu 6,985 vs 5,606). int8 workers spend ~40% of busy wall time off-CPU (int8d B 1 gu 2,561 vs 1,488; int8g B 4 gu 2,907 vs 2,080), i.e. waiting on the GIL or on memory between numpy calls. So int8 is closer to a shared ceiling, and rebalancing its tiles has nothing to recover.
- **Does dynamic claiming at T cut layer time by ≥ 1.1x (ratio ≤ 0.909)?** **For t5 yes, for int8 no.** dy_T / st_T, 4 run medians: t5 B 1 0.901 / 0.891 / 0.889 / 0.911, median **0.896x (1.12x faster)**. t5 B 4 0.856 / 0.833 / 0.979 / 0.900, median **0.878x (1.14x)**. int8 B 1 0.989 / 0.977 / 1.001 / 1.035, median 0.995x. int8 B 4 0.968 / 0.914 / 0.970 / 0.797, median 0.941x (one noisy run; 3 of 4 runs above 0.909).
- **Per-site slowest worker (dy_T / st_T, median of 4):** t5 B 4 o 0.858, down 0.870, gu 0.931, qkv 0.927. t5 B 1 qkv 0.960, o / gu / down 1.00–1.03: at B 1 the win comes from the site span (qkv span 1,393 -> 1,258 µs, gu 3,190 -> 2,977 µs) and less tail, not from a faster slowest worker. int8 slowest worker gets slower with claiming (qkv 1.06–1.11, gu 1.14–1.18) because all 8 workers now contend, and only down B 1 improves (0.844). For int8 the dynamic split evens out the tiles but each tile gets slower, so layer time does not move.
- **Smaller tiles (T/2, T/4) lose:** t5 dy_T2 0.996x (B 1) / 0.914x (B 4), dy_T4 1.343x / 1.673x, st_T4 1.281x / 1.631x. int8 dy_T2 1.398x / 2.611x, dy_T4 2.321x / 2.524x, st_T4 2.017x / 2.242x. Imbalance falls to 1.02–1.12 on gu at T/2–T/4, but the extra np.take / copyto / matmul calls per tile, each a GIL round trip, cost far more than the idle time recovered. int8's off-CPU share grows with tile count (int8g B 4 gu dy_T4 wall 10.5 ms vs 2.1 ms CPU in run 1).
- **Does it narrow the gap to int8?** A little, not enough. Best t5 / best int8 (each side's fastest mode, paired per round): B 1 **1.652x (st_T/st_T) -> 1.490x** (t5 dy_T), B 4 **2.199x -> 2.124x** (t5 dy_T / int8 dy_T). Both stay far above the 1.3x bar. The t5_T4 / int8_T4 ratios of 1.05x (B 1) and 0.95x dy_T4 B 1 do not show t5 winning: they come from int8 slowing down more at tiny tiles (both sides are 1.3–2.4x slower than their best).
- **Error:** every T and T/2 mode is bit-identical to its st_T layer (B 1 all modes; B 4 all except the T/4 modes and int8 dy_T2, where the matmul blocking changes: max |Δ| vs st_T 1.1–1.3e-06 t5, 4.8–5.0e-06 int8). vs float64 refs: t5 max |err| 5.57e-07 (B 1) / 1.23e-06 (B 4), cos 0.99999999999999; int8 3.19e-06 / 4.42e-06, cos 0.9999999999998.
- **Bytes:** weights per layer 13,986,304 B t5 vs 69,559,296 B int8 (4.973x), unchanged. Scratch for all 8 workers shrinks with the tile: tmL 50,982,912 / 25,491,456 / 12,745,728 B at T 384 / 192 / 96 (+ B 1 table 1,344,276 B); t5vg 70,809,600 / 35,404,800 / 17,702,400 B at 256 / 128 / 64; int8 35,389,440 / 17,694,720 / 8,847,360 B at 128 / 64 / 32. Dynamic claiming adds one itertools.count per site.

Bar used: a scheduling change counts if it gives ≥ 1.1x layer time (ratio ≤ 0.909), same as the 10:41 note. Ternary is "viable" only if t5 beats or gets within the 1.3x gap-to-int8 bar of this series on speed (≥ 1.5x on speed or memory vs int8 for a new win). The memory win is already known, so it does not count as new. Dynamic claiming clears the 1.1x scheduling bar for t5 but leaves t5 1.49–2.12x slower than int8, so the verdict is **not-viable**.

## Method

- Same layer, weights, P5 base-3 pack, kernels and float64 references as the 05:45 / 10:41 notes. dyn.py execs /workspace/bench0545_1125/xb5.py without its trailing main() call (-> /workspace/bench1455_1118 base.py / par.py: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gu / down; output rows per site 3840 / 2560 / 13824 / 2560). Kernels rows_tmL, rows_t5vg, rows8d, rows8g are xb5's objects, unchanged.
- Paths (the 05:45 best): B 1: t5 tmL (T-MAC 243-entry B tables, T 384), int8 8d (cast + per-row np.dot, T 128). B 4: t5 t5vg (void5 LUT expand + tile @ x.T, T 256), int8 8g (cast + tile @ x.T, T 128).
- Modes, one submit/result round per site (tmL keeps its static table-build round before it; glue on the main thread as in xb5):
  - st_T: xb5's static split, nt = ceil(out / T), per = ceil(nt / W) contiguous tiles per worker (the baseline).
  - st_T4: the same static split at T/4 (control: tile size alone).
  - dy_T, dy_T2, dy_T4: 8 tasks always submitted. Each loops j = next(cnt) on a per-site itertools.count() (atomic under the GIL) and runs the kernel on rows [j·Tt, min(out, (j+1)·Tt)) until j·Tt ≥ out.
- Every task appends (site, worker, start, end relative to the site's submit, time.thread_time delta, tiles). It does this in all modes, so the timing overhead is the same. Per site: span = last end, slowest = max worker busy, mean = mean busy over all 8 workers (idle = 0), imbalance = slowest / mean. Profile = 25 extra layers per config after the timing rounds; per-run value = median over the 25.
- Timing per B: 1 warm-up, then 7 rounds x 5 reps per config, config order rotated each round. Value = median of round medians. Spread = min–max of round medians. Ratios paired per round. Two input seeds, each run twice (run1/run2 = seeds 1/2, rep1/rep2 = seeds 1/2 again). Tables give the median of the 4 run medians, with per-run values or the range of round ratios across all runs.
- Box: 8 CPUs, Python 3.13.5, numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 ThreadPoolExecutor, load ~0.5–0.7. No network, nothing bound, no installs.

Command:

    cd /workspace/bench1450_1125 && for r in "1 run1" "2 run2" "1 rep1" "2 rep2"; do set -- $r; OPENBLAS_NUM_THREADS=1 XSEED=$1 python3 dyn.py $2.json > $2.txt 2>&1; done && python3 agg.py > agg.txt

Script (/workspace/bench1450_1125/dyn.py; execs /workspace/bench0545_1125/xb5.py, whose full text is in the 05:45 note):

    #!/usr/bin/env python3
    """Per-worker load imbalance + dynamic tile claiming on the W=8 decode layer, full bitnetd layer, ctx-free, 14:50 MT 11/25.
    Base: /workspace/bench0545_1125/xb5.py (weights, P5 base-3 pack, kernels rows_tmL / rows_t5vg / rows8d / rows8g, float64 refs);
    its source is exec'd here without its trailing main() call, so every kernel and weight buffer is the same object.
    Modes per site (same kernels, same glue, one submit/result round per site; tmL keeps its static table-build round):
      st_T   : static split as xb5 (nt = ceil(out/T) tiles, per = ceil(nt/W) contiguous tiles per worker)  <- baseline
      st_T4  : same static split at tile T/4 (control: separates tile size from claiming)
      dy_T, dy_T2, dy_T4 : W tasks always submitted; each pulls tile j = next(itertools.count()) until j*Tt >= out (dynamic claiming).
    Every task records wall start/end (perf_counter), time.thread_time delta and tile count -> per-site slowest-worker busy,
    mean busy over W, span (submit -> last end), imbalance = max/mean.
    Paths: B 1: t5 tmL (T 384), int8 8d (T 128); B 4: t5 t5vg (T 256), int8 8g (T 128).
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 dyn.py out.json"""
    import json, os, sys, time, itertools
    import numpy as np
    SRC = open("/workspace/bench0545_1125/xb5.py").read(); SRC = SRC[:SRC.rstrip().rfind("main()")]
    _argv = sys.argv; exec(compile(SRC, "xb5.py", "exec")); sys.argv = _argv
    ROUNDS, REPS, PROF = 7, 5, 25
    pc, tt = time.perf_counter, time.thread_time
    def run_site(fn, R, arg, y, out, T, extra, mode, rec, site):
        pool = par.pool(W); t_sub = pc()
        def task(w, r0, r1, dyn, cnt):
            t0 = pc(); c0 = tt(); n = 0
            if dyn:
                while True:
                    o0 = next(cnt) * T
                    if o0 >= out: break
                    fn(R, arg, y, o0, min(out, o0 + T), T, w, extra); n += 1
            elif r0 < r1:
                fn(R, arg, y, r0, r1, T, w, extra); n = -(-(r1 - r0) // T)
            rec.append((site, w, t0 - t_sub, pc() - t_sub, tt() - c0, n))
        if mode == "st":
            nt = -(-out // T); per = -(-nt // W)
            futs = [pool.submit(task, w, min(out, w * per * T), min(out, (w + 1) * per * T), False, None) for w in range(W)]
        else:
            cnt = itertools.count(); futs = [pool.submit(task, w, 0, 0, True, cnt) for w in range(W)]
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
            run_site(rows_tmL, R, TABL[:B, :q].reshape(B, q * 243), y, out, T, B, mode, rec, k)
        elif kind == "8d":
            a = np.ascontiguousarray(a, np.float32); y = np.empty((B, out), np.float32)
            run_site(rows8d, R, a, y, out, T, i, mode, rec, k)
        else:
            fn = rows_t5vg if kind == "t5vg" else rows8g
            xT = np.ascontiguousarray(np.asarray(a, np.float32).T); yT = np.empty((out, B), np.float32)
            run_site(fn, R, xT, yT, out, T, i, mode, rec, k); y = np.ascontiguousarray(yT.T)
        y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
    def layer(x, kind, T, mode, rec):
        f = lambda k, a: mm(k, a, kind, T, mode, rec)
        h1 = rms(x); q, kk, v = f("qkv", h1)
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        o = f("o", att)[0]; x2 = x + o; h2 = rms(x2)
        g, u = f("gu", h2); z = (silu(g) * u).astype(np.float32)
        return x2 + f("down", z)[0]
    PATHS = {1: (("t5", "tmL", 384), ("i8", "8d", 128)), 4: (("t5", "t5vg", 256), ("i8", "8g", 128))}
    MODES = (("st_T", "st", 1), ("st_T4", "st", 4), ("dy_T", "dy", 1), ("dy_T2", "dy", 2), ("dy_T4", "dy", 4))
    def cfgs(B):
        c = {}
        for tag, kind, T in PATHS[B]:
            for mn, mode, dv in MODES: c[f"{tag}_{mn}"] = (kind, T // dv, mode)
        return c
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    def main():
        seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W)
        W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        res["bytes"] = dict(t5=W5, int8=W8, int8_over_t5=W8 / W5,
                            scr_tmL={T: W * T * F5 * 12 for T in (384, 192, 96)}, scr_t5vg={T: W * T * F5 * 25 for T in (256, 128, 64)},
                            scr_int8={T: W * T * F * 4 + W * T * (F // 4) * 4 for T in (128, 64, 32)}, tab_B1=F5 * 243 * 4)
        print("bytes", res["bytes"], flush=True)
        res["tiles"] = {k: {str(T): -(-(P5[k]["PR"].shape[0]) // T) for T in (384, 256, 192, 128, 96, 64, 32)} for k in SITES}
        print("tiles", res["tiles"], flush=True)
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
            for tag in ("t5", "i8"):
                for mn, _, _ in MODES[1:]:
                    a, b = f"{tag}_{mn}", f"{tag}_st_T"; v = [p / q for p, q in zip(rm[a], rm[b])]; rat[f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v))
            for mn, _, _ in MODES:
                a, b = f"t5_{mn}", f"i8_{mn}"; v = [p / q for p, q in zip(rm[a], rm[b])]; rat[f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v))
            bt = min((c for c in names if c.startswith("t5")), key=lambda c: o[c]["med"]); bi = min((c for c in names if c.startswith("i8")), key=lambda c: o[c]["med"])
            v = [p / q for p, q in zip(rm[bt], rm[bi])]; rat["best_t5/best_i8"] = dict(r=o[bt]["med"] / o[bi]["med"], lo=min(v), hi=max(v), a=bt, b=bi)
            res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
            for c in names: print(f"B{B} {c:9s} {C[c][0]:5s}_T{C[c][1]:<4d} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
            for k, v in rat.items(): print(f"B{B} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f})", flush=True)
            # profile: per site per worker busy
            pr = {}
            for c in names:
                acc = {}
                for _ in range(PROF):
                    rec = []; layer(xb, *C[c], rec); bys = {}
                    for site, w, t0, t1, cpu, n in rec: bys.setdefault(site, []).append((w, t0, t1, cpu, n))
                    for site, L in bys.items():
                        busy = np.zeros(W); cpuv = np.zeros(W); nt = np.zeros(W)
                        for w, t0, t1, cpu, n in L: busy[w] += t1 - t0; cpuv[w] += cpu; nt[w] += n
                        span = max(t1 for _, _, t1, _, _ in L); a = acc.setdefault(site, dict(span=[], mx=[], mean=[], cpu_mx=[], cpu_mean=[], act=[], tiles=[]))
                        a["span"].append(span * 1e6); a["mx"].append(busy.max() * 1e6); a["mean"].append(busy.mean() * 1e6)
                        a["cpu_mx"].append(cpuv.max() * 1e6); a["cpu_mean"].append(cpuv.mean() * 1e6); a["act"].append(int((nt > 0).sum())); a["tiles"].append(sorted(nt.astype(int).tolist()))
                pr[c] = {s: dict(span=float(np.median(a["span"])), mx=float(np.median(a["mx"])), mean=float(np.median(a["mean"])), imb=float(np.median(np.array(a["mx"]) / np.array(a["mean"]))),
                                 cpu_mx=float(np.median(a["cpu_mx"])), cpu_mean=float(np.median(a["cpu_mean"])), act=int(np.median(a["act"])), tiles=a["tiles"][-1]) for s, a in acc.items()}
                for s, d in pr[c].items(): print(f"B{B} prof {c:9s} {s:9s} span {d['span']:8.1f} us  slowest {d['mx']:8.1f}  mean {d['mean']:8.1f}  imb {d['imb']:.2f}  cpu max/mean {d['cpu_mx']:.0f}/{d['cpu_mean']:.0f}  active {d['act']}  tiles {d['tiles']}", flush=True)
            res["prof"][f"B{B}"] = pr
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    main()

Aggregator (/workspace/bench1450_1125/agg.py):

    import json, numpy as np
    R = [json.load(open(f"{n}.json")) for n in ("run1", "run2", "rep1", "rep2")]
    for B in ("B1", "B4"):
        print(f"## {B}")
        cf = list(R[0]["time"][B]["cfg"])
        for c in cf:
            m = [r["time"][B]["cfg"][c]["med"] for r in R]
            print(f"{c:10s} meds " + " ".join(f"{x:7.3f}" for x in m) + f"  median {np.median(m):7.3f}  min {min(m):.3f} max {max(m):.3f}")
        for k in R[0]["time"][B]["ratios"]:
            m = [r["time"][B]["ratios"][k]["r"] for r in R]
            lo = min(r["time"][B]["ratios"][k]["lo"] for r in R); hi = max(r["time"][B]["ratios"][k]["hi"] for r in R)
            extra = [r["time"][B]["ratios"][k].get("a", "") + "/" + r["time"][B]["ratios"][k].get("b", "") for r in R] if k.startswith("best") else ""
            print(f"{k:24s} " + " ".join(f"{x:.3f}" for x in m) + f"  median {np.median(m):.3f}  rounds [{lo:.3f}-{hi:.3f}] {extra}")
        for c in ("t5_st_T", "t5_dy_T", "t5_dy_T2", "i8_st_T", "i8_dy_T"):
            for s in ("qkv", "o", "gu", "down"):
                mx = [r["prof"][B][c][s]["mx"] for r in R]; mn = [r["prof"][B][c][s]["mean"] for r in R]; im = [r["prof"][B][c][s]["imb"] for r in R]
                sp = [r["prof"][B][c][s]["span"] for r in R]; act = [r["prof"][B][c][s]["act"] for r in R]; cm = [r["prof"][B][c][s]["cpu_mx"] for r in R]
                print(f"prof {c:9s} {s:5s} span {np.median(sp):7.0f} slowest {np.median(mx):7.0f} mean {np.median(mn):7.0f} imb {np.median(im):.2f} cpu_slowest {np.median(cm):6.0f} active {act} tiles {R[0]['prof'][B][c][s]['tiles']}")
        for s in ("qkv", "o", "gu", "down"):
            for tag in ("t5", "i8"):
                for d in ("dy_T", "dy_T2"):
                    if tag == "i8" and d == "dy_T2": continue
                    r_ = [r["prof"][B][f"{tag}_{d}"][s]["mx"] / r["prof"][B][f"{tag}_st_T"][s]["mx"] for r in R]
                    print(f"slowest {tag}_{d}/st_T {s:5s} " + " ".join(f"{x:.3f}" for x in r_) + f" median {np.median(r_):.3f}")
    for B in ("B1", "B4"):
        e = R[0]["err"][B]; print(B, {c: (f"{v['mab']:.2e}", round(v['cos'], 12), v["identical_st_T"]) for c, v in e.items()})
        print(B, "all identical over runs:", all(v["identical_st_T"] for r in R for v in r["err"][B].values()))

## Results

Layer time, ms (median of 4 run medians; per-run medians run1 / run2 / rep1 / rep2):

| B | config | kernel / tile | median ms | run medians | ratio vs own st_T (round range) |
|---|---|---|---|---|---|
| 1 | t5_st_T | tmL T384 | **11.768** | 11.486 / 12.066 / 12.050 / 9.499 | 1 |
| 1 | t5_st_T4 | tmL T96 | 14.652 | 14.421 / 14.883 / 15.746 / 13.066 | 1.281x (1.111–1.487) |
| 1 | t5_dy_T | tmL T384 | **10.534** | 10.353 / 10.746 / 10.715 / 8.651 | **0.896x** (0.757–1.020) |
| 1 | t5_dy_T2 | tmL T192 | 11.359 | 11.456 / 11.994 / 11.262 / 10.142 | 0.996x (0.800–1.143) |
| 1 | t5_dy_T4 | tmL T96 | 15.246 | 15.440 / 15.053 / 16.167 / 13.735 | 1.343x (1.023–1.744) |
| 1 | i8_st_T | 8d T128 | **6.964** | 6.858 / 7.069 / 7.397 / 5.952 | 1 |
| 1 | i8_st_T4 | 8d T32 | 14.323 | 14.194 / 14.705 / 14.452 / 11.690 | 2.017x (1.842–2.331) |
| 1 | i8_dy_T | 8d T128 | **6.842** | 6.780 / 6.904 / 7.402 / 6.160 | 0.995x (0.910–1.220) |
| 1 | i8_dy_T2 | 8d T64 | 9.913 | 9.955 / 9.872 / 10.356 / 8.175 | 1.398x (1.230–1.596) |
| 1 | i8_dy_T4 | 8d T32 | 16.526 | 16.433 / 16.618 / 16.945 / 13.311 | 2.321x (1.956–2.519) |
| 4 | t5_st_T | t5vg T256 | **18.620** | 19.272 / 20.496 / 17.048 / 17.967 | 1 |
| 4 | t5_st_T4 | t5vg T64 | 31.301 | 29.862 / 32.331 / 33.080 / 30.271 | 1.631x (1.200–2.131) |
| 4 | t5_dy_T | t5vg T256 | **16.590** | 16.493 / 17.064 / 16.687 / 16.169 | **0.878x** (0.683–1.033) |
| 4 | t5_dy_T2 | t5vg T128 | 17.535 | 16.412 / 18.130 / 18.250 / 16.941 | 0.914x (0.703–1.103) |
| 4 | t5_dy_T4 | t5vg T64 | 32.099 | 30.909 / 32.956 / 35.329 / 31.241 | 1.673x (1.259–2.292) |
| 4 | i8_st_T | 8g T128 | **8.518** | 7.915 / 8.868 / 8.168 / 9.271 | 1 |
| 4 | i8_st_T4 | 8g T32 | 18.300 | 18.726 / 20.368 / 17.874 / 16.899 | 2.242x (1.777–2.638) |
| 4 | i8_dy_T | 8g T128 | **7.796** | 7.665 / 8.101 / 7.927 / 7.388 | 0.941x (0.747–1.201) |
| 4 | i8_dy_T2 | 8g T64 | 21.380 | 22.095 / 23.870 / 20.665 / 18.747 | 2.611x (1.920–3.054) |
| 4 | i8_dy_T4 | 8g T32 | 20.921 | 21.922 / 23.133 / 19.919 / 18.580 | 2.524x (1.894–3.045) |

t5 / int8 gap (paired per round; median of 4 runs, round range):

| B | pair | run ratios | median | rounds |
|---|---|---|---|---|
| 1 | t5_st_T / i8_st_T (baseline gap) | 1.675 / 1.707 / 1.629 / 1.596 | **1.652x** | 1.364–2.006 |
| 1 | t5_dy_T / i8_dy_T | 1.527 / 1.556 / 1.448 / 1.404 | 1.487x | 1.215–1.747 |
| 1 | best t5 / best int8 (t5_dy_T vs i8_dy_T or i8_st_T) | 1.527 / 1.556 / 1.449 / 1.453 | **1.490x** | 1.293–1.747 |
| 1 | t5_st_T4 / i8_st_T4 (both ≥ 1.3x off their best) | 1.016 / 1.012 / 1.090 / 1.118 | 1.053x | 0.926–1.164 |
| 4 | t5_st_T / i8_st_T (baseline gap) | 2.435 / 2.311 / 2.087 / 1.938 | **2.199x** | 1.786–2.756 |
| 4 | t5_dy_T / i8_dy_T | 2.152 / 2.106 / 2.105 / 2.188 | 2.129x | 1.557–2.613 |
| 4 | best t5 / best int8 (t5_dy_T or dy_T2 vs i8_dy_T) | 2.141 / 2.106 / 2.105 / 2.188 | **2.124x** | 1.557–2.613 |

Per-site per-worker profile, µs (median of 4 runs; each run = median of 25 profiled layers). Slowest = max worker wall busy, mean = over all 8 workers, cpu = slowest worker's thread_time, tiles = per-worker tile counts (run 1, sorted):

| B | config | site | span | slowest | mean | imbalance | cpu slowest | active | tiles |
|---|---|---|---|---|---|---|---|---|---|
| 1 | t5_st_T | qkv | 1393 | 1255 | 683 | 1.78 | 1190 | 5 | 0,0,0,2,2,2,2,2 |
| 1 | t5_st_T | o | 956 | 804 | 539 | 1.45 | 674 | 7 | 0,1×7 |
| 1 | t5_st_T | gu | 3190 | 2990 | 2386 | 1.26 | 2501 | 8 | 1,5×7 |
| 1 | t5_st_T | down | 2335 | 2177 | 1588 | 1.33 | 2105 | 7 | 0,1×7 |
| 1 | t5_dy_T | qkv | 1258 | 1139 | 859 | 1.35 | 971 | 8 | 1×6,2,2 |
| 1 | t5_dy_T | o | 911 | 808 | 555 | 1.43 | 675 | 7 | 0,1×7 |
| 1 | t5_dy_T | gu | 2977 | 2882 | 2616 | 1.10 | 2415 | 8 | 4,4,4,4,5,5,5,5 |
| 1 | t5_dy_T | down | 2481 | 2357 | 1615 | 1.41 | 2246 | 7 | 0,1×7 |
| 1 | i8_st_T | qkv | 1108 | 954 | 726 | 1.29 | 523 | 8 | 2,4×7 |
| 1 | i8_st_T | o | 763 | 653 | 428 | 1.48 | 406 | 7 | 0,2,3×6 |
| 1 | i8_st_T | gu | 2725 | 2561 | 2231 | 1.14 | 1488 | 8 | 10,14×7 |
| 1 | i8_st_T | down | 1520 | 1355 | 859 | 1.49 | 1206 | 7 | 0,2,3×6 |
| 1 | i8_dy_T | qkv | 1133 | 1014 | 860 | 1.19 | 532 | 8 | 3,3,3,3,4,4,5,5 |
| 1 | i8_dy_T | o | 805 | 712 | 562 | 1.25 | 396 | 8 | 2×4,3×4 |
| 1 | i8_dy_T | gu | 3027 | 2911 | 2776 | 1.05 | 1527 | 8 | 12,12,13,13,13,14,14,17 |
| 1 | i8_dy_T | down | 1262 | 1128 | 934 | 1.21 | 1033 | 8 | 2×4,3×4 |
| 4 | t5_st_T | qkv | 2656 | 2416 | 1846 | 1.29 | 1670 | 8 | 1,2×7 |
| 4 | t5_st_T | o | 2093 | 1938 | 1040 | 1.83 | 1772 | 5 | 0,0,0,2×5 |
| 4 | t5_st_T | gu | 7178 | 6985 | 5835 | 1.19 | 5606 | 8 | 5,7×7 |
| 4 | t5_st_T | down | 4709 | 4457 | 2568 | 1.73 | 4348 | 5 | 0,0,0,2×5 |
| 4 | t5_dy_T | qkv | 2488 | 2321 | 1888 | 1.20 | 1834 | 8 | 1,2×7 |
| 4 | t5_dy_T | o | 1851 | 1691 | 1219 | 1.40 | 1474 | 8 | 1×6,2,2 |
| 4 | t5_dy_T | gu | 6590 | 6416 | 5974 | 1.07 | 5687 | 8 | 6,6,6,7,7,7,7,8 |
| 4 | t5_dy_T | down | 4166 | 4009 | 2789 | 1.48 | 3896 | 8 | 1×6,2,2 |
| 4 | i8_st_T | qkv | 1217 | 1012 | 773 | 1.27 | 678 | 8 | 2,4×7 |
| 4 | i8_st_T | o | 851 | 680 | 482 | 1.40 | 539 | 7 | 0,2,3×6 |
| 4 | i8_st_T | gu | 3035 | 2907 | 2263 | 1.16 | 2080 | 8 | 10,14×7 |
| 4 | i8_st_T | down | 1656 | 1505 | 1131 | 1.32 | 1400 | 7 | 0,2,3×6 |
| 4 | i8_dy_T | qkv | 1277 | 1150 | 941 | 1.18 | 786 | 8 | 2,3,3,4,4,4,5,5 |
| 4 | i8_dy_T | o | 948 | 837 | 617 | 1.26 | 572 | 8 | 2×4,3×4 |
| 4 | i8_dy_T | gu | 3354 | 3232 | 3053 | 1.05 | 2400 | 8 | 11,13,13,14,14,14,14,15 |
| 4 | i8_dy_T | down | 1710 | 1542 | 1301 | 1.25 | 1479 | 8 | 2×4,3×4 |

Slowest-worker ratio dy_T / st_T per site (run1 / run2 / rep1 / rep2 -> median): t5 B 1 qkv 0.730 / 0.867 / 1.151 / 1.054 -> 0.960, o 1.028, gu 0.996, down 1.020; t5 B 4 qkv 0.927, o 0.798 / 0.758 / 0.919 / 0.973 -> 0.858, gu 0.931, down 0.746 / 0.830 / 0.910 / 0.940 -> 0.870; int8 B 1 qkv 1.063, o 1.091, gu 1.138, down 0.844; int8 B 4 qkv 1.114, o 1.121, gu 1.179, down 0.980. The tmL table-build rounds (static in every mode) show imbalance 1.7–1.95 at ~110–290 µs per site, small next to the row rounds. (o / down at T 384 have 7 tiles, so one worker idles in any scheme.)

Error vs float64 refs (identical in all 4 runs per seed): B 1 t5 every mode max |err| 5.57e-07, cos 0.999999999999991; int8 3.19e-06, cos 1 - 1e-13. B 4 t5 T / T2 modes 1.23e-06 (cos 0.99999999999997), T4 modes 7.6e-07; int8 T modes 4.42e-06 (cos 0.9999999999998), T2/T4 2.8–3.1e-06. Bit-identical to st_T: all B 1 modes; B 4 dy_T / dy_T2 (t5) and dy_T (int8).

Bytes: weights per layer 13,986,304 B t5 vs 69,559,296 B int8 (4.973x), unchanged. Scratch for all 8 workers: tmL 50,982,912 / 25,491,456 / 12,745,728 B (T 384 / 192 / 96) + B 1 table 1,344,276 B; t5vg 70,809,600 / 35,404,800 / 17,702,400 B (T 256 / 128 / 64); int8 35,389,440 / 17,694,720 / 8,847,360 B (T 128 / 64 / 32).

## Pair files this wave

No pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) for this wave reached this lane, so none are linked; pairs idle. The depth-1 clone of JuniorLLM (14:55 MT) shows this lane's own 88cd86c (10:48 MT, the 10:41 note) as HEAD on main. Nothing is newer, so no Trit Pack, Handshake, LLC or unattributed commit in the last 6 hours (since ~08:50 MT) touches junior_osai/vault/findings. This lane did not read JuniorHome this wave.

## Next question

Keep dynamic claiming and fix its two weaknesses: the tail and the per-tile cost. Guided self-scheduling on the W 8 decode layer: workers claim large tiles first (T, or 2T for int8 / t5vg gu) from the shared counter, then switch to T/2 and T/4 tiles only when fewer than W·T rows are left (a two- or three-level chunk schedule precomputed per site, claimed by index). Also try longest-site-first ordering for the fused gu site. Does guided claiming beat dy_T by ≥ 1.05x and st_T by ≥ 1.15x for t5 tmL_T384 (B 1) and t5vg_T256 (B 4)? Does it finally help int8 (≥ 1.1x vs st_T)? And what is left of t5's 1.49x / 2.12x gap to int8? Record per-worker wall vs thread_time again to check whether int8's ~40% off-CPU share grows with chunk count. No earlier note tested guided or multi-level chunk schedules.

close: f607d9f42e31c31d
