---
local: true
date: 2026-11-25
wave: "2026-11-25 10:55 slot (fired 10:41 MT, ran ~10:41–10:55 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (decode-layer pool dispatch, full bitnetd layer, W 8, ctx-free: one persistent task per worker per layer synchronized on a threading.Barrier(8) gives the same output bit for bit but does not cut layer time by 1.1x on any path. Median barrier / submit over 6 paired runs (3 passes x 2 seeds): int8d B 1 1.064x (slower), tmL B 1 0.975x, int8g B 4 0.949x, t5vg B 4 0.958x. Only 2 of 24 run medians reach the 1.1x bar (int8g B 4, 0.861x and 0.894x). One Barrier.wait costs 132–143 µs vs 121–132 µs for one submit/result round, and the barrier layer needs 2 syncs per site (3 for tmL) where submit needs 1 (2). The t5 / int8 gap barely moves: B 1 1.74x -> 1.64x, B 4 2.27x -> 2.30x (medians of 6). No 1.5x speed or memory gain; the 4.973x t5 weight-memory win vs int8 is already known)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, dispatch-overhead, barrier, persistent-workers, sync, base-3, 1p6-bit, five-trit-pack, tmac, lut-expand, int8, numpy, not-viable]
---

# Pool dispatch overhead on the W 8 decode layer: a persistent-worker Barrier layer gives no 1.1x gain over four submit/result rounds, and t5's 1.6–2.3x gap to int8 stays

The 05:45 note ([[2026-11-25_0545_b1-b32-small-batch-crossover-base3-t5-tmac-243-entry-b-tables-vs-void5-lut-expand-matmul-full-bitnetd-layer-8-thread-pool-crossover-b2-b4-gap-to-int8-1p7-2p7x]]) found the best t5 path 1.7–2.7x slower than int8 at every B and asked: today a decode layer makes four submit/result rounds on the W 8 ThreadPool (one per site qkv / o / gu / down; tmL makes eight, because each site also has a table-build round), with rms, repeat, silu, gamma, splits, transposes and table-input padding on the main thread between rounds. Does one persistent task per worker per layer, walking all four sites and syncing on a threading.Barrier with worker 0 doing the serial glue, cut layer time by ≥ 1.1x for int8 or t5 at B 1 and B 4 on the best paths, and does it narrow t5's gap to int8? First size the fixed dispatch overhead with an empty-work layer for both schemes.

This is not a repeat. No earlier note measured pool dispatch or sync overhead, an empty-work layer, or a barrier-synchronized persistent-worker layer. All earlier W 8 notes (11-16 onward) used per-site submit/result rounds only.

Answers:
- **Empty-work dispatch (fixed overhead):** 4 submit/result rounds of 8 no-op tasks take **486–493 µs** per layer (pass A / B, both seed processes; 668–686 µs in the first, noisier pass 0). The barrier scheme for the same layer (1 submit round + 8 barrier passes) takes **1,138–1,198 µs** (2.31x–2.44x more). For tmL, 8 rounds take 978–1,019 µs vs 1,674–2,055 µs for 1 round + 12 barriers (1.70x–2.05x). The marginal cost of one Barrier(8).wait is **132–143 µs**. One extra submit/result round costs **121–132 µs**. So a Python threading.Barrier is not cheaper than a submit/result round. It is slightly dearer, because it also wakes every waiter through a Condition. The barrier layer needs two syncs per site (after w0's glue, after the row tiles) and three for tmL (plus the table build), where the submit layer needs one (two). Dispatch is ~0.5 ms of a 5–6 ms int8d B 1 layer (8–10%) and ~1.0 ms of a 9–11 ms tmL B 1 layer (9–11%). Removing all of it would give at most ~1.10x / ~1.12x.
- **Does the barrier layer cut layer time by ≥ 1.1x (ratio ≤ 0.909)?** **No.** Paired barrier / submit, seed 1 / seed 2, pass 0 / A / B:
  - int8d_T128 at B 1: 1.078 / 0.994, 0.988 / 1.065, 1.063 / 1.079 (median 1.064x, i.e. slower).
  - tmL_T384 at B 1: 0.946 / 1.003, 0.967 / 0.964, 0.982 / 1.041 (median 0.975x).
  - int8g_T128 at B 4: 0.941 / 0.992, 0.957 / **0.861**, **0.894** / 0.964 (median 0.949x).
  - t5vg_T256 at B 4: 0.933 / 0.934, 0.970 / 0.966, 1.034 / 0.951 (median 0.958x).
  - Only 2 of 24 run medians reach the 1.1x bar (both int8g at B 4), and the same config in the other runs reads 0.94–0.99x. The real layers do better than the empty-work numbers predict (the barrier scheme is not 0.7 ms slower), because workers reach each barrier at staggered times, so most of the wake-up cost hides behind the slowest worker's tiles. The gain is still at most ~5% on median.
- **Does it narrow t5's gap to int8?** **Barely at B 1, not at B 4.** Paired t5 / int8, submit -> barrier, seed 1 / seed 2:
  - B 1, pass 0: 1.879 -> 1.649 / 1.775 -> 1.791.
  - B 1, pass A: 1.555 -> 1.522 / 1.821 -> 1.650.
  - B 1, pass B: 1.700 -> 1.570 / 1.682 -> 1.623.
  - B 1 median of 6: **1.74x -> 1.64x**. This is the barrier helping tmL a little (8 rounds -> 12 barriers, with its table builds now in the same task) while int8d gets slightly worse.
  - B 4: 2.289 -> 2.269 / 2.258 -> 2.126, 2.296 -> 2.327 / 2.242 -> 2.516, 2.049 -> 2.371 / 2.318 -> 2.286. Median **2.27x -> 2.30x**, no change.
  - Both are far from the 1.5x-better t5-vs-int8 bar.
- **Added variant bar1 (one barrier per site, glue recomputed redundantly by every worker):** slower everywhere. bar1 / submit is 1.19–1.27x for int8d B 1, 1.08–1.13x for tmL B 1, 1.09–1.23x for int8g B 4 and 1.00–1.14x for t5vg B 4. Eight copies of rms / silu / transpose / x_pad fight over the GIL, and that costs more than the barrier they save (its empty-work dispatch is 605–628 µs, 1.23–1.27x the 4-round submit). bar1 narrows the B 1 gap to 1.46–1.58x only because it slows int8d more than tmL.
- **Correctness:** the barrier and bar1 layers are **bit-identical** to the submit layer at B 1 and B 4 on all four paths, both seeds (max abs 0, cos 1.0, np.array_equal True). vs the float64 reference on the first B of 32 fixed rows: tmL B 1 max abs 5.6e-07 / 6.2e-07, t5vg B 4 1.2e-06 / 9.9e-07, cos ≥ 0.99999999999997; int8d B 1 3.2e-06 / 3.4e-06, int8g B 4 4.4e-06 / 4.1e-06, cos ≥ 0.99999999999982 (seed 1 / seed 2; same as 05:45).
- **Memory:** unchanged. t5 weights 13,986,304 B vs int8 69,559,296 B (4.973x, already known). The barrier scheme adds no buffers (the same per-worker scratch and site outputs). bar1 adds 8 worker-local activation copies (≤ 8 x 4 x 6,915 x 4 B ≈ 0.9 MB at B 4).
- **Verdict: not-viable.** On this box the pool's dispatch costs ~120–140 µs per sync, whether the sync is a submit/result round or a Barrier.wait. The barrier layer cannot remove syncs, because worker 0's serial glue must be fenced on both sides, so it does not shrink the 8–11% dispatch share. t5's gap to int8 comes from the kernels (expand / gather cost per weight pass), not from the pool.

## Method

- Same layer, weights, P5 base-3 pack, kernels and float64 references as the 05:45 note. bar.py execs /workspace/bench0545_1125/xb5.py without its trailing main() call (→ /workspace/bench1455_1118 base.py / par.py: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gate|up / down, per-row gamma, attention stub (row-local, ctx-free)). So rows_tmL / rows_t5vg / rows8d / rows8g, bldL, TABL / XPL, the per-worker scratch and the pool par.pool(8) are the same objects.
- Paths (05:45 best): B 1: t5 tmL_T384 (T-MAC, B tables), int8 int8d_T128 (cast + per-row np.dot). B 4: t5 t5vg_T256 (void5 expand + tile @ x.T), int8 int8g_T128 (cast + tile @ x.T).
- sub_*: xb5's layer5 / layerg unchanged. One submit/result round of ≤ 8 row-range tasks per site, plus one table-build round per site for tmL. The main thread does the glue between rounds.
- bar_*: 8 tasks per layer, one per worker, each walking all four sites. Phases per site: worker 0 glue (gamma * y + split of the previous site, repeat / residual / rms / silu, transpose or x_pad staging, output alloc) | Barrier(8).wait | (tmL: table-build slice | wait) | own row range (the same ceil(out/T) tiles, ceil(tiles/W) per worker as sub) | wait. Worker 0 does the final residual. Barrier.wait has a 30 s timeout, and any exception aborts the barrier.
- bar1_*: same persistent workers, but every worker recomputes the glue into worker-local arrays (output buffers preallocated by the caller), so only one barrier per site is needed (tmL: two).
- emp_*: the same dispatch with no math: emp_sub_R = R rounds of 8 no-op submits + result(); emp_bar_N = 1 round of 8 tasks each passing N barrier waits. int8 / t5vg: sub 4 rounds vs bar 8 barriers (bar1: 4). tmL: 8 rounds vs 12 barriers (bar1: 8). 7 rounds x 60 reps, rotated order; median of round medians.
- Timing per B: 1 warm-up, then 7 rounds x 5 reps per config, with config order rotated each round. Value = median of the round medians; spread = min–max of the round medians; ratios are paired per round. Two input seeds (weights fixed by base.SEED), each run in two passes (A = run*.json, B = rep*.json). An earlier pass 0 (bar.py before the bar1 / emp_bar_4 additions, same sub / bar code; kept as v1/) is reported for the barrier / submit ratios only. OPENBLAS_NUM_THREADS=1, W 8 ThreadPoolExecutor on 8 cores, numpy 2.5.4. ~40 s per seed per pass. python3 + numpy only; no installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench1041_1125 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 900 python3 bar.py run$s.json > run$s.txt 2>&1; done && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 900 python3 bar.py rep$s.json > rep$s.txt 2>&1; done

Script (/workspace/bench1041_1125/bar.py; execs /workspace/bench0545_1125/xb5.py, whose full text is in the 05:45 note):

    #!/usr/bin/env python3
    """Decode-layer dispatch / sync overhead on the W=8 ThreadPool, full bitnetd layer, ctx-free, 10:41 MT 11/25.
    Base: /workspace/bench0545_1125/xb5.py (weights, P5 base-3 pack, kernels rows_tmL / rows_t5vg / rows8d / rows8g, layer5 / layerg,
    float64 refs); its source is exec'd here without its trailing main() call, so every kernel and buffer is the same object.
    sub_* : the xb5 layer as-is: one pool.submit/result round per site (tmL: 2 per site, table build + rows), main thread does
            rms / repeat / silu / residual / gamma / split / transposes / table-input pad between rounds.
    bar_* : one persistent task per worker per layer (8 submits, 1 join); the workers walk all four sites and sync on a
            threading.Barrier(8) between phases; worker 0 does the serial glue (same numpy calls, same order) while the others wait.
            Phases per site: [w0 prep] | barrier | (tmL: [table build slice] | barrier) | [row tiles] | barrier; final residual on w0.
    emp_* : the same dispatch with no math (no-op tasks / barrier waits only): emp_sub_R = R submit/result rounds of 8 tasks,
            emp_bar_N = 1 round of 8 tasks each passing N barriers. int8d/int8g/t5vg: R 4 vs N 8 (bar1: N 4); tmL: R 8 vs N 12 (bar1: N 8).
    bar1_*: added variant: same persistent workers, but every worker recomputes the serial glue itself (redundant rms / repeat /
            silu / residual / gamma*y copy / transposes / table-input pad, into worker-local arrays; output buffers preallocated
            by the caller), so one barrier per site is enough (tmL: 2, table build + rows); worker 0 does the final residual.
    Paths: B 1: t5 tmL_T384, int8 int8d_T128; B 4: t5 t5vg_T256, int8 int8g_T128 (the 05:45 best paths).
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 bar.py out.json"""
    import json, os, sys, time, threading
    import numpy as np
    SRC = open("/workspace/bench0545_1125/xb5.py").read(); SRC = SRC[:SRC.rstrip().rfind("main()")]
    _argv = sys.argv; exec(compile(SRC, "xb5.py", "exec")); sys.argv = _argv
    ROUNDS, REPS, EREPS = 7, 5, 60
    def site_spec(k, kind):
        if kind in ("tmL", "t5vg"): P = P5[k]; R = P["PR"]; i = P["inp"]
        else: P = par.F8[k]; R = P["Q"]; i = R.shape[1]
        return P, R, i
    def layer_bar(x, kind, T):
        """kind in tmL / t5vg / 8d / 8g; identical math to layer5 / layerg, barrier-synchronized persistent workers."""
        B = x.shape[0]; pool = par.pool(W); bar = threading.Barrier(W); st = {}
        fn = dict(tmL=rows_tmL, t5vg=rows_t5vg, **{"8d": rows8d, "8g": rows8g})[kind]
        def post(k):                                    # w0: gamma + split of the site output, as fmm* return it
            P = st["P"]
            y = np.ascontiguousarray(st["yT"].T) if kind in ("t5vg", "8g") else st["y"]
            y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
        def prep(k, a):                                 # w0: stage the site input and output buffers
            P, R, i = site_spec(k, kind); out = R.shape[0]; st.update(P=P, R=R, i=i, out=out)
            if kind in ("t5vg", "8g"):
                st["arg"] = np.ascontiguousarray(np.asarray(a, np.float32).T); st["yT"] = np.empty((out, B), np.float32); st["yo"] = st["yT"]; st["ex"] = i
            else:
                a = np.ascontiguousarray(a, np.float32); st["y"] = np.empty((B, out), np.float32); st["yo"] = st["y"]
                if kind == "tmL":
                    q = R.shape[1]; xp = XPL[:B, :q * 5]; xp[:, :i] = a; xp[:, i:] = 0
                    st.update(xp=xp, q=q, tab=TABL[:B, :q], arg=TABL[:B, :q].reshape(B, q * 243), ex=B)
                else: st["arg"] = a; st["ex"] = i
        def wk(w):
            try:
                for k in SITES:
                    if w == 0:
                        if k == "qkv": prep(k, rms(x))
                        elif k == "o":
                            q_, kk_, v = post("qkv"); prep(k, np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D))
                        elif k == "gu":
                            o = post("o")[0]; st["x2"] = x + o; prep(k, rms(st["x2"]))
                        else:
                            g, u = post("gu"); prep(k, (silu(g) * u).astype(np.float32))
                    bar.wait(30)
                    if kind == "tmL":
                        q = st["q"]; gper = -(-q // W)
                        if w * gper < q: bldL(st["xp"], st["tab"], B, w * gper, min(q, (w + 1) * gper), q)
                        bar.wait(30)
                    out = st["out"]; nt = -(-out // T); per = -(-nt // W); r0 = w * per * T
                    if r0 < out: fn(st["R"], st["arg"], st["yo"], r0, min(out, (w + 1) * per * T), T, w, st["ex"])
                    bar.wait(30)
                if w == 0: st["res"] = st["x2"] + post("down")[0]
            except BaseException:
                bar.abort(); raise
        futs = [pool.submit(wk, w) for w in range(W)]
        for f in futs: f.result()
        return st["res"]
    def layer_bar1(x, kind, T):
        B = x.shape[0]; pool = par.pool(W); bar = threading.Barrier(W); st = {}
        fn = dict(tmL=rows_tmL, t5vg=rows_t5vg, **{"8d": rows8d, "8g": rows8g})[kind]; g_ = kind in ("t5vg", "8g")
        Y = {}
        for k in SITES:
            P, R, i = site_spec(k, kind); out = R.shape[0]; Y[k] = np.empty((out, B) if g_ else (B, out), np.float32)
        def post(k):                                    # worker-local gamma + split (no in-place write to the shared buffer)
            P = site_spec(k, kind)[0]; y = np.ascontiguousarray(Y[k].T) if g_ else Y[k]
            y = y * P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
        def wk(w):
            try:
                a = rms(x); x2 = None
                for k in SITES:
                    if k == "o":
                        q_, kk_, v = post("qkv"); a = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
                    elif k == "gu":
                        x2 = x + post("o")[0]; a = rms(x2)
                    elif k == "down":
                        g, u = post("gu"); a = (silu(g) * u).astype(np.float32)
                    P, R, i = site_spec(k, kind); out = R.shape[0]
                    if g_: arg = np.ascontiguousarray(np.asarray(a, np.float32).T); ex = i
                    elif kind == "tmL":
                        q = R.shape[1]; gper = -(-q // W); g0, g1 = w * gper, min(q, (w + 1) * gper)
                        if g0 < q:
                            xp = np.zeros((B, q * 5), np.float32); xp[:, :i] = a; bldL(xp, TABL[:B, :q], B, g0, g1, q)
                        bar.wait(30); arg = TABL[:B, :q].reshape(B, q * 243); ex = B
                    else: arg = np.ascontiguousarray(a, np.float32); ex = i
                    nt = -(-out // T); per = -(-nt // W); r0 = w * per * T
                    if r0 < out: fn(R, arg, Y[k], r0, min(out, (w + 1) * per * T), T, w, ex)
                    bar.wait(30)
                if w == 0: st["res"] = x2 + post("down")[0]
            except BaseException:
                bar.abort(); raise
        futs = [pool.submit(wk, w) for w in range(W)]
        for f in futs: f.result()
        return st["res"]
    def _noop(): return None
    def emp_sub(R):
        pool = par.pool(W)
        for _ in range(R):
            fs = [pool.submit(_noop) for _ in range(W)]
            for f in fs: f.result()
    def emp_bar(N):
        pool = par.pool(W); bar = threading.Barrier(W)
        def wk():
            for _ in range(N): bar.wait(30)
        fs = [pool.submit(wk) for _ in range(W)]
        for f in fs: f.result()
    PATHS = {1: dict(t5=("tmL", 384), i8=("8d", 128)), 4: dict(t5=("t5vg", 256), i8=("8g", 128))}
    def sub_layer(kind, T):
        if kind == "tmL": return lambda x: layer5(x, "tmL", T)
        return lambda x: layerg(x, kind, T)
    def main():
        seed = int(os.environ.get("XSEED", "1")); res = {"xseed": seed, "numpy": np.__version__, "cpus": os.cpu_count(), "W": W, "err": {}, "time": {}, "empty": {}}
        # empty-work dispatch
        EC = {"emp_sub_4": lambda: emp_sub(4), "emp_bar_8": lambda: emp_bar(8), "emp_bar_4": lambda: emp_bar(4), "emp_sub_8": lambda: emp_sub(8), "emp_bar_12": lambda: emp_bar(12)}
        for c in EC: EC[c]()
        em = {c: [] for c in EC}; names = list(EC)
        for rd in range(ROUNDS):
            for c in names[rd % len(names):] + names[:rd % len(names)]:
                s = []
                for _ in range(EREPS):
                    t0 = time.perf_counter(); EC[c](); s.append((time.perf_counter() - t0) * 1e6)
                em[c].append(float(np.median(s)))
        for c, v in em.items():
            res["empty"][c] = dict(med_us=float(np.median(v)), lo=min(v), hi=max(v), rm=v); print(f"empty {c:11s} {np.median(v):8.1f} us [{min(v):.1f}-{max(v):.1f}]", flush=True)
        for a, b in (("emp_bar_8", "emp_sub_4"), ("emp_bar_4", "emp_sub_4"), ("emp_bar_12", "emp_sub_8"), ("emp_bar_8", "emp_sub_8")):
            v = [p / q for p, q in zip(em[a], em[b])]; res["empty"][f"{a}/{b}"] = dict(r=float(np.median(em[a]) / np.median(em[b])), lo=min(v), hi=max(v))
            print(f"empty ratio {a}/{b} {res['empty'][a + '/' + b]['r']:.3f}x (rounds {min(v):.2f}-{max(v):.2f})", flush=True)
        for B, pth in PATHS.items():
            CFG2 = {}
            for lab, (kind, T) in pth.items():
                CFG2[f"sub_{lab}"] = sub_layer(kind, T); CFG2[f"bar_{lab}"] = (lambda kind, T: (lambda x: layer_bar(x, kind, T)))(kind, T)
                CFG2[f"bar1_{lab}"] = (lambda kind, T: (lambda x: layer_bar1(x, kind, T)))(kind, T)
            tag = {c: pth[c.split("_")[1]] for c in CFG2}
            xb = XREF[:B]; e = {}
            for c, f in CFG2.items():
                y = f(xb); r = REF[8 if c.endswith("i8") else 2][:B]; e[c] = dict(mab=float(np.abs(y - r).max()), cos=cos(y, r))
            for lab in ("t5", "i8"):
                b = CFG2[f"sub_{lab}"](xb)
                for v_ in ("bar", "bar1"):
                    a = CFG2[f"{v_}_{lab}"](xb); e[f"{v_}_{lab}_vs_sub"] = dict(mab=float(np.abs(a - b).max()), cos=cos(a, b), identical=bool(np.array_equal(a, b)))
            res["err"][f"B{B}"] = e; print(f"err B{B}", e, flush=True)
            xr = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32); rm = {c: [] for c in CFG2}; nm = list(CFG2)
            for c in nm: CFG2[c](xr)
            for rd in range(ROUNDS):
                for c in nm[rd % len(nm):] + nm[:rd % len(nm)]:
                    s = []
                    for _ in range(REPS):
                        t0 = time.perf_counter(); CFG2[c](xr); s.append((time.perf_counter() - t0) * 1e3)
                    rm[c].append(float(np.median(s)))
            o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), rm=v, path=f"{tag[c][0]}_T{tag[c][1]}") for c, v in rm.items()}; rat = {}
            for a, b in (("bar_i8", "sub_i8"), ("bar_t5", "sub_t5"), ("bar1_i8", "sub_i8"), ("bar1_t5", "sub_t5"), ("sub_t5", "sub_i8"), ("bar_t5", "bar_i8"), ("bar1_t5", "bar1_i8")):
                v = [p / q for p, q in zip(rm[a], rm[b])]; rat[f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v))
            res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
            for c in nm: print(f"B{B} {c:8s} {o[c]['path']:10s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
            for k, v in rat.items(): print(f"B{B} ratio {k:16s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f})", flush=True)
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    main()

Pass 0 (v1/bar_v1.py, the same sub_* / bar_* code without bar1 or emp_bar_4; seed 1 / seed 2): empty 4 rounds 686 / 668 µs, 1 round + 8 barriers 1,643 / 1,673 µs (2.394x / 2.505x), 8 rounds 1,381 / 1,360 µs, 1 round + 12 barriers 2,383 / 2,415 µs (1.725x / 1.776x). Layer bar / sub: int8d B 1 1.078x (0.80–1.22) / 0.994x (0.90–1.05); tmL B 1 0.946x (0.71–1.15) / 1.003x (0.86–1.13); int8g B 4 0.941x (0.90–1.09) / 0.992x (0.89–1.18); t5vg B 4 0.933x (0.71–1.07) / 0.934x (0.91–1.03). t5 / int8 submit -> barrier: B 1 1.879 -> 1.649 / 1.775 -> 1.791; B 4 2.289 -> 2.269 / 2.258 -> 2.126. Layer ms (sub / bar): int8d B 1 6.37 / 6.87, 6.68 / 6.64; tmL B 1 11.97 / 11.32, 11.85 / 11.89; int8g B 4 8.79 / 8.27, 8.51 / 8.44; t5vg B 4 20.13 / 18.78, 19.20 / 17.94.

## Results

Layer time, ms, median of 7 round medians [min–max], seed 1 / seed 2, pass A (run*.json) and pass B (rep*.json). B 1 paths: t5 tmL_T384, int8 int8d_T128; B 4: t5 t5vg_T256, int8 int8g_T128.

| config | B 1 pass A | B 1 pass B | B 4 pass A | B 4 pass B |
|---|---|---|---|---|
| int8 submit (sub_i8) | 5.83 [5.4–7.1] / 6.28 [6.0–7.9] | 5.09 [4.8–6.4] / 5.28 [5.1–6.6] | 7.64 [7.0–8.2] / 7.75 [6.3–8.5] | 7.37 [6.8–8.6] / 7.45 [7.1–8.4] |
| int8 barrier (bar_i8) | 5.77 [5.5–7.4] / 6.68 [6.5–7.8] | 5.42 [5.1–5.7] / 5.69 [5.2–6.2] | 7.32 [7.0–8.5] / 6.67 [6.0–8.3] | 6.59 [6.2–9.0] / 7.19 [6.6–7.9] |
| int8 bar1 (bar1_i8) | 6.94 [6.5–8.9] / 7.78 [7.2–8.1] | 6.49 [6.2–6.9] / 6.67 [6.3–7.3] | 8.90 [8.2–10.9] / 8.42 [7.9–11.4] | 8.20 [7.6–10.7] / 9.20 [8.4–9.7] |
| t5 submit (sub_t5) | 9.08 [8.4–11.6] / 11.43 [8.1–12.6] | 8.66 [8.1–10.6] / 8.87 [7.8–10.7] | 17.55 [16.4–19.2] / 17.37 [16.2–21.3] | 15.11 [14.9–18.8] / 17.28 [16.8–19.3] |
| t5 barrier (bar_t5) | 8.78 [8.3–11.4] / 11.03 [7.7–13.0] | 8.50 [7.8–10.0] / 9.24 [8.5–10.5] | 17.03 [15.6–18.6] / 16.78 [15.5–20.6] | 15.63 [14.7–16.2] / 16.43 [15.7–19.2] |
| t5 bar1 (bar1_t5) | 10.16 [9.1–13.2] / 12.31 [11.7–14.3] | 9.64 [8.9–10.4] / 10.02 [9.0–12.5] | 17.60 [16.9–19.8] / 19.85 [16.7–25.1] | 16.72 [15.6–19.1] / 19.35 [17.9–20.0] |

Paired ratios (per-round range in brackets), seed 1 / seed 2:

| ratio | B 1 pass A | B 1 pass B | B 4 pass A | B 4 pass B |
|---|---|---|---|---|
| bar_i8/sub_i8 | 0.988x (0.83–1.09) / 1.065x (0.96–1.14) | 1.063x (0.89–1.07) / 1.079x (0.94–1.09) | 0.957x (0.90–1.04) / 0.861x (0.86–0.97) | 0.894x (0.82–1.06) / 0.964x (0.86–1.07) |
| bar_t5/sub_t5 | 0.967x (0.76–1.26) / 0.964x (0.84–1.07) | 0.982x (0.89–1.05) / 1.041x (0.99–1.14) | 0.970x (0.87–1.07) / 0.966x (0.88–1.03) | 1.034x (0.78–1.07) / 0.951x (0.82–1.11) |
| bar1_i8/sub_i8 | 1.190x (0.94–1.59) / 1.240x (1.00–1.30) | 1.274x (1.05–1.40) / 1.264x (1.11–1.27) | 1.165x (1.05–1.34) / 1.087x (1.07–1.36) | 1.112x (1.00–1.46) / 1.234x (1.10–1.31) |
| bar1_t5/sub_t5 | 1.120x (0.84–1.47) / 1.077x (0.94–1.46) | 1.114x (0.85–1.21) / 1.130x (0.96–1.34) | 1.003x (0.91–1.13) / 1.143x (1.00–1.51) | 1.106x (0.89–1.26) / 1.120x (0.99–1.19) |
| sub_t5/sub_i8 | 1.555x (1.21–2.07) / 1.821x (1.28–1.97) | 1.700x (1.36–2.19) / 1.682x (1.48–1.79) | 2.296x (2.15–2.57) / 2.242x (2.11–2.77) | 2.049x (1.76–2.55) / 2.318x (2.05–2.44) |
| bar_t5/bar_i8 | 1.522x (1.47–1.93) / 1.650x (1.13–1.83) | 1.570x (1.47–1.95) / 1.623x (1.57–1.70) | 2.327x (2.02–2.61) / 2.516x (2.35–2.59) | 2.371x (1.79–2.62) / 2.286x (2.02–2.51) |
| bar1_t5/bar1_i8 | 1.463x (1.10–1.90) / 1.582x (1.47–1.81) | 1.486x (1.33–1.61) / 1.503x (1.35–1.72) | 1.976x (1.61–2.23) / 2.357x (2.05–3.16) | 2.040x (1.56–2.25) / 2.103x (1.87–2.35) |

Empty-work dispatch, µs per layer, median of 7 round medians (60 reps each) [min–max], seed-1 process / seed-2 process:

| scheme | pass A | pass B |
|---|---|---|
| 4 submit/result rounds (int8 / t5vg submit layer) | 493 [407–505] / 492 [471–629] | 486 [469–505] / 493 [485–521] |
| 1 round + 4 barriers (bar1, int8 / t5vg) | 628 [614–638] / 626 [610–795] | 620 [579–638] / 605 [601–648] |
| 1 round + 8 barriers (barrier int8 / t5vg; bar1 tmL) | 1169 [953–1206] / 1198 [1143–1465] | 1147 [1131–1187] / 1138 [1124–1246] |
| 8 submit/result rounds (tmL submit layer) | 1019 [980–1030] / 1005 [968–1191] | 989 [934–1026] / 978 [961–1028] |
| 1 round + 12 barriers (barrier tmL) | 1747 [1704–1787] / 2055 [1664–2154] | 1679 [1588–1793] / 1674 [1659–1759] |
| ratio emp_bar_8/emp_sub_4 | 2.371x / 2.437x | 2.360x / 2.307x |
| ratio emp_bar_12/emp_sub_8 | 1.714x / 2.045x | 1.699x / 1.712x |
| ratio emp_bar_4/emp_sub_4 | 1.273x / 1.274x | 1.274x / 1.226x |
| ratio emp_bar_8/emp_sub_8 | 1.147x / 1.192x | 1.161x / 1.164x |


Marginal cost per sync, from the empty-work rows ((8 − 4 rounds) / 4 and (8 − 4 barriers) / 4), passes A / B, seed-1 / seed-2 process: submit/result round 131.5 / 128.4, 125.6 / 121.1 µs; Barrier(8).wait 135.3 / 142.9, 132.0 / 133.4 µs.

Summary over the 6 paired runs (pass 0 / A / B x seed 1 / 2), median [min–max]: bar / sub int8d B 1 1.064x [0.988–1.079], tmL B 1 0.975x [0.946–1.041], int8g B 4 0.949x [0.861–0.992], t5vg B 4 0.958x [0.933–1.034]. t5 / int8 B 1 submit 1.74x [1.56–1.88] -> barrier 1.64x [1.52–1.79]; B 4 submit 2.27x [2.05–2.32] -> barrier 2.30x [2.13–2.52].

Error (first B of 32 fixed rows vs own float64 reference, seed 1 / seed 2; identical for sub, bar, bar1): tmL B 1 5.57e-07 / 6.18e-07 (cos 0.999999999999991); t5vg B 4 1.23e-06 / 9.86e-07 (cos 0.99999999999997); int8d B 1 3.19e-06 / 3.41e-06 (cos 0.99999999999985); int8g B 4 4.42e-06 / 4.08e-06 (cos 0.99999999999983). bar vs sub and bar1 vs sub: max abs 0.0, cos 1.0, array_equal True, all 8 cases.

Bytes: weights per layer 13,986,304 B t5 vs 69,559,296 B int8 (4.973x), unchanged. Scratch all 8 workers as in 05:45: tmL_T384 50,982,912 B + B 1 table 1,344,276 B; t5vg_T256 70,809,600 B; int8_T128 35,389,440 B. The barrier scheme adds nothing; bar1 adds ≤ ~0.9 MB of worker-local activation copies.

## Pair files this wave

No pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) for this wave reached this lane, so none are linked. The depth-1 clone of JuniorLLM (10:47 MT) shows this lane's own 2a87ba6 (05:55 MT, the 05:45 note) as HEAD on main. No commit newer than that, so no Trit Pack, Handshake, LLC or unattributed commit in the last 6 hours (since ~04:41 MT) touches junior_osai/vault/findings. This lane did not read JuniorHome this wave.

## Next question

Rotate from sync cost to work distribution inside a site. Per-worker load imbalance and idle time on the W 8 decode layer: record per-worker wall start / end and time.thread_time per site for tmL_T384 / int8d_T128 at B 1 and t5vg_T256 / int8g_T128 at B 4 (full bitnetd layer, ctx-free), and report each site's critical path (slowest worker) vs mean worker busy time. Then compare the static ceil(tiles / W) split with dynamic tile claiming: a shared itertools.count / lock-free index that workers pull T-row tiles from until exhausted, and tile sizes T/2 and T/4. Does dynamic claiming cut the slowest-worker time per site by ≥ 1.1x, and does it narrow t5's 1.6–2.3x gap to int8? No earlier note measured per-worker busy / idle time or dynamic tile claiming in the pool.

close: 842550a64615c36b
