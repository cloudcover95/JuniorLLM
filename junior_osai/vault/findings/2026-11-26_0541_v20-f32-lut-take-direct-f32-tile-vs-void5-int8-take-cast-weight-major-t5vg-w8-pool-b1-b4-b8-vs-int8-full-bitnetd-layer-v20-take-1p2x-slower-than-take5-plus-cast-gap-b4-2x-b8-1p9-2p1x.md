---
local: true
date: 2026-11-26
wave: "2026-11-26 05:55 slot (fired 05:41 MT, ran ~05:41–06:00 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (V20, a single np.take from a 243-row f32 LUT viewed as 20-byte void rows that writes the f32 weight tile directly, is bit-identical to the void5 take + int8 cast but SLOWER on the W 8 decode layer, full bitnetd layer, ctx-free: best void5 / best V20 = 0.853–0.859x at B 4 and 0.843–0.924x at B 8 (3 runs), i.e. V20 costs 1.08–1.19x more. The 20-byte take alone is 83–91 worker-ms per layer vs 70–79 ms for take5 (53–60) + cast (15–20); the matmul is unchanged. The (243, 5) f32 axis-0 take (f20g) is the same speed as V20 (0.94–1.05x). t5 / int8 best-vs-best gap stays B 1 1.66–1.76x (tmL vs int8d), B 4 1.96–2.04x, B 8 1.88–2.14x with void5; with V20 it widens to 2.30–2.39x (B 4) and 2.11–2.32x (B 8). No 1.5x speed or memory win beyond the known 4.973x t5 weight-memory win; V20 only saves the 5 B/weight V5 scratch, 70.8 → 56.6 MB over 8 workers)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, v20, void5, f32-lut, np-take, base-3, 1p6-bit, five-trit-pack, weight-major, int8, numpy, not-viable]
---

# V20 f32-LUT take straight into the f32 tile loses to the void5 int8 take + cast: one 20-byte gather is slower than a 5-byte gather plus a vectorised cast, at B 4 and B 8

The 02:47 note ([[2026-11-26_0247_decode-layer-per-op-stage-breakdown-perf-counter-thread-time-t5-tmac-tml-b1-t5vg-void5-expand-b4-vs-int8-cast-dot-mm-w8-pool-full-bitnetd-layer-b4-excess-is-void5-take-b1-is-bld-round-and-critical-path]]) found that at B 4 the void5 np.take LUT expand alone (63–68 worker-ms per layer) accounts for roughly the whole t5 − int8 excess, while t5's cast + matmul is no more than int8's. It asked whether one take from a 243-row f32 LUT viewed as 20-byte void rows ("V20": 5 f32 per packed byte), writing the f32 tile directly and skipping the int8 expand + cast, closes that gap.

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 4 and B 8 (B 1 for reference), does a V20 take (np.take(LUT20V, P5 tile, out=f32 tile viewed as V20) → np.matmul) beat the current void5 take + int8 → f32 cast (t5vg) by at least 1.3x, and does it bring t5 within 1.5x of int8 weights (int8g: cast + weight-major matmul)? How do the take20 vs take5 + cast stage times compare, and what is the scratch cost?

## Method

- Box: 8 CPUs, numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network. The box is shared with other agents, so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1855_1125/gd.py (exec'd without main(); it execs bench0545_1125/xb5.py: same weights, P5 base-3 pack with exact trit round-trip assert, float64 references, dy_T / gd2 schedules), same harness as the 02:47 sp.py. Weight-major t5 kernels at T 256, with three expand variants that share the same f32 tile scratch and np.matmul:
  - **t5vg** (current): np.take(LUT5V 243 × V5, P5 tile, out=V5 scratch) → np.copyto int8 view → f32 tile.
  - **v20g** (V20): np.take(LUT20V 243 × V20, P5 tile, out=f32 tile.view("V20")) — one 20-byte copy per packed byte, no cast. LUT20V = LUT5F (243, 5) f32 viewed as V20 (4,860 B), asserted equal to the trit table.
  - **f20g**: np.take(LUT5F (243, 5) f32, P5 tile, axis=0, out=f32 tile viewed (n, q, 5)) — the plain 2-D f32 table take.
  - Baselines: int8g_T128 (cast + weight-major matmul); at B 1 also int8d_T128 (cast + per-row np.dot, the B 1 int8 best) and t5 tmL_T384 (T-MAC, the B 1 t5 best). gd2 chunk lists for v20g / f20g are t5vg's.
- Timing: plain layer, 7 rounds × 5 reps, rotated order, median of round medians, spread = min–max round median. Stage profile: 15 instrumented layers per config (perf_counter / thread_time around every numpy call, toggled by a flag inside the same kernels), worker-ms summed over 4 sites (qkv, o, gu, down) and 8 workers, medians. Three seeds (XSEED 1, 2, 3).
- Correctness: v20g and f20g outputs are bit-identical (np.array_equal) to t5vg_dy at every B and seed, and instrumented outputs are bit-identical to plain. t5 vs float64 ternary ref: max|err| 8.7e-07–1.04e-06 (B 1, cos 0.999999999999977–9), 9.9e-07–1.23e-06 (B 4 / B 8, cos 0.999999999999972–3). int8 vs float64 int8 ref: 3.2e-06–4.5e-06 (cos ≥ 0.99999999999982). tmL B 1: 5.6–6.2e-07.
- Bytes: weights t5 13,986,304 B vs int8 69,559,296 B (4.973x, unchanged). LUT: V5 1,215 B, V20 4,860 B. Expand scratch over 8 workers at T 256: t5vg 70,809,600 B (V5 + f32 tile), v20g / f20g 56,647,680 B (f32 tile only), int8g 35,389,440 B.

Command:

```
mkdir -p /workspace/bench0555_1126 && cd /workspace/bench0555_1126
OPENBLAS_NUM_THREADS=1 XSEED=1 python3 v20.py run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 XSEED=2 python3 v20.py run2.json > run2.txt
OPENBLAS_NUM_THREADS=1 XSEED=3 python3 v20.py run3.json > run3.txt
python3 agg.py > agg.txt
```

Inline script (v20.py):

```python
#!/usr/bin/env python3
"""V20: f32-LUT take that writes the f32 weight tile directly vs the void5 int8 take + cast, W=8 decode layer, full bitnetd layer, ctx-free, 05:4x MT 11/26.
Base: /workspace/bench1855_1125/gd.py (exec'd without main(); it execs bench0545_1125/xb5.py: weights, P5 base-3 pack + exact trit round-trip
assert, kernels rows_t5vg / rows8g / rows8d / rows_tmL, float64 refs, dy_T / gd2 schedules). Same harness as bench0255_1126/sp.py.
t5 weight-major kernels (tile (n, in) @ x.T (in, B) -> yT), T = 256:
  t5vg : np.take(LUT5V  243 x V5,  P5 tile, out=V5 scratch) -> np.copyto int8 view -> f32 tile -> np.matmul   (current, 2 passes + 5 B/w scratch)
  v20g : np.take(LUT20V 243 x V20, P5 tile, out=f32 tile viewed as V20) -> np.matmul                           (V20: one 20-byte copy per packed byte)
  f20g : np.take(LUT5F (243, 5) f32, P5 tile, axis=0, out=f32 tile viewed (n, q, 5)) -> np.matmul               (same table, strided-axis take)
int8 baselines: 8g_T128 (cast + weight-major matmul), and at B1 8d_T128 (cast + per-row np.dot) and t5 tmL_T384 (T-MAC, B1 best) for reference.
Plain layer timing ROUNDS x REPS rotated, then PROF instrumented layers give per-stage worker time (perf_counter / thread_time) summed over sites.
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v20.py out.json"""
import json, os, sys, time, itertools
import numpy as np
_S = open("/workspace/bench1855_1125/gd.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "gd.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
LUT20V = np.ascontiguousarray(LUT5F).view("V20").ravel()      # 243 x 20-byte rows (4,860 B), same values as L5
assert LUT20V.shape == (243,) and np.array_equal(LUT20V.view(np.float32).reshape(243, 5), L5.astype(np.float32))
PRF = [False]; ACC = [dict() for _ in range(W)]; CUR = ["?"]
def _add(w, st, dw, dc):
    a = ACC[w].get((CUR[0], st))
    if a is None: ACC[w][(CUR[0], st)] = [dw, dc, 1]
    else: a[0] += dw; a[1] += dc; a[2] += 1
def f32tile(w, T, q):
    s1 = scr(("f5", w, T), (T * F5 * 5,), np.float32); return s1, s1[:T * q * 5].reshape(T, q * 5)
def r_t5vg(PR, xT, yT, r0, r1, T, w, i):
    q = PR.shape[1]; sv = scr(("v5", w, T), (T * F5,), "V5")[:T * q].reshape(T, q); _, s = f32tile(w, T, q); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT5V, PR[o0:o0+n], out=sv[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.copyto(s[:n], sv[:n].view(np.int8).reshape(n, q * 5), casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n, :i], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
def r_v20g(PR, xT, yT, r0, r1, T, w, i):
    q = PR.shape[1]; s1, s = f32tile(w, T, q); sv = s1.view("V20")[:T * q].reshape(T, q); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT20V, PR[o0:o0+n], out=sv[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n, :i], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
def r_f20g(PR, xT, yT, r0, r1, T, w, i):
    q = PR.shape[1]; _, s = f32tile(w, T, q); s3 = s.reshape(T, q, 5); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT5F, PR[o0:o0+n], axis=0, out=s3[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n, :i], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
def r_8g(Q, xT, yT, r0, r1, T, w, i):
    _, s = par.scratch(w, T, i // 4); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.copyto(s[:n], Q[o0:o0+n], casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
KF = {"t5vg": r_t5vg, "v20g": r_v20g, "f20g": r_f20g, "8g": r_8g}
_mm_gd = mm
def mm(k, a, kind, T, mode, rec):                       # gd.mm for tmL / 8d; weight-major kinds dispatched to the kernels above
    CUR[0] = k
    if kind in ("tmL", "8d"): return _mm_gd(k, a, kind, T, mode, rec)
    B = a.shape[0]
    if kind == "8g": P = par.F8[k]; R = P["Q"]; i = R.shape[1]
    else: P = P5[k]; R = P["PR"]; i = P["inp"]
    out = R.shape[0]; xT = np.ascontiguousarray(np.asarray(a, np.float32).T); yT = np.empty((out, B), np.float32)
    run_site(KF[kind], R, xT, yT, out, T, i, mode, rec, k, "t5vg" if kind != "8g" else "8g")   # gd2 chunk sizes: v20g/f20g = t5vg's
    y = np.ascontiguousarray(yT.T); y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
G = {"t5vg_dy": ("t5vg", 256, "dy"), "t5vg_gd2": ("t5vg", 256, "gd2"), "v20g_dy": ("v20g", 256, "dy"), "v20g_gd2": ("v20g", 256, "gd2"),
     "f20g_dy": ("f20g", 256, "dy"), "i8g_dy": ("8g", 128, "dy"), "i8g_gd2": ("8g", 128, "gd2")}
CF = {1: dict(G, tmL_dy=("tmL", 384, "dy"), i8d_dy=("8d", 128, "dy"), i8d_gd2=("8d", 128, "gd2")), 4: G, 8: G}
STG = ("take", "cast", "mm", "dot", "idx")
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, time={}, prof={}, err={})
    W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(t5=W5, int8=W8, int8_over_t5=W8 / W5, lut_v5=LUT5V.nbytes, lut_v20=LUT20V.nbytes,
                        scr_t5vg=W * 256 * F5 * (5 + 20), scr_v20g=W * 256 * F5 * 20, scr_i8g=par.scratch_bytes(W, 128))
    print("bytes", res["bytes"], flush=True)
    for B in (1, 4, 8):
        C = CF[B]; names = list(C); xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0] in ("8g", "8d") else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in names:
            if C[c][0] in ("v20g", "f20g", "t5vg"): e[c]["identical_t5vg_dy"] = bool(np.array_equal(outs[c], outs["t5vg_dy"])); e[c]["vs_t5vg_dy_mab"] = float(np.abs(outs[c] - outs["t5vg_dy"]).max())
        res["err"][f"B{B}"] = e; print(f"err B{B}", {c: (f"{v['mab']:.2e}", f"{v['cos']:.15f}", v["instr_identical"], v.get("identical_t5vg_dy")) for c, v in e.items()}, flush=True)
        xb = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32); rm = {c: [] for c in names}
        for c in names: layer(xb, *C[c], [])
        for rd in range(ROUNDS):
            for c in names[rd % len(names):] + names[:rd % len(names)]:
                s = []
                for _ in range(REPS):
                    t0 = pc(); layer(xb, *C[c], []); s.append((pc() - t0) * 1e3)
                rm[c].append(float(np.median(s)))
        o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), rm=v) for c, v in rm.items()}
        rat = {}
        def pr_(a, b, key=None):
            v = [p / q for p, q in zip(rm[a], rm[b])]; rat[key or f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v), a=a, b=b)
        for m in ("dy", "gd2"): pr_(f"t5vg_{m}", f"v20g_{m}"); pr_(f"v20g_{m}", f"i8g_{m}"); pr_(f"t5vg_{m}", f"i8g_{m}")
        pr_("t5vg_dy", "f20g_dy"); pr_("f20g_dy", "v20g_dy")
        i8n = [c for c in names if C[c][0] in ("8g", "8d")]; bi = min(i8n, key=lambda c: o[c]["med"])
        bo = min(("t5vg_dy", "t5vg_gd2"), key=lambda c: o[c]["med"]); bv = min(("v20g_dy", "v20g_gd2", "f20g_dy"), key=lambda c: o[c]["med"])
        bt = min((c for c in names if c not in i8n), key=lambda c: o[c]["med"])
        pr_(bo, bv, "best_void5/best_V20"); pr_(bv, bi, "best_V20/best_i8"); pr_(bo, bi, "best_void5/best_i8"); pr_(bt, bi, "best_t5/best_i8")
        res["time"][f"B{B}"] = dict(cfg=o, ratios=rat)
        for c in names: print(f"B{B} {c:9s} {o[c]['med']:8.3f} ms [{o[c]['lo']:.3f}-{o[c]['hi']:.3f}]", flush=True)
        for k, v in rat.items(): print(f"B{B} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.3f}-{v['hi']:.3f}) [{v['a']}/{v['b']}]", flush=True)
        pr = {}; PRF[0] = True
        for c in [n for n in names if C[n][0] in KF]:
            per = {}
            for _ in range(PROF):
                for d in ACC: d.clear()
                rec = []; layer(xb, *C[c], rec); it = {}
                for (site, st), (dw, dc, n) in ((k, v) for d in ACC for k, v in d.items()):
                    a = it.setdefault(st, [0.0, 0.0]); a[0] += dw; a[1] += dc
                a = it.setdefault("task", [0.0, 0.0]); sp = {}
                for site, w, t0, t1, cpu, n in rec: a[0] += t1 - t0; a[1] += cpu; sp[site] = max(sp.get(site, 0), t1)
                it["span"] = [sum(sp.values()), 0.0]
                for st, v in it.items(): per.setdefault(st, []).append(v)
            pr[c] = {st: dict(wall=float(np.median([v[0] for v in L])) * 1e3, cpu=float(np.median([v[1] for v in L])) * 1e3) for st, L in per.items()}
            print(f"B{B} prof {c:9s} " + "  ".join(f"{st} {v['wall']:.1f}/{v['cpu']:.1f}ms" for st, v in pr[c].items()), flush=True)
        PRF[0] = False; res["prof"][f"B{B}"] = pr
    json.dump(res, open(sys.argv[1], "w"), indent=1)
main()
```

Aggregator (agg.py):

```python
import json
R=[json.load(open(f"/workspace/bench0555_1126/run{i}.json")) for i in (1,2,3)]
print("bytes", R[0]["bytes"])
for B in ("B1","B4","B8"):
  print(f"## {B} plain layer ms median [round min-max]")
  for c in R[0]["time"][B]["cfg"]:
    print(f"| {B[1:]} | {c} | " + " | ".join(f"{r['time'][B]['cfg'][c]['med']:.3f} [{r['time'][B]['cfg'][c]['lo']:.3f}–{r['time'][B]['cfg'][c]['hi']:.3f}]" for r in R)+" |")
  print(f"## {B} ratios")
  for k in R[0]["time"][B]["ratios"]:
    print(f"| {B[1:]} | {k} | " + " | ".join(f"{r['time'][B]['ratios'][k]['r']:.3f}x [{r['time'][B]['ratios'][k]['lo']:.3f}–{r['time'][B]['ratios'][k]['hi']:.3f}] ({r['time'][B]['ratios'][k]['a']}/{r['time'][B]['ratios'][k]['b']})" for r in R)+" |")
  print(f"## {B} stage worker ms wall/cpu")
  for c in R[0]["prof"][B]:
    for i,r in enumerate(R):
      p=r["prof"][B][c]; f=lambda s: f"{p[s]['wall']:.1f} / {p[s]['cpu']:.1f}" if s in p else "–"
      print(f"| {B[1:]} | {c} | {i+1} | {f('take')} | {f('cast')} | {f('mm')} | {f('task')} | {p['span']['wall']:.2f} |")
  print(f"## {B} err"); 
  for c,v in R[0]["err"][B].items(): print(B,c,[ (f"{r['err'][B][c]['mab']:.2e}", f"{r['err'][B][c]['cos']:.15f}", r['err'][B][c]['instr_identical'], r['err'][B][c].get('identical_t5vg_dy')) for r in R])
```

## Results

### Plain layer time (ms, median [round min–max])

| B | config | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | t5vg dy_T | 21.851 [21.119–23.002] | 22.269 [21.452–22.773] | 23.049 [22.653–25.161] |
| 1 | t5vg gd2 | 17.015 [16.141–18.188] | 17.121 [16.208–18.243] | 17.473 [16.931–20.038] |
| 1 | v20g dy_T | 24.584 [23.986–24.976] | 24.011 [23.600–25.347] | 25.403 [24.924–26.188] |
| 1 | v20g gd2 | 18.966 [18.227–19.750] | 18.580 [18.363–19.584] | 20.253 [19.062–22.428] |
| 1 | f20g dy_T | 24.467 [24.094–25.515] | 23.969 [23.766–24.732] | 25.150 [24.022–25.824] |
| 1 | i8 8g dy_T | 12.312 [11.225–14.427] | 12.351 [11.768–12.595] | 13.298 [12.354–14.652] |
| 1 | i8 8g gd2 | 15.021 [14.460–18.263] | 15.669 [15.127–16.013] | 16.123 [15.715–16.488] |
| 1 | t5 tmL dy_T (ref) | 8.934 [8.686–10.709] | 8.784 [8.257–9.793] | 9.233 [8.765–10.072] |
| 1 | i8 8d dy_T (ref) | 5.675 [5.169–6.485] | 5.524 [5.421–6.749] | 5.978 [5.788–6.227] |
| 1 | i8 8d gd2 (ref) | 5.068 [4.989–6.037] | 5.289 [5.079–6.144] | 5.525 [5.281–6.177] |
| 4 | t5vg dy_T | 15.339 [14.411–15.882] | 15.627 [14.700–16.794] | 15.852 [14.623–16.915] |
| 4 | t5vg gd2 | 14.046 [13.559–14.940] | 14.520 [14.183–15.567] | 14.760 [13.497–17.096] |
| 4 | v20g dy_T | 17.722 [17.141–18.214] | 18.486 [17.479–19.091] | 18.942 [17.839–20.583] |
| 4 | v20g gd2 | 16.455 [15.491–18.653] | 17.015 [16.420–18.420] | 17.185 [15.989–19.193] |
| 4 | f20g dy_T | 17.664 [16.582–19.293] | 17.458 [17.010–20.011] | 18.321 [17.198–19.689] |
| 4 | i8 8g dy_T | 6.873 [6.573–6.989] | 7.392 [6.860–8.218] | 7.428 [7.073–7.840] |
| 4 | i8 8g gd2 | 8.487 [8.047–9.360] | 9.404 [8.122–11.001] | 9.147 [8.321–10.427] |
| 8 | t5vg dy_T | 16.549 [15.822–18.769] | 16.651 [15.351–20.190] | 15.061 [14.780–16.763] |
| 8 | t5vg gd2 | 15.667 [14.266–18.436] | 15.660 [14.411–16.720] | 14.305 [13.683–15.472] |
| 8 | v20g dy_T | 18.162 [17.674–22.239] | 18.977 [18.246–20.517] | 18.325 [17.668–18.985] |
| 8 | v20g gd2 | 16.950 [16.280–18.942] | 16.953 [16.069–18.456] | 16.968 [15.992–18.088] |
| 8 | f20g dy_T | 19.144 [17.993–20.728] | 18.472 [17.419–19.804] | 17.843 [17.368–19.951] |
| 8 | i8 8g dy_T | 8.151 [7.416–9.287] | 7.841 [7.433–8.674] | 7.625 [7.432–8.835] |
| 8 | i8 8g gd2 | 8.034 [6.974–9.506] | 7.320 [6.984–8.811] | 7.600 [7.094–9.198] |

### Ratios (median ratio [per-round min–max]; > 1 means the first is slower)

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | t5vg / v20g (dy_T) | 0.889x [0.871–0.931] | 0.927x [0.846–0.965] | 0.907x [0.891–0.961] |
| 1 | best void5 / best V20 (gd2 / gd2) | 0.897x [0.830–0.985] | 0.921x [0.828–0.993] | 0.863x [0.816–0.988] |
| 1 | best t5 / best i8 (tmL dy_T / 8d gd2) | 1.763x [1.637–1.774] | 1.661x [1.594–1.702] | 1.671x [1.522–1.783] |
| 1 | best V20 / best i8 (v20g gd2 / 8d gd2) | 3.742x | 3.513x | 3.666x |
| 4 | t5vg / v20g (dy_T) | 0.866x [0.804–0.926] | 0.845x [0.811–0.894] | 0.837x [0.815–0.850] |
| 4 | best void5 / best V20 (gd2 / gd2) | 0.854x [0.727–0.915] | 0.853x [0.797–0.893] | 0.859x [0.811–0.926] |
| 4 | f20g / v20g (dy_T) | 0.997x [0.950–1.067] | 0.944x [0.891–1.105] | 0.967x [0.957–1.023] |
| 4 | best void5 / best i8 (t5vg gd2 / 8g dy_T) | 2.044x [1.940–2.173] | 1.964x [1.726–2.131] | 1.987x [1.897–2.181] |
| 4 | best V20 / best i8 (v20g gd2 / 8g dy_T) | 2.394x [2.254–2.669] | 2.302x [2.039–2.672] | 2.314x [2.153–2.448] |
| 8 | t5vg / v20g (dy_T) | 0.911x [0.838–0.963] | 0.877x [0.830–0.984] | 0.822x [0.790–0.929] |
| 8 | best void5 / best V20 (gd2 / gd2) | 0.924x [0.842–0.973] | 0.924x [0.788–0.968] | 0.843x [0.809–0.892] |
| 8 | f20g / v20g (dy_T) | 1.054x [0.891–1.090] | 0.973x [0.935–1.012] | 0.974x [0.948–1.103] |
| 8 | best void5 / best i8 (t5vg gd2 / 8g gd2) | 1.950x [1.825–2.170] | 2.139x [1.840–2.228] | 1.882x [1.682–1.959] |
| 8 | best V20 / best i8 (v20g gd2 / 8g gd2) | 2.110x [1.993–2.431] | 2.316x [2.095–2.597] | 2.232x [1.908–2.335] |

At B 1 the weight-major t5vg / v20g paths are only a reference (2x slower than tmL). The V20 vs void5 ordering holds there too.

### Stage worker time per layer, gd2, summed over 4 sites and 8 workers (wall ms / CPU ms; runs 1 / 2 / 3)

| B | config | take | cast | mm | task | Σ site spans (ms) |
|---|---|---|---|---|---|---|
| 4 | t5vg (void5) | 54.5 / 45.2, 57.2 / 46.8, 55.7 / 46.1 | 17.7 / 15.5, 16.4 / 14.5, 15.6 / 14.3 | 22.6, 22.1, 21.9 | 96.6, 96.7, 96.0 | 13.92, 13.95, 13.83 |
| 4 | v20g (V20) | 82.8 / 74.7, 86.5 / 76.2, 85.8 / 75.1 | – | 22.5, 23.2, 23.2 | 106.7, 111.2, 110.0 | 15.42, 16.25, 15.94 |
| 4 | i8 8g (dy_T) | – | 17.6 / 11.4, 21.4 / 15.3, 20.5 / 14.7 | 24.4, 27.7, 24.9 | 42.8, 50.4, 46.0 | 6.40, 7.54, 6.88 |
| 8 | t5vg (void5) | 59.8 / 50.5, 58.0 / 47.5, 55.4 / 45.9 | 19.6 / 18.0, 17.7 / 15.5, 15.9 / 14.4 | 27.2, 25.0, 23.5 | 109.5, 102.5, 95.5 | 16.04, 15.10, 14.01 |
| 8 | v20g (V20) | 90.7 / 83.1, 87.3 / 76.8, 88.3 / 79.2 | – | 26.7, 24.1, 25.5 | 119.3, 113.5, 114.5 | 17.39, 16.55, 16.57 |
| 8 | i8 8g (gd2) | – | 16.9 / 12.6, 16.8 / 12.7, 17.6 / 13.6 | 24.3, 24.0, 25.9 | 42.6, 42.3, 44.3 | 6.39, 6.25, 6.68 |

f20g dy_T take: 82.9–91.3 ms (B 4), 82.5–87.5 ms (B 8), the same as V20. Per-config numbers for dy_T and B 1 are in agg.txt / run1–3.txt beside v20.py.

### Reading

- V20 removes the cast (15.6–19.6 worker-ms) but its take costs 28–33 ms more than the void5 take (82.8–90.7 vs 54.5–59.8 ms). Net is +10 to +19 worker-ms per layer and +1.3 to +2.7 ms of plain layer time at B 4 / B 8 (gd2). Both item sizes (5 and 20 bytes) are presumably outside numpy's fixed-size take copy cases, so both take one generic per-element copy (not verified in numpy source this wave). V20 writes 4x the bytes per gathered item (20 vs 5), and that write traffic costs more than the separate, contiguous SIMD int8 → f32 cast it replaces. V20's take is 88–92% on-CPU (vs 82–84% for take5), so this is memory and copy work, not GIL wait.
- The 2-D f32 axis-0 take (f20g) is the same kernel cost as V20 (0.94–1.05x): the void view saves nothing over the strided take.
- The matmul is identical across expand variants (21.9–27.2 ms) and ≈ int8g's (24.0–27.7 ms). The t5 excess over int8 is still the expand: void5 take + cast 70–79 ms vs int8's cast 17–21 ms.
- Scratch: V20 drops the V5 buffer (70.8 → 56.6 MB over 8 workers, −20%). This is not a speed or weight-memory win.

## Verdict

**not-viable**. V20 is bit-identical but 1.08–1.19x slower than the void5 int8 take + cast at B 4 (best/best 0.853–0.859x) and B 8 (0.843–0.924x), and slower at B 1 too. It widens the t5 / int8 gap instead of closing it: B 4 1.96–2.04x → 2.30–2.39x, B 8 1.88–2.14x → 2.11–2.32x; B 1 (tmL vs int8d) is 1.66–1.76x. There is no 1.5x ternary speed win, and the 4.973x t5 weight-memory win is unchanged and already known.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-26 05:55 wave reached this lane. JuniorLLM HEAD at clone was 92eae7b (cloudcover95, 02:53 MT, "rails/shallowclone: one shallow clone guard", an unattributed rails push after this lane's 77a66b2). No Trit Pack, Handshake or LLC commit appears in the last 6 hours. This note only adds one new file under junior_osai/vault/findings.

## Next question

Both expand takes are presumably on numpy's generic per-item gather (5- or 20-byte items). Does padding the t5 LUT to an 8-byte item (243 × u64: 5 int8 trits + 3 zero bytes) hit numpy's fixed-size take fast path? That would be np.take(LUT_U64, P5 tile) into a u64 scratch, then one strided cast np.copyto(f32 tile (n, q, 5), u64.view(int8).reshape(n, q, 8)[:, :, :5]). Measure whether take8 + strided cast beats take5 + cast by ≥ 1.3x at B 4 and B 8 (stage split, 8 B/w scratch) and how far it moves t5vg toward int8g.

close: 2cc6bc9f29554f76
