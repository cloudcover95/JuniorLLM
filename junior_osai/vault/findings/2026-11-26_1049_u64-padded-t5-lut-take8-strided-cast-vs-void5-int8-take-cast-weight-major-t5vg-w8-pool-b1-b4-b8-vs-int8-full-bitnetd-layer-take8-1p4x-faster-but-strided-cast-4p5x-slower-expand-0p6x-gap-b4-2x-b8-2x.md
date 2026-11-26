---
local: true
date: 2026-11-26
wave: "2026-11-26 10:55 slot (fired 10:49 MT, ran ~10:49–11:05 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (padding the 243-row t5 LUT to 8-byte u64 entries (5 int8 trits + 3 zero bytes) does speed up the gather: on the W 8 decode layer, full bitnetd layer, ctx-free, take8 is 1.29–1.57x faster than the void5 take5 (37–48 vs 55–62 worker-ms per layer at B 4 / B 8). But the strided int8 → f32 cast of the (n, q, 8)[:, :, :5] view costs 74–84 worker-ms vs 16–20 ms for the contiguous cast, so take8 + strided cast is 0.60–0.69x of take5 + cast for the expand stage at B 4 / B 8 (bar was >= 1.3x). Plain layer: best void5 / best u8g = 0.686–0.701x (B 4), 0.687–0.712x (B 8), 0.740–0.773x (B 1), i.e. 1.4x slower; bit-identical to void5. An extra variant (u8p: take8 + contiguous cast of all 8 bytes into a padded f32 tile, matmul against zero-padded x) brings the expand to parity (0.87–1.13x) but doubles the matmul (36–47 vs 22–27 ms), so it is still 0.82–0.90x of void5. t5 / int8 best-vs-best stays B 1 1.53–1.59x (tmL vs int8d), B 4 1.96–2.09x, B 8 1.94–2.02x; with u8g it widens to 2.83–3.00x (B 4) and 2.82–2.83x (B 8). Memory: u64 scratch is 8 B per packed byte vs 5 (expand scratch over 8 workers 70.8 → 79.3 MB, +12%; u8p 113.3 MB, +60%). No 1.5x speed or memory win beyond the known 4.973x t5 weight-memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, lut-expand, u64, void5, np-take, fixed-size-take, strided-cast, base-3, 1p6-bit, five-trit-pack, weight-major, int8, numpy, not-viable]
---

# u64-padded t5 LUT: the 8-byte take is ~1.4x faster than the void5 take, but the strided [:, :, :5] cast is ~4.5x slower than the contiguous cast, so the expand loses (0.6–0.7x) at B 4 and B 8

The 05:41 note ([[2026-11-26_0541_v20-f32-lut-take-direct-f32-tile-vs-void5-int8-take-cast-weight-major-t5vg-w8-pool-b1-b4-b8-vs-int8-full-bitnetd-layer-v20-take-1p2x-slower-than-take5-plus-cast-gap-b4-2x-b8-1p9-2p1x]]) found that a 20-byte V20 take straight into the f32 tile is slower than the 5-byte void5 take plus a contiguous int8 → f32 cast. It guessed that both item sizes miss numpy's fixed-size take copy cases. It asked whether padding the t5 LUT to an 8-byte item (243 × u64: 5 int8 trits + 3 zero bytes) hits a fast take path, followed by one strided cast of the int8 view [:, :, :5] into the f32 tile.

## Question

On the W 8 decode layer (full bitnetd layer, ctx-free), at B 4 and B 8 (B 1 for reference), does a u64 take (np.take(LUT8U 243 × u64, P5 tile, out=u64 scratch)) followed by one strided cast (np.copyto(f32 tile (n, q, 5), u64.view(int8).reshape(n, q, 8)[:, :, :5])) beat the current void5 take + contiguous int8 cast (t5vg) by at least 1.3x for the expand stage? Does it move t5 toward int8 weights (int8g: cast + weight-major matmul)? What does the 8 B/weight u64 scratch cost compared with 5 B?

## Method

- Box: 8 CPUs, numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network. The box is shared with other agents, so ratios are taken within a run (same rounds, rotated order).
- Base: /workspace/bench1855_1125/gd.py (exec'd without main(); it execs bench0545_1125/xb5.py: same weights, P5 base-3 pack with exact trit round-trip assert, float64 references, dy_T / gd2 schedules), same harness as the 05:41 v20.py. The weight-major t5 kernels run at T 256 and share the same np.matmul. The expand variants are:
  - **t5vg** (current baseline): np.take(LUT5V 243 × V5, P5 tile, out=V5 scratch) → np.copyto of the contiguous int8 view into the f32 tile.
  - **u8g** (this question): np.take(LUT8U 243 × u64, P5 tile, out=u64 scratch) → np.copyto(f32 tile viewed (n, q, 5), u64 scratch viewed int8 (n, q, 8)[:, :, :5]), a single strided cast. LUT8U (1,944 B) is asserted equal to the trit table with zero pad bytes.
  - **u8p** (extra, not asked): the same u64 take → contiguous cast of all 8 bytes into a padded f32 tile (n, q·8) → np.matmul against x.T zero-padded to (q·8, B) (zero rows at the 3 pad slots per group and past the input width). This removes the strided cast but makes the matmul K 1.6x longer.
  - Baselines: int8g_T128 (cast + weight-major matmul). At B 1 also int8d_T128 (cast + per-row np.dot, the B 1 int8 best) and t5 tmL_T384 (T-MAC, the B 1 t5 best). The gd2 chunk lists for u8g / u8p are t5vg's.
- Timing: plain layer, 7 rounds × 5 reps, rotated order, median of round medians, spread = min–max round median. Stage profile: 15 instrumented layers per config (perf_counter / thread_time around every numpy call, toggled by a flag inside the same kernels), worker-ms summed over 4 sites (qkv, o, gu, down) and 8 workers, medians. Three seeds (XSEED 1, 2, 3).
- Correctness: u8g outputs are bit-identical (np.array_equal) to t5vg_dy at every B and seed, and instrumented outputs are bit-identical to plain. u8p is not bit-identical, because the matmul K order and length change (the zero terms alter the BLAS summation), but it is just as accurate: max|err| vs the float64 ternary ref is 9.4e-07–1.28e-06 (cos ≥ 0.999999999999965). t5 vs float64 ternary ref: 8.7e-07–1.04e-06 (B 1, cos 0.999999999999977–9), 9.9e-07–1.23e-06 (B 4 / B 8, cos 0.999999999999972–3). int8 vs float64 int8 ref: 3.2e-06–4.5e-06 (cos ≥ 0.99999999999982). tmL B 1: 5.6–6.2e-07.
- Bytes: weights t5 13,986,304 B vs int8 69,559,296 B (4.973x, unchanged). LUT: V5 1,215 B, u64 1,944 B. Gather scratch per packed byte: 5 B (V5) vs 8 B (u64). Expand scratch over 8 workers at T 256: t5vg 70,809,600 B (V5 + f32 tile), u8g 79,306,752 B (u64 + f32 tile, +12.0%), u8p 113,295,360 B (u64 + padded f32 tile, +60.0%), int8g 35,389,440 B.

Command:

```
mkdir -p /workspace/bench1055_1126 && cd /workspace/bench1055_1126
OPENBLAS_NUM_THREADS=1 XSEED=1 python3 v8.py run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 XSEED=2 python3 v8.py run2.json > run2.txt
OPENBLAS_NUM_THREADS=1 XSEED=3 python3 v8.py run3.json > run3.txt
python3 agg.py > agg.txt
```

Inline script (v8.py):

```python
#!/usr/bin/env python3
"""V8: 8-byte (u64) padded t5 LUT take + strided int8 -> f32 cast vs the void5 int8 take + cast, W=8 decode layer, full bitnetd layer, ctx-free, 10:5x MT 11/26.
Base: /workspace/bench1855_1125/gd.py (exec'd without main(); it execs bench0545_1125/xb5.py: weights, P5 base-3 pack + exact trit round-trip
assert, kernels, float64 refs, dy_T / gd2 schedules). Same harness as bench0555_1126/v20.py.
t5 weight-major kernels (tile (n, in) @ x.T (in, B) -> yT), T = 256:
  t5vg : np.take(LUT5V 243 x V5, P5 tile, out=V5 scratch) -> np.copyto int8 view -> f32 tile -> np.matmul       (current baseline)
  u8g  : np.take(LUT8U 243 x u64 (5 trits + 3 zero bytes), P5 tile, out=u64 scratch) -> np.copyto(f32 tile (n,q,5), u64.view(int8)(n,q,8)[:,:,:5]) -> np.matmul
  u8p  : same u64 take -> contiguous cast of all 8 bytes into a padded f32 tile (n, q*8) -> np.matmul with zero-padded x.T (q*8, B)   (extra: no strided cast, 1.6x matmul K)
int8 baselines: 8g_T128 (cast + weight-major matmul), and at B1 8d_T128 (cast + per-row np.dot) and t5 tmL_T384 (T-MAC, B1 best) for reference.
Plain layer timing ROUNDS x REPS rotated, then PROF instrumented layers give per-stage worker time (perf_counter / thread_time) summed over sites.
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 v8.py out.json"""
import json, os, sys, time, itertools
import numpy as np
_S = open("/workspace/bench1855_1125/gd.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "gd.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 15
_L8 = np.zeros((243, 8), np.int8); _L8[:, :5] = L5
LUT8U = _L8.view(np.uint64).ravel()                              # 243 x 8-byte rows (1,944 B): 5 int8 trits + 3 zero pad bytes
assert LUT8U.shape == (243,) and np.array_equal(LUT8U.view(np.int8).reshape(243, 8)[:, :5], L5) and not LUT8U.view(np.int8).reshape(243, 8)[:, 5:].any()
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
def r_u8g(PR, xT, yT, r0, r1, T, w, i):
    q = PR.shape[1]; su = scr(("u8", w, T), (T * F5,), np.uint64)[:T * q].reshape(T, q); _, s = f32tile(w, T, q); s3 = s.reshape(T, q, 5)
    s8 = su.view(np.int8).reshape(T, q, 8); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT8U, PR[o0:o0+n], out=su[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.copyto(s3[:n], s8[:n, :, :5], casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n, :i], xT, out=yT[o0:o0+n])
        if P: _add(w, "mm", pc() - t0, tt() - c0)
def r_u8p(PR, xT8, yT, r0, r1, T, w, i):                # xT8: (q*8, B) with zero rows at pad slots and past i
    q = PR.shape[1]; su = scr(("u8", w, T), (T * F5,), np.uint64)[:T * q].reshape(T, q)
    s = scr(("f8", w, T), (T * F5 * 8,), np.float32)[:T * q * 8].reshape(T, q * 8); s8 = su.view(np.int8).reshape(T, q * 8); P = PRF[0]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        if P: t0 = pc(); c0 = tt()
        np.take(LUT8U, PR[o0:o0+n], out=su[:n])
        if P: _add(w, "take", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.copyto(s[:n], s8[:n], casting="unsafe")
        if P: _add(w, "cast", pc() - t0, tt() - c0); t0 = pc(); c0 = tt()
        np.matmul(s[:n], xT8, out=yT[o0:o0+n])
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
KF = {"t5vg": r_t5vg, "u8g": r_u8g, "u8p": r_u8p, "8g": r_8g}
_mm_gd = mm
def mm(k, a, kind, T, mode, rec):                       # gd.mm for tmL / 8d; weight-major kinds dispatched to the kernels above
    CUR[0] = k
    if kind in ("tmL", "8d"): return _mm_gd(k, a, kind, T, mode, rec)
    B = a.shape[0]
    if kind == "8g": P = par.F8[k]; R = P["Q"]; i = R.shape[1]
    else: P = P5[k]; R = P["PR"]; i = P["inp"]
    out = R.shape[0]; xT = np.ascontiguousarray(np.asarray(a, np.float32).T); yT = np.empty((out, B), np.float32)
    if kind == "u8p":
        q = R.shape[1]; t = np.zeros((q * 5, B), np.float32); t[:i] = xT; x8 = np.zeros((q, 8, B), np.float32); x8[:, :5] = t.reshape(q, 5, B); xT = x8.reshape(q * 8, B)
    run_site(KF[kind], R, xT, yT, out, T, i, mode, rec, k, "t5vg" if kind != "8g" else "8g")   # gd2 chunk sizes: u8g/u8p = t5vg's
    y = np.ascontiguousarray(yT.T); y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
G = {"t5vg_dy": ("t5vg", 256, "dy"), "t5vg_gd2": ("t5vg", 256, "gd2"), "u8g_dy": ("u8g", 256, "dy"), "u8g_gd2": ("u8g", 256, "gd2"),
     "u8p_gd2": ("u8p", 256, "gd2"), "i8g_dy": ("8g", 128, "dy"), "i8g_gd2": ("8g", 128, "gd2")}
CF = {1: dict(G, tmL_dy=("tmL", 384, "dy"), i8d_dy=("8d", 128, "dy"), i8d_gd2=("8d", 128, "gd2")), 4: G, 8: G}
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, time={}, prof={}, err={})
    W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
    res["bytes"] = dict(t5=W5, int8=W8, int8_over_t5=W8 / W5, lut_v5=LUT5V.nbytes, lut_u8=LUT8U.nbytes,
                        scr_t5vg=W * 256 * F5 * (5 + 20), scr_u8g=W * 256 * F5 * (8 + 20), scr_u8p=W * 256 * F5 * (8 + 32), scr_i8g=par.scratch_bytes(W, 128))
    print("bytes", res["bytes"], flush=True)
    for B in (1, 4, 8):
        C = CF[B]; names = list(C); xr = XREF[:B]; e = {}; outs = {}
        for c in names:
            PRF[0] = False; y0 = layer(xr, *C[c], []); PRF[0] = True; y1 = layer(xr, *C[c], []); PRF[0] = False; outs[c] = y0
            r = REF[8 if C[c][0] in ("8g", "8d") else 2][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        for c in names:
            if C[c][0] in ("u8g", "u8p", "t5vg"): e[c]["identical_t5vg_dy"] = bool(np.array_equal(outs[c], outs["t5vg_dy"])); e[c]["vs_t5vg_dy_mab"] = float(np.abs(outs[c] - outs["t5vg_dy"]).max())
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
        for m in ("dy", "gd2"): pr_(f"t5vg_{m}", f"u8g_{m}"); pr_(f"u8g_{m}", f"i8g_{m}"); pr_(f"t5vg_{m}", f"i8g_{m}")
        pr_("t5vg_gd2", "u8p_gd2"); pr_("u8p_gd2", "u8g_gd2")
        i8n = [c for c in names if C[c][0] in ("8g", "8d")]; bi = min(i8n, key=lambda c: o[c]["med"])
        bo = min(("t5vg_dy", "t5vg_gd2"), key=lambda c: o[c]["med"]); bv = min(("u8g_dy", "u8g_gd2"), key=lambda c: o[c]["med"])
        bt = min((c for c in names if c not in i8n), key=lambda c: o[c]["med"])
        pr_(bo, bv, "best_void5/best_u8"); pr_(bv, bi, "best_u8/best_i8"); pr_(bo, bi, "best_void5/best_i8"); pr_(bt, bi, "best_t5/best_i8")
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
R=[json.load(open(f"/workspace/bench1055_1126/run{i}.json")) for i in (1,2,3)]
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
| 1 | t5vg dy_T | 23.456 [21.989–27.162] | 23.225 [21.854–24.158] | 22.964 [21.958–23.881] |
| 1 | t5vg gd2 | 18.406 [17.193–19.746] | 18.070 [17.323–19.090] | 18.009 [17.075–20.360] |
| 1 | u8g dy_T | 31.151 [30.700–32.305] | 30.234 [29.972–32.464] | 29.773 [28.936–31.264] |
| 1 | u8g gd2 | 24.136 [22.283–25.612] | 23.374 [22.106–24.978] | 24.341 [22.452–26.962] |
| 1 | u8p gd2 (padded tile) | 24.795 [22.606–27.025] | 24.741 [23.003–26.173] | 23.901 [22.278–24.815] |
| 1 | i8 8g dy_T | 13.263 [12.680–16.528] | 13.327 [12.013–13.813] | 12.735 [12.499–13.647] |
| 1 | i8 8g gd2 | 16.236 [14.815–17.390] | 15.880 [14.555–16.516] | 14.875 [14.183–15.288] |
| 1 | t5 tmL dy_T (ref) | 8.880 [8.326–12.105] | 8.984 [8.530–10.990] | 9.657 [9.207–18.309] |
| 1 | i8 8d dy_T (ref) | 5.789 [5.632–7.761] | 6.024 [5.688–7.528] | 6.599 [5.955–10.051] |
| 1 | i8 8d gd2 (ref) | 5.840 [5.153–7.365] | 5.659 [5.120–8.297] | 6.174 [5.713–10.090] |
| 4 | t5vg dy_T | 16.333 [16.024–18.081] | 15.520 [14.557–15.896] | 16.641 [15.251–17.220] |
| 4 | t5vg gd2 | 15.829 [14.777–17.265] | 14.471 [14.224–16.136] | 14.914 [14.640–16.476] |
| 4 | u8g dy_T | 24.338 [23.027–25.459] | 22.870 [21.967–23.964] | 22.867 [22.469–23.689] |
| 4 | u8g gd2 | 22.795 [21.554–23.751] | 20.656 [20.270–21.577] | 21.735 [20.865–24.409] |
| 4 | u8p gd2 (padded tile) | 17.624 [17.232–18.149] | 16.490 [16.021–18.664] | 17.179 [15.797–19.391] |
| 4 | i8 8g dy_T | 7.589 [7.131–7.710] | 7.295 [6.881–8.166] | 7.622 [7.184–8.220] |
| 4 | i8 8g gd2 | 9.586 [8.915–10.890] | 9.032 [8.646–9.721] | 9.146 [8.886–9.867] |
| 8 | t5vg dy_T | 15.895 [15.575–16.988] | 15.342 [14.780–16.854] | 16.911 [16.074–18.566] |
| 8 | t5vg gd2 | 15.715 [15.256–16.625] | 14.950 [13.803–17.415] | 15.589 [15.075–17.696] |
| 8 | u8g dy_T | 23.436 [22.821–24.135] | 22.609 [22.113–24.775] | 23.931 [22.820–25.585] |
| 8 | u8g gd2 | 22.067 [20.924–22.701] | 21.117 [20.114–25.050] | 22.692 [20.922–25.433] |
| 8 | u8p gd2 (padded tile) | 18.423 [17.822–20.509] | 17.326 [15.914–21.734] | 19.033 [17.140–22.864] |
| 8 | i8 8g dy_T | 7.793 [7.533–8.216] | 7.485 [7.198–8.344] | 8.107 [7.585–9.052] |
| 8 | i8 8g gd2 | 7.891 [7.455–8.285] | 7.599 [7.101–8.869] | 8.044 [7.131–8.938] |

### Ratios (median ratio [per-round min–max]; > 1 means the first is slower)

| B | ratio | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | t5vg_dy / u8g_dy (t5vg_dy / u8g_dy) | 0.753x [0.706–0.878] | 0.768x [0.688–0.802] | 0.771x [0.725–0.801] |
| 1 | best_void5 / best_u8 (t5vg_gd2 / u8g_gd2) | 0.763x [0.700–0.850] | 0.773x [0.719–0.854] | 0.740x [0.679–0.907] |
| 1 | t5vg_gd2 / u8p_gd2 (t5vg_gd2 / u8p_gd2) | 0.742x [0.715–0.838] | 0.730x [0.683–0.820] | 0.754x [0.688–0.871] |
| 1 | u8p_gd2 / u8g_gd2 (u8p_gd2 / u8g_gd2) | 1.027x [0.946–1.055] | 1.058x [0.971–1.120] | 0.982x [0.894–1.041] |
| 1 | best_u8 / best_i8 (u8g_gd2 / i8d_dy) | 4.170x [3.133–4.306] | 4.131x [2.764–4.726] | 3.943x [2.672–4.291] |
| 1 | best_t5 / best_i8 (tmL_dy / i8d_dy) | 1.534x [1.444–1.560] | 1.588x [1.301–1.901] | 1.564x [1.411–1.815] |
| 4 | t5vg_dy / u8g_dy (t5vg_dy / u8g_dy) | 0.671x [0.649–0.730] | 0.679x [0.628–0.716] | 0.728x [0.677–0.735] |
| 4 | best_void5 / best_u8 (t5vg_gd2 / u8g_gd2) | 0.694x [0.654–0.786] | 0.701x [0.665–0.787] | 0.686x [0.657–0.740] |
| 4 | t5vg_gd2 / u8p_gd2 (t5vg_gd2 / u8p_gd2) | 0.898x [0.836–1.002] | 0.878x [0.788–0.976] | 0.868x [0.799–0.934] |
| 4 | u8p_gd2 / u8g_gd2 (u8p_gd2 / u8g_gd2) | 0.773x [0.743–0.820] | 0.798x [0.742–0.920] | 0.790x [0.754–0.859] |
| 4 | best_void5 / best_i8 (t5vg_gd2 / i8g_dy) | 2.086x [1.950–2.239] | 1.984x [1.742–2.320] | 1.957x [1.878–2.076] |
| 4 | best_u8 / best_i8 (u8g_gd2 / i8g_dy) | 3.004x [2.840–3.197] | 2.832x [2.482–3.119] | 2.851x [2.709–3.024] |
| 4 | best_t5 / best_i8 (t5vg_gd2 / i8g_dy) | 2.086x [1.950–2.239] | 1.984x [1.742–2.320] | 1.957x [1.878–2.076] |
| 8 | t5vg_dy / u8g_dy (t5vg_dy / u8g_dy) | 0.678x [0.646–0.707] | 0.679x [0.643–0.745] | 0.707x [0.649–0.814] |
| 8 | best_void5 / best_u8 (t5vg_gd2 / u8g_gd2) | 0.712x [0.691–0.757] | 0.708x [0.635–0.755] | 0.687x [0.598–0.835] |
| 8 | t5vg_gd2 / u8p_gd2 (t5vg_gd2 / u8p_gd2) | 0.853x [0.768–0.882] | 0.863x [0.790–0.898] | 0.819x [0.659–0.982] |
| 8 | u8p_gd2 / u8g_gd2 (u8p_gd2 / u8g_gd2) | 0.835x [0.808–0.903] | 0.821x [0.774–0.868] | 0.839x [0.748–0.969] |
| 8 | best_void5 / best_i8 (t5vg_gd2 / i8g_dy) | 2.017x [1.913–2.142] | 1.997x [1.837–2.193] | 1.938x [1.766–2.355] |
| 8 | best_u8 / best_i8 (u8g_gd2 / i8g_dy) | 2.832x [2.684–2.916] | 2.821x [2.716–3.154] | 2.821x [2.539–3.566] |
| 8 | best_t5 / best_i8 (t5vg_gd2 / i8g_dy) | 2.017x [1.913–2.142] | 1.997x [1.837–2.193] | 1.938x [1.766–2.355] |

At B 1 the weight-major paths are only a reference: they are about 2x slower than tmL. The u8g vs void5 ordering holds there too.

### Stage worker time per layer, gd2, summed over 4 sites and 8 workers (wall ms / CPU ms; runs 1 / 2 / 3)

| B | config | take | cast | take + cast | mm | task | Σ site spans (ms) |
|---|---|---|---|---|---|---|---|
| 4 | t5vg gd2 | 58.3 / 48.8, 54.8 / 45.0, 60.7 / 46.1 | 19.6 / 18.5, 15.7 / 14.2, 19.5 / 16.2 | 77.8, 70.5, 80.2 | 23.6, 22.2, 24.5 | 104.0, 94.4, 105.9 | 15.09, 13.70, 15.67 |
| 4 | u8g gd2 | 43.7 / 32.2, 38.4 / 28.4, 40.7 / 29.4 | 83.6 / 81.9, 75.7 / 73.9, 76.6 / 74.7 | 127.2, 114.1, 117.3 | 26.3, 23.4, 26.2 | 159.6, 138.9, 147.0 | 22.30, 19.98, 20.95 |
| 4 | u8p gd2 (padded tile) | 47.3 / 36.6, 37.7 / 28.3, 45.2 / 31.1 | 41.8 / 39.9, 27.2 / 25.8, 38.0 / 35.5 | 89.1, 64.8, 83.2 | 47.1, 35.8, 41.6 | 138.6, 102.8, 126.2 | 19.92, 14.99, 18.48 |
| 4 | i8 8g gd2 | – | 33.7 / 17.7, 25.1 / 12.7, 28.2 / 13.8 | 33.7, 25.1, 28.2 | 28.9, 23.3, 25.3 | 64.9, 49.6, 55.9 | 10.20, 7.85, 8.70 |
| 8 | t5vg gd2 | 56.9 / 47.8, 58.3 / 48.8, 61.6 / 48.9 | 16.5 / 15.1, 18.8 / 16.9, 18.7 / 15.3 | 73.5, 77.1, 80.2 | 24.6, 26.5, 25.7 | 100.0, 105.4, 106.2 | 14.42, 15.21, 15.49 |
| 8 | u8g gd2 | 39.3 / 29.7, 37.1 / 28.2, 47.6 / 29.6 | 77.8 / 75.7, 74.4 / 73.1, 81.2 / 75.9 | 117.1, 111.5, 128.8 | 25.3, 24.0, 26.7 | 143.8, 137.9, 153.1 | 20.66, 19.78, 22.30 |
| 8 | u8p gd2 (padded tile) | 40.6 / 30.6, 37.8 / 28.6, 44.9 / 30.8 | 32.1 / 30.1, 30.5 / 28.4, 36.4 / 32.6 | 72.7, 68.3, 81.3 | 42.0, 38.1, 43.0 | 116.2, 110.0, 124.8 | 17.19, 15.63, 18.20 |
| 8 | i8 8g gd2 | – | 21.9 / 17.8, 17.4 / 13.0, 21.5 / 14.8 | 21.9, 17.4, 21.5 | 30.5, 24.6, 30.4 | 53.6, 43.2, 54.7 | 8.22, 6.59, 8.11 |

Expand-stage ratio (t5vg take + cast) / (u8g take + cast), gd2: B 4 0.612, 0.618, 0.683; B 8 0.627, 0.691, 0.623 (dy_T: B 4 0.603–0.654, B 8 0.613–0.642; B 1 0.71–0.78). The take alone: t5vg take5 / u8g take8 = 1.33–1.49x (B 4), 1.29–1.57x (B 8). u8p expand vs t5vg: B 4 0.874, 1.087, 0.963; B 8 1.010, 1.128, 0.988. Per-config numbers for dy_T and B 1 are in agg.txt / run1–3.txt beside v8.py.

### Reading

- The fixed-size take helps. The 8-byte gather cuts take worker time from 55–62 ms to 37–48 ms (1.3–1.6x), and its CPU share falls (28–33 ms CPU vs 45–49 ms). This fits a fixed 8-byte element copy replacing the generic per-item 5-byte copy (consistent with the 05:41 guess; numpy source not checked this wave). A standalone single-thread micro-check outside the layer, (256 × 512) tile with uint8 indices, did not show the win (take8 674 µs vs take5 390 µs). The in-layer W 8 numbers are the ones the verdict uses.
- The strided cast kills it. Casting the (n, q, 8)[:, :, :5] int8 view into the (n, q, 5) f32 tile costs 74–84 worker-ms, 98% on-CPU, vs 16–20 ms for the contiguous int8 → f32 cast (about 4.5x). numpy can't fuse the inner dimension of 5 out of a stride-8 source into one SIMD loop, so it runs a short 5-element inner loop for every packed byte. Net expand cost is +37 to +50 worker-ms per layer, and the plain layer is 1.4x slower.
- Padding instead of slicing (u8p) shows where the cost is. The contiguous 8-byte cast is 27–42 ms, so take8 + contiguous cast ≈ take5 + cast (0.87–1.13x). But that pushes 1.6x more K through the matmul (36–47 vs 22–27 ms), and u8p ends at 0.82–0.90x of t5vg. The pad bytes either cost a strided cast or extra matmul work. Neither route recovers the expand.
- Memory: u8g's u64 scratch is 8 B per packed byte instead of 5 (+60% for the gather buffer, +12% for the whole expand scratch with the f32 tile: 70.8 → 79.3 MB over 8 workers). u8p adds the padded f32 tile (113.3 MB). Weights are unchanged (13.99 MB t5, 4.973x below int8). Nothing here is a memory win.

## Verdict

**not-viable**. The 8-byte take is a real 1.3–1.6x gather win, but the strided [:, :, :5] cast it forces is ~4.5x the contiguous cast. Expand stage take8 + strided cast vs take5 + cast is 0.60–0.69x at B 4 / B 8, far below the 1.3x bar. Plain layer best void5 / best u8g is 0.686–0.701x (B 4) and 0.687–0.712x (B 8). Outputs are bit-identical. The padded-tile alternative (u8p) only reaches expand parity and loses on matmul (0.82–0.90x). t5 / int8 best-vs-best: B 1 1.53–1.59x (tmL vs int8d), B 4 1.96–2.09x, B 8 1.94–2.02x with void5; with u8g it widens to 2.83–3.00x and 2.82–2.83x. Scratch grows 8 vs 5 B per packed byte (+12% expand scratch). There is no 1.5x ternary speed win, and the 4.973x t5 weight-memory win is unchanged and already known.

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-26 10:55 wave reached this lane. JuniorLLM HEAD at clone was 1408541 (cloudcover95, 05:46 MT, this lane's 05:41 V20 note). Nothing landed after it, so no Trit Pack, Handshake or LLC commit falls in the last 6 hours (the shallow clone shows only HEAD; the 05:41 note found none in the window before it). This note only adds one new file under junior_osai/vault/findings.

## Next question

The take side is now solved (take8 is 1.3–1.6x faster than take5); the loss is all in turning 8-byte items into 5 f32 columns. Can the 3 pad bytes carry real weights instead of zeros? Keep the u64 take, but use a pair layout: a uint16 index over two packed bytes, a 59,049-row table of 10 trits + 6 pad (V16, 16-byte fixed path, 945 KB). Or more simply, an 8-trit row (3^8 = 6,561 rows × 8 int8 = 52 KB) indexed by a uint16 built from the P5 stream at pack time (2 bits/weight, 4.0x vs int8 instead of 4.97x). Then the take output is exactly the contiguous int8 tile, and the cast stays the cheap contiguous one. Measure take(6561 × u64, u16 idx) + contiguous cast vs take5 + cast (≥ 1.3x at B 4 / B 8) and the weight-memory trade (4.97x → 4.0x vs int8).

close: 922367190c7b4b92
