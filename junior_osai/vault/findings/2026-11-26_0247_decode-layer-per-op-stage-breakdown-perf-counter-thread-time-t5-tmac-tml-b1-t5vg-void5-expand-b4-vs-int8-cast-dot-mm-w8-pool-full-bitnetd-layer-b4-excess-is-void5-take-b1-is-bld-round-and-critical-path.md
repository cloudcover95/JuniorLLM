---
local: true
date: 2026-11-26
wave: "2026-11-26 02:55 slot (fired 02:47 MT, ran ~02:47–03:10 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (diagnostic split of the t5 kernels on the W 8 decode layer, full bitnetd layer, ctx-free. B 4: the void5 np.take LUT expand alone costs 63–68 worker-ms per layer (55–58% of t5 task time, 72–77% on-CPU), which is more than the whole t5 − int8 worker-time excess (50–69 ms); t5's own cast + matmul (37–47 ms) is no more than int8's cast + matmul (46–64 ms). B 1: t5 tmL spends no more worker time than int8d (52–60 vs 48–64 ms incl. the table build) but loses on wall: the static table-build round adds 1.29–1.68 ms of serial span per layer (70–75% off-CPU), and idx + take (43–61 ms, 25–39% off-CPU) stretch the site spans. Plain best-vs-best gap: B 1 1.52–1.54x, B 4 1.78–1.99x. No 1.5x speed or memory win for ternary beyond the known 4.973x t5 weight-memory win)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, thread-pool, per-op-breakdown, stage-timing, thread-time, off-cpu, gil, base-3, 1p6-bit, five-trit-pack, tmac, lut-expand, void5, int8, numpy, not-viable]
---

# Per-op stage breakdown on the W 8 decode layer: at B 4 the t5 excess is the void5 LUT take, at B 1 it is the table-build round and a longer on-CPU critical path, not extra worker time

The 18:43 note ([[2026-11-25_1843_decode-layer-guided-self-scheduling-multi-level-chunk-lists-tb-t2-t4-w8-pool-b1-b4-t5-tmac-expand-vs-int8-full-bitnetd-layer-gd2-b4-only-1p15x-int8-none-gap-1p5-2p2x]]) found that scheduling buys t5 at most 1.08–1.15x and that the 1.54x (B 1) / 2.21x (B 4) gap to int8 sits inside the kernels. It asked which numpy stage holds the t5 excess, in µs per site and as a share of the gap, and whether that stage is on-CPU (compute or memory) or off-CPU (GIL wait).

## Question

For tmL_T384 at B 1 and t5vg_T256 at B 4 in their best schedules (dy_T and gd2), time every numpy call per chunk on the W 8 pool with perf_counter (wall) and thread_time (CPU). Stages are tmL np.add index build / np.take gather / np.dot reduce plus the static table-build round, and t5vg V5 np.take expand / int8 → f32 np.copyto / np.matmul. Compare them with int8d (cast + per-row np.dot, B 1) and int8g (cast + weight-major matmul, B 4) on the same rows and schedules.

## Method

- Box: 8 CPUs, numpy 2.5.4, OPENBLAS_NUM_THREADS=1, W 8 thread pool, nothing bound, no network. The box is shared with other agents, so absolute ms drift between runs (run 2 B 1 is about 1.25x slower across all paths), which is why ratios are taken within a run.
- Base: /workspace/bench1855_1125/gd.py (exec'd without main(); it execs bench0545_1125/xb5.py: same weights, P5 base-3 pack with exact trit round-trip assert, float64 references). Kernels are re-bound to instrumented copies that wrap each numpy call in perf_counter + thread_time and add into a per-worker (site, stage) accumulator. "task" is the whole worker task. "other" = task − Σ stages (Python loop, claim counter, slicing). "bld" is tmL's table-build round, recorded as its own submit round.
- Configs: B 1 t5 = tmL_T384, i8 = 8d_T128; B 4 t5 = t5vg_T256, i8 = 8g_T128; each in dy_T and gd2. Plain (uninstrumented) layer timing: 7 rounds × 5 reps, rotated order, median of round medians, spread = min–max round median. Profile: 25 instrumented layers per config, per-site per-stage medians, summed over the 4 sites (qkv, o, gu, down). Two seeds (XSEED 1, 2).
- Correctness: instrumented outputs are bit-identical to the plain kernels for all 8 configs. t5 vs float64 ternary ref: B 1 max|err| 5.57e-07 (cos 0.999999999999991), B 4 1.23e-06. int8 vs float64 int8 ref: 3.19e-06 / 3.67–4.42e-06.

Command:

```
mkdir -p /workspace/bench0255_1126 && cd /workspace/bench0255_1126
OPENBLAS_NUM_THREADS=1 XSEED=1 python3 sp.py run1.json > run1.txt
OPENBLAS_NUM_THREADS=1 XSEED=2 python3 sp.py run2.json > run2.txt
python3 agg.py > agg.txt
```

Inline script (sp.py):

```python
#!/usr/bin/env python3
"""Per-op stage breakdown of the t5 kernels vs int8 inside the best W=8 decode schedules, full bitnetd layer, ctx-free, 02:5x MT 11/26.
Base: /workspace/bench1855_1125/gd.py (exec'd without main(); it execs bench0545_1125/xb5.py: weights, P5 base-3 pack, kernels, float64 refs).
Kernels are re-bound to instrumented copies that wrap every numpy call with perf_counter + thread_time and add the deltas into a
per-worker accumulator keyed (site, stage). Stages:
  tmL  (B1 t5, T384): idx = np.add(P5 tile, g*243) | take = np.take(tab_b, idx) x B | dot = np.dot(., ones) x B ; plus the static table-build round (bld)
  t5vg (B4 t5, T256): take = np.take(LUT5V, P5 tile) (void5 expand) | cast = np.copyto int8 -> f32 | mm = np.matmul(tile, xT)
  8d   (B1 int8, T128): cast = np.copyto int8 -> f32 | dot = np.dot(tile, x_b) x B
  8g   (B4 int8, T128): cast | mm
  task = whole task wall/cpu per worker; other = task - sum(stages) (Python loop, claim counter, slicing).
Schedules: dy_T (shared counter over T tiles) and gd2 (two-level guided list), both kinds, B1 and B4.
Uninstrumented layer medians (ROUNDS x REPS, rotated order) are timed first; then PROF instrumented layers give per-stage medians.
Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 sp.py out.json"""
import json, os, sys, time, itertools
import numpy as np
_S = open("/workspace/bench1855_1125/gd.py").read(); _S = _S[:_S.rstrip().rfind("main()")]
_argv = sys.argv; exec(compile(_S, "gd.py", "exec")); sys.argv = _argv
ROUNDS, REPS, PROF = 7, 5, 25
ACC = [dict() for _ in range(W)]; CUR = ["?"]; ON = [False]
def _add(w, st, dw, dc):
    a = ACC[w].get((CUR[0], st))
    if a is None: ACC[w][(CUR[0], st)] = [dw, dc, 1]
    else: a[0] += dw; a[1] += dc; a[2] += 1
def i_tmL(PR, tab, y, r0, r1, T, w, B):
    q = PR.shape[1]; idx = scr(("iL", w, T), (T * F5,), np.intp)[:T * q].reshape(T, q); gv = scr(("gL", w, T), (T * F5,), np.float32)[:T * q].reshape(T, q)
    off = OFF5[:q]; one = ONES[:q]
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        t0 = pc(); c0 = tt(); np.add(PR[o0:o0+n], off, out=idx[:n]); _add(w, "idx", pc() - t0, tt() - c0)
        for b in range(B):
            t0 = pc(); c0 = tt(); np.take(tab[b], idx[:n], out=gv[:n]); _add(w, "take", pc() - t0, tt() - c0)
            t0 = pc(); c0 = tt(); np.dot(gv[:n], one, out=y[b, o0:o0+n]); _add(w, "dot", pc() - t0, tt() - c0)
def i_t5vg(PR, xT, yT, r0, r1, T, w, i):
    q = PR.shape[1]; sv = scr(("v5", w, T), (T * F5,), "V5")[:T * q].reshape(T, q); s = scr(("f5", w, T), (T * F5 * 5,), np.float32)[:T * q * 5].reshape(T, q * 5)
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        t0 = pc(); c0 = tt(); np.take(LUT5V, PR[o0:o0+n], out=sv[:n]); _add(w, "take", pc() - t0, tt() - c0)
        t0 = pc(); c0 = tt(); np.copyto(s[:n], sv[:n].view(np.int8).reshape(n, q * 5), casting="unsafe"); _add(w, "cast", pc() - t0, tt() - c0)
        t0 = pc(); c0 = tt(); np.matmul(s[:n, :i], xT, out=yT[o0:o0+n]); _add(w, "mm", pc() - t0, tt() - c0)
def i_8g(Q, xT, yT, r0, r1, T, w, i):
    _, s = par.scratch(w, T, i // 4)
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        t0 = pc(); c0 = tt(); np.copyto(s[:n], Q[o0:o0+n], casting="unsafe"); _add(w, "cast", pc() - t0, tt() - c0)
        t0 = pc(); c0 = tt(); np.matmul(s[:n], xT, out=yT[o0:o0+n]); _add(w, "mm", pc() - t0, tt() - c0)
def i_8d(Q, x, y, r0, r1, T, w, i):
    _, s = par.scratch(w, T, i // 4)
    for o0 in range(r0, r1, T):
        n = min(T, r1 - o0)
        t0 = pc(); c0 = tt(); np.copyto(s[:n], Q[o0:o0+n], casting="unsafe"); _add(w, "cast", pc() - t0, tt() - c0)
        for b in range(x.shape[0]):
            t0 = pc(); c0 = tt(); np.dot(s[:n], x[b], out=y[b, o0:o0+n]); _add(w, "dot", pc() - t0, tt() - c0)
O_tmL, O_t5vg, O_8g, O_8d = rows_tmL, rows_t5vg, rows8g, rows8d
def instr(on):
    g = globals(); ON[0] = on
    g["rows_tmL"], g["rows_t5vg"], g["rows8g"], g["rows8d"] = (i_tmL, i_t5vg, i_8g, i_8d) if on else (O_tmL, O_t5vg, O_8g, O_8d)
_mm0 = mm
def mm_wrap(k, a, kind, T, mode, rec):
    CUR[0] = k; return _mm0(k, a, kind, T, mode, rec)
mm = mm_wrap
CF = {1: {"t5_dy": ("tmL", 384, "dy"), "t5_gd2": ("tmL", 384, "gd2"), "i8_dy": ("8d", 128, "dy"), "i8_gd2": ("8d", 128, "gd2")},
      4: {"t5_dy": ("t5vg", 256, "dy"), "t5_gd2": ("t5vg", 256, "gd2"), "i8_dy": ("8g", 128, "dy"), "i8_gd2": ("8g", 128, "gd2")}}
def main():
    seed = int(os.environ.get("XSEED", "1")); res = dict(xseed=seed, numpy=np.__version__, cpus=os.cpu_count(), W=W, time={}, prof={}, err={})
    for B in (1, 4):
        C = CF[B]; names = list(C); xr = XREF[:B]; e = {}
        for c in names:
            instr(False); y0 = layer(xr, *C[c], []); instr(True); y1 = layer(xr, *C[c], []); instr(False)
            r = REF[2 if c.startswith("t5") else 8][:B]
            e[c] = dict(mab=float(np.abs(y0 - r).max()), cos=cos(y0, r), instr_identical=bool(np.array_equal(y0, y1)))
        res["err"][f"B{B}"] = e; print(f"err B{B}", e, flush=True)
        xb = np.random.default_rng(seed * 100 + B).standard_normal((B, D)).astype(np.float32)
        tm = {}
        for on in (False, True):
            instr(on); rm = {c: [] for c in names}
            for c in names: layer(xb, *C[c], [])
            for rd in range(ROUNDS):
                for c in names[rd % len(names):] + names[:rd % len(names)]:
                    s = []
                    for _ in range(REPS):
                        t0 = pc(); layer(xb, *C[c], []); s.append((pc() - t0) * 1e3)
                    rm[c].append(float(np.median(s)))
            tm["instr" if on else "plain"] = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), rm=v) for c, v in rm.items()}
        instr(False); res["time"][f"B{B}"] = tm
        for c in names: print(f"B{B} {c:7s} plain {tm['plain'][c]['med']:.3f} ms [{tm['plain'][c]['lo']:.3f}-{tm['plain'][c]['hi']:.3f}]  instr {tm['instr'][c]['med']:.3f} ms", flush=True)
        pr = {}
        instr(True)
        for c in names:
            per = {}
            for _ in range(PROF):
                for d in ACC: d.clear()
                rec = []; layer(xb, *C[c], rec)
                it = {}
                for (site, st), (dw, dc, n) in ((k, v) for d in ACC for k, v in d.items()):
                    a = it.setdefault(site, {}).setdefault(st, [0.0, 0.0, 0]); a[0] += dw; a[1] += dc; a[2] += n
                for site, w, t0, t1, cpu, n in rec:
                    s0 = site.split(":")[0]; st = "bld" if site.endswith(":bld") else "task"
                    a = it.setdefault(s0, {}).setdefault(st, [0.0, 0.0, 0]); a[0] += t1 - t0; a[1] += cpu; a[2] += 1
                    if st == "task": sp = it[s0].setdefault("span", [0.0, 0.0, 0]); sp[0] = max(sp[0], t1)
                    else: sp = it[s0].setdefault("bspan", [0.0, 0.0, 0]); sp[0] = max(sp[0], t1)
                for site, d in it.items():
                    stg = sum(v[0] for k, v in d.items() if k not in ("task", "bld", "span", "bspan")); stc = sum(v[1] for k, v in d.items() if k not in ("task", "bld", "span", "bspan"))
                    d["other"] = [d["task"][0] - stg, d["task"][1] - stc, 0]
                    for st, v in d.items(): per.setdefault(site, {}).setdefault(st, []).append(v)
            pr[c] = {s: {st: dict(wall=float(np.median([v[0] for v in L])) * 1e6, cpu=float(np.median([v[1] for v in L])) * 1e6, calls=int(np.median([v[2] for v in L]))) for st, L in d.items()} for s, d in per.items()}
            for s, d in pr[c].items():
                print(f"B{B} {c:7s} {s:5s} " + "  ".join(f"{st} {v['wall']:.0f}/{v['cpu']:.0f}us({v['calls']})" for st, v in d.items()), flush=True)
        instr(False); res["prof"][f"B{B}"] = pr
    json.dump(res, open(sys.argv[1], "w"), indent=1)
main()
```

## Results

### Plain layer time (ms, median [round min–max])

| B | config | run 1 | run 2 |
|---|---|---|---|
| 1 | t5 tmL dy_T | 8.620 [8.282–9.264] | 10.846 [10.224–10.965] |
| 1 | t5 tmL gd2 | 8.605 [8.428–11.717] | 10.583 [10.233–11.066] |
| 1 | i8 8d dy_T | 5.727 [5.384–5.999] | 7.100 [6.850–7.381] |
| 1 | i8 8d gd2 | 5.588 [5.022–5.952] | 6.941 [6.305–7.746] |
| 4 | t5 t5vg dy_T | 18.076 [17.551–19.039] | 17.932 [17.114–19.773] |
| 4 | t5 t5vg gd2 | 16.680 [16.046–18.210] | 18.044 [15.737–18.937] |
| 4 | i8 8g dy_T | 9.348 [8.828–9.515] | 9.002 [8.288–9.549] |
| 4 | i8 8g gd2 | 10.590 [9.990–10.806] | 10.173 [9.797–10.637] |

Best t5 / best i8: B 1 1.540x (run 1), 1.525x (run 2); B 4 1.784x, 1.992x. Instrumentation overhead: B 1 +0.5–3.7 ms (largest in run 1, where per-call timers widen the GIL window), B 4 −1.1 to +1.2 ms. So the B 4 split is trustworthy, and the B 1 split is read from run 2 first.

### Stage worker time per layer, summed over 4 sites and 8 workers (wall ms / CPU ms)

| B | config | run | idx | take | cast | dot / mm | other | task | bld round | Σ site spans + bld spans (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | t5 dy_T | 1 | 15.5 / 12.0 | 31.2 / 22.2 | – | 5.9 / 2.3 | 1.5 / 1.3 | 55.0 / 37.9 | 4.9 / 1.4 | 8.87 + 1.64 |
| 1 | t5 dy_T | 2 | 14.6 / 11.5 | 28.0 / 21.4 | – | 4.0 / 1.9 | 1.1 / 1.0 | 48.3 / 35.8 | 3.8 / 1.2 | 8.42 + 1.38 |
| 1 | t5 gd2 | 2 | 16.3 / 11.9 | 34.7 / 22.2 | – | 6.1 / 2.1 | 1.5 / 1.4 | 58.5 / 37.6 | 3.4 / 1.2 | 8.97 + 1.29 |
| 1 | i8 dy_T | 1 | – | – | 34.2 / 15.5 | 26.5 / 10.2 | 3.0 / 2.8 | 63.5 / 28.5 | – | 9.28 |
| 1 | i8 dy_T | 2 | – | – | 26.1 / 14.5 | 20.0 / 8.5 | 2.0 / 1.8 | 48.1 / 24.7 | – | 7.26 |
| 1 | i8 gd2 | 2 | – | – | 20.2 / 13.3 | 17.1 / 9.4 | 1.6 / 1.4 | 39.3 / 24.1 | – | 6.09 |
| 4 | t5 dy_T | 1 | – | 64.5 / 46.3 | 17.3 / 13.4 | 24.4 / 20.7 | 2.3 / 2.0 | 109.0 / 82.6 | – | 17.27 |
| 4 | t5 dy_T | 2 | – | 63.4 / 48.5 | 16.4 / 13.7 | 23.2 / 20.9 | 1.8 / 1.6 | 105.3 / 84.8 | – | 17.07 |
| 4 | t5 gd2 | 1 | – | 67.6 / 49.5 | 20.5 / 17.1 | 26.2 / 22.9 | 2.3 / 2.0 | 117.1 / 91.5 | – | 17.10 |
| 4 | t5 gd2 | 2 | – | 67.4 / 48.6 | 21.1 / 17.8 | 25.9 / 22.7 | 1.8 / 1.5 | 116.9 / 91.5 | – | 17.27 |
| 4 | i8 dy_T | 1 | – | – | 29.2 / 14.9 | 35.2 / 23.2 | 2.6 / 2.4 | 66.9 / 40.4 | – | 9.75 |
| 4 | i8 dy_T | 2 | – | – | 19.9 / 13.3 | 26.4 / 20.6 | 1.5 / 1.4 | 48.0 / 35.8 | – | 7.35 |
| 4 | i8 gd2 | 1 | – | – | 38.3 / 17.7 | 33.5 / 23.8 | 2.2 / 2.0 | 74.5 / 43.3 | – | 11.47 |

Per-site numbers (µs, call counts, wall/CPU per stage per site) are in run1.txt / run2.txt beside sp.py.

### Gap attribution

- **B 4 (t5vg gd2 vs i8 dy_T):** worker-time excess is 50.2 ms (run 1) / 68.9 ms (run 2). The void5 take is 67.6 / 67.4 ms, so 98–135% of the excess. t5's cast (20.5–21.1) + matmul (25.9–26.2) = 46.4–47.3 ms is at or below int8's cast + matmul (46.3–64.4 ms), because the t5 tile cast reads 1 B/weight from the 5-byte expanded rows like int8 does. The take is 72–74% on-CPU, so it is a real compute/memory gather (one 5-byte copy per packed byte), not GIL wait. The int8 stages are 33–56% off-CPU. Σ site spans: t5 17.1–17.3 ms vs int8 7.4–9.8 ms.
- **B 1 (tmL dy_T vs i8 gd2, run 2):** t5 spends 48.3 + 3.8 = 52.1 worker-ms vs int8's 39.3 (dy_T: 48.1). That is close, but t5's stages are 74% on-CPU vs int8's 61% / 51%. The critical path is longer: Σ spans 8.42 + 1.38 (bld) = 9.80 ms vs 6.09 ms. That 3.71 ms span gap is 37% table-build round: a whole extra submit/result round per site, 0.32–0.45 ms each, with only ~0.15 ms of CPU per site over 8 workers (68–75% off-CPU = dispatch and GIL hand-off). The other 63% is longer site spans driven by idx + take (42.6 ms, 25% off-CPU). np.dot over the gathered row is small (4.0–5.9 ms, 60% off-CPU).

## Verdict

**not-viable**. This is a diagnostic. The t5 paths stay 1.52–1.54x (B 1) and 1.78–1.99x (B 4) slower than int8, above the 1.3x bar and nowhere near a 1.5x ternary win. The t5 weight-memory win (4.973x vs int8) is unchanged and already known. Where it goes: at B 4 the excess is the on-CPU void5 LUT take expand (≈100% of the worker-time gap). At B 1 the excess is not extra worker time but the serial table-build round (~37% of the span gap) plus a more on-CPU idx/take critical path (~63%).

## Pair files this wave

No Trit Pack / Handshake / Quant Chief pair report for the 2026-11-26 02:55 wave reached this lane. JuniorLLM HEAD at clone was 5cb92ae (this lane's own 18:50 push), so no pair or LLC path was touched in the last 6 hours.

## Next question

At B 4, can the expand + cast pair be replaced by a single take from a 243-row f32 LUT viewed as 20-byte void rows ("V20": 5 f32 per packed byte), writing the f32 tile directly and skipping the int8 cast? Alternatively, take into int8 and fold the cast into np.matmul's input. Measure the t5vg stage split again (take20 vs take5 + cast) at B 4 and B 8, the scratch bytes (4x larger expanded tile), and whether t5vg's layer time drops by at least 1.3x toward int8g.

close: 7ab94ec77e3987b3
