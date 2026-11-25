---
local: true
date: 2026-11-25
wave: "2026-11-25 02:55 slot (fired 02:47 MT, ran ~02:47–03:00 MT; no pair reports for this wave reached this lane; Quant Chief has named no eligible item: builders idle, ROADMAP.md / BETA_TO_OS.md exhausted)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (prefill) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: "not-viable (base-3 t5 prefill via a 243 x 5 LUT expand + np.matmul reaches speed parity with the 2-bit u32-LUT expand: t5v_T256 / u32_T256 = 1.057x / 1.039x at P 256, 0.978x / 1.054x at P 1024, 1.007x / 1.030x at P 2048 (seed 1 / seed 2), within the 1.1x bar at every P, output bit-identical to 2-bit; vs int8_T128 it is 1.255x / 1.244x at P 256, 1.059x / 1.079x at P 1024, 1.115x / 1.122x at P 2048, so the 1.1x int8 bar is met only at P 1024 (2-bit itself is 1.19x / 1.20x of int8 at P 256); no new >= 1.5x gain; the 4.973x weight-memory win vs int8 is already known)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, prefill, chunked, base-3, 1p6-bit, five-trit-pack, lut-expand, u32-lut, int8, matmul, thread-pool, numpy, not-viable]
---

# Prefill with the base-3 t5 pack: 243 x 5 LUT expand + matmul is at parity with the 2-bit u32-LUT expand (0.98–1.06x), 1.06–1.26x of int8; t5 can be the single resident format, but no new speed win

The 18:55 note ([[2026-11-24_1855_b1-base3-t5-tmac-balanced-w-way-row-partition-sub-tile-t384-vs-unbalanced-and-int8-same-partition-full-bitnetd-layer-8-thread-pool-no-gain-gap-stays-1p5x-series-retired]]) retired the B 1 numpy GEMV series and asked: rotate to prefill with the base-3 pack, which no earlier note covered (the 11-17 prefill notes used only the 2-bit u32-LUT expand). At prefill P 256 / 1024 / 2048 (chunk 512, W 8, full bitnetd layer), expand each P5 row tile to an int8 / f32 tile with a 243 x 5 LUT take, drop the padding, then np.matmul against the chunk. Is that within 1.1x of the 2-bit u32-LUT expand prefill and of int8 prefill? If so, t5 can be the single resident weight format for decode and prefill at 4.97x less weight memory than int8 and 1.25x less than 2-bit.

This is not a repeat. The 11-17 prefill notes (10:55, 14:55, 18:55) expanded only the 2-bit u32 pack. The 11-24 base-3 notes were all B 1 T-MAC GEMV, none at prefill.

Answers:
- **t5 vs 2-bit u32-LUT expand (bar 1.1x):** t5v_T256 / u32_T256 = **1.057x (rounds 1.02–1.30) / 1.039x (0.94–1.10) at P 256, 0.978x (0.95–1.11) / 1.054x (0.95–1.12) at P 1024, 1.007x (0.95–1.10) / 1.030x (0.88–1.06) at P 2048** (seed 1 / seed 2). **Met at every P.** The expand is a small part of a prefill chunk: one 256-row tile of gate|up takes ~0.57 ms to expand vs ~3.3 ms for its matmul at C 512 (single-thread micro-check). So 5 trits per byte instead of 4 costs nothing that shows up.
- **t5 vs int8 (bar 1.1x):** t5v_T256 / int8_T128 = **1.255x (1.15–1.49) / 1.244x (1.11–1.28) at P 256, 1.059x (0.79–1.18) / 1.079x (1.01–1.22) at P 1024, 1.115x (1.04–1.22) / 1.122x (0.99–1.17) at P 2048**. **Met only at P 1024**; P 2048 misses by a hair, P 256 clearly. This is the 2-bit expand's own gap, not a t5 cost: u32_T256 / int8_T128 is 1.187x / 1.197x at P 256, 1.083x / 1.024x at P 1024, 1.108x / 1.089x at P 2048.
- **Expand variants:** the void5 take + int8→f32 cast (t5v) beats a direct f32 (243, 5) axis-0 take (t5f: 1.123x / 1.058x, 1.048x / 1.070x, 1.019x / 1.022x of u32) and T 512 (t5v_T512: 1.116x / 1.061x, 1.074x / 1.073x, 1.019x / 1.045x of u32). Keep t5v at T 256.
- **Correctness:** every t5 config gives **bit-identical output to the 2-bit path** on a full C 512 chunk (max abs 0.0, cos 1.0, both seeds), since the trits are the same and the padding columns are dropped by a strided view. vs the float64 ternary reference on 32 tokens: max abs **1.2e-06 / 1.3e-06**, cos 0.99999999999997. int8 vs its own f64 reference: 4.7e-06 / 4.5e-06.
- **Memory:** t5 weights 13,986,304 B (13.34 MiB) vs 17,458,176 B 2-bit (1.248x) vs 69,559,296 B int8 (4.973x). Already known, not new. Expand scratch (all W 8 workers, sized for in 6912): t5v_T256 70,809,600 B (8.44 MiB / worker: void5 + f32 tile at in5 6915), t5f_T256 56,647,680 B, u32_T256 70,778,880 B, int8_T128 35,389,440 B. LUTs: 1,215 B (void5) / 4,860 B (f32) vs 1,024 B (u32).
- **Verdict: not-viable for the routine** (no new >= 1.5x gain on speed or memory). The question's own bar is met against 2-bit: t5 can replace the 2-bit pack as the single resident weight format for both decode (11-24: t5 T-MAC 1.25x faster than 2-bit T-MAC) and prefill (parity here) with 1.248x less weight memory. Against int8 the prefill speed gap (1.06–1.26x) is unchanged from the 2-bit expand.

## Method

- Same layer, weights and P5 base-3 pack as 11-24 18:55 (tm5b.py → /workspace/bench1455_1118 base.py / par.py: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits, fused qkv / o / gate|up / down, per-row gamma, attention stub). Every trit round-trips through the base-3 pack (asserted at build). Prefill harness as 11-17 14:55 (crossover.py): P tokens processed in chunks C = 512 through the threaded layer, configs rotated each round.
- New t5 prefill paths: per worker tile of T rows, t5v: np.take(LUT5 viewed as 243 x void5, P5 tile) → (n, in5) int8 view → cast into a (T, in5) f32 scratch → np.matmul(chunk, s[:n, :in].T) (strided view drops the in5 − in padding columns, no copy). t5f: np.take(LUT5F (243, 5) f32, P5 tile, axis=0) straight into the f32 scratch, same matmul.
- Baselines rerun in the same process: u32_T256 (par.layer: 2-bit u32-LUT take + int8→f32 cast + matmul; the 11-17 best at C 512) and int8_T128 / int8_T256 (par.layer8: int8 rows cast to f32 + matmul). All paths use the same dispatch (ceil(out/T) tiles, ceil(tiles/W) per worker), OPENBLAS_NUM_THREADS=1, W 8 ThreadPoolExecutor on 8 cores.
- Timing: per P, 1 warm-up, 7 rounds x 3 reps per config, config order rotated each round. Value = median of round medians; spread = min–max of round medians; ratios are per-round paired. Two input seeds (weights fixed by base.SEED). Total run time ~3.7 min for both seeds. python3 + numpy only; no installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench0255_1125 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 900 python3 pf5.py run$s.json > run$s.txt 2>&1; done

Script (pf5.py; imports base / par from /workspace/bench1455_1118):

    #!/usr/bin/env python3
    """Prefill with the base-3 (5 trits/byte, 1.6 bit) resident pack, W=8 pool, full bitnetd layer, 02:50 MT 11/25.
    Base: bench1855_1124/tm5b.py (same layer, weights, P5 pack + exact round-trip assert, pool) and bench1455_1117/crossover.py
    (prefill P 256/1024/2048, chunked, rounds rotated). Chunk C 512 throughout.
    t5v_T*: per worker tile, np.take(LUT5 viewed as 243 x void5, P5 tile) -> (n, in5) int8 view -> cast to a (T, in5) f32 scratch ->
            np.matmul(chunk, s[:n, :in].T) (padding columns dropped by a strided view, no copy).
    t5f_T*: per worker tile, np.take(LUT5F (243, 5) f32, P5 tile, axis=0) straight into the (T, in5/5, 5) f32 scratch (no int8 step) -> same matmul.
    Baselines rerun in the same process: u32_T256 (par.layer: 2-bit u32-LUT take + int8->f32 cast + matmul, 11-17 best at C 512) and
    int8_T128 / int8_T256 (par.layer8: int8 rows cast to f32 + matmul). Dispatch for all: ceil(out/T) tiles, ceil(tiles/W) tiles per worker.
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=s python3 pf5.py out.json"""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import D, F, KVH, QH, HD, rms, silu
    W, ROUNDS, REPS, WARM, C = 8, 7, 3, 1, 512
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
    # f64 reference weights, kept per site only as long as needed (box has ~2.5 GB free): ternary (gamma * trit) and int8 (m8 * q8)
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
    LUT5F = np.ascontiguousarray(L5.astype(np.float32))             # (243, 5) f32 (4,860 B)
    F5 = -(-F // 5)
    _s5 = {}
    def scr5(w, T, q):
        k = (w, T)
        if k not in _s5: _s5[k] = (np.empty(T * F5, "V5"), np.empty(T * F5 * 5, np.float32))
        a, b = _s5[k]; return a[:T * q].reshape(T, q), b[:T * q * 5].reshape(T, q * 5)
    def rows_t5v(PR, x, y, r0, r1, T, w, i):
        q = PR.shape[1]; sv, s = scr5(w, T, q)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.take(LUT5V, PR[o0:o0+n], out=sv[:n])
            np.copyto(s[:n], sv[:n].view(np.int8).reshape(n, q * 5), casting="unsafe")
            np.matmul(x, s[:n, :i].T, out=y[:, o0:o0+n])
    def rows_t5f(PR, x, y, r0, r1, T, w, i):
        q = PR.shape[1]; _, s = scr5(w, T, q)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.take(LUT5F, PR[o0:o0+n], axis=0, out=s[:n].reshape(n, q, 5))
            np.matmul(x, s[:n, :i].T, out=y[:, o0:o0+n])
    FN = {"t5v": rows_t5v, "t5f": rows_t5f}
    def fmm5(key, x, kind, T):
        P = P5[key]; PR = P["PR"]; out = PR.shape[0]; i = P["inp"]
        x = np.ascontiguousarray(x, np.float32); y = np.empty((x.shape[0], out), np.float32)
        nt = -(-out // T); per = -(-nt // W)
        futs = [par.pool(W).submit(FN[kind], PR, x, y, w * per * T, min(out, (w + 1) * per * T), T, w, i) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
        y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
    def layer5(x, kind, T):
        mm = lambda k, a: fmm5(k, a, kind, T)
        h1 = rms(x); q, kk, v = mm("qkv", h1)
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        o = mm("o", att)[0]; x2 = x + o; h2 = rms(x2)
        g, u = mm("gu", h2); z = (silu(g) * u).astype(np.float32)
        return x2 + mm("down", z)[0]
    CFG = {"u32_T256": lambda x: par.layer(x, 256, W), "int8_T128": lambda x: par.layer8(x, 128, W), "int8_T256": lambda x: par.layer8(x, 256, W),
           "t5v_T256": lambda x: layer5(x, "t5v", 256), "t5f_T256": lambda x: layer5(x, "t5f", 256), "t5v_T512": lambda x: layer5(x, "t5v", 512)}
    BITS = {c: 8 if c.startswith("int8") else 2 for c in CFG}
    def chunked(f): return lambda x: np.concatenate([f(x[i:i + C]) for i in range(0, x.shape[0], C)])
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    def main():
        seed = int(os.environ.get("XSEED", "1")); res = {"xseed": seed, "numpy": np.__version__, "cpus": os.cpu_count(), "C": C, "W": W}
        W2 = sum(par.FP[k]["PR"].nbytes + par.FP[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        W5 = sum(P5[k]["PR"].nbytes + P5[k]["g"].nbytes for k in SITES)
        sc = dict(u32_T256=W * 256 * (F // 4) * 4 + W * 256 * F * 4, int8_T128=W * 128 * F * 4 + W * 128 * (F // 4) * 4, int8_T256=W * 256 * F * 4 + W * 256 * (F // 4) * 4,
                  t5v_T256=W * 256 * F5 * (5 + 20), t5f_T256=W * 256 * F5 * 20, t5v_T512=W * 512 * F5 * (5 + 20))
        res["bytes"] = dict(t5=W5, u2=W2, int8=W8, int8_over_t5=W8 / W5, u2_over_t5=W2 / W5, scratch_alloc=sc, lut5v=LUT5V.nbytes, lut5f=LUT5F.nbytes, lut_u32=par.LUT_U32.nbytes)
        print("bytes", res["bytes"], flush=True)
        err = {}
        for c, f in CFG.items():
            y = f(XREF); err[c] = dict(mab=float(np.abs(y - REF[BITS[c]]).max()), cos=cos(y, REF[BITS[c]]))
        rng = np.random.default_rng(seed); xc = rng.standard_normal((C, D)).astype(np.float32)
        yu = CFG["u32_T256"](xc)
        for c in ("t5v_T256", "t5f_T256", "t5v_T512"):
            y = CFG[c](xc); err[c]["vs_u32_C512_mab"] = float(np.abs(y - yu).max()); err[c]["vs_u32_C512_cos"] = cos(y, yu)
        res["err"] = err; print("err", err, flush=True)
        names = list(CFG); res["time"] = {}
        for Pn in (256, 1024, 2048):
            xb = np.random.default_rng(seed * 100 + Pn).standard_normal((Pn, D)).astype(np.float32)
            paths = {c: chunked(f) for c, f in CFG.items()}; rm = {c: [] for c in names}
            for c in names:
                for _ in range(WARM): paths[c](xb)
            for rd in range(ROUNDS):
                for c in names[rd % len(names):] + names[:rd % len(names)]:
                    s = []
                    for _ in range(REPS):
                        t0 = time.perf_counter(); paths[c](xb); s.append((time.perf_counter() - t0) * 1e3)
                    rm[c].append(float(np.median(s)))
            o = {c: dict(med=float(np.median(v)), lo=min(v), hi=max(v), us_tok=float(np.median(v)) * 1e3 / Pn, rm=v) for c, v in rm.items()}
            rat = {}
            for a in ("t5v_T256", "t5f_T256", "t5v_T512"):
                for b in ("u32_T256", "int8_T128", "int8_T256"):
                    v = [p / q for p, q in zip(rm[a], rm[b])]; rat[f"{a}/{b}"] = dict(r=o[a]["med"] / o[b]["med"], lo=min(v), hi=max(v))
            for b in ("int8_T128", "int8_T256"):
                v = [p / q for p, q in zip(rm["u32_T256"], rm[b])]; rat[f"u32_T256/{b}"] = dict(r=o["u32_T256"]["med"] / o[b]["med"], lo=min(v), hi=max(v))
            res["time"][f"P{Pn}"] = dict(cfg=o, ratios=rat)
            for c in names: print(f"P{Pn} {c:10s} {o[c]['med']:8.2f} ms [{o[c]['lo']:.2f}-{o[c]['hi']:.2f}] {o[c]['us_tok']:.1f} us/tok", flush=True)
            for k, v in rat.items(): print(f"P{Pn} ratio {k:22s} {v['r']:.3f}x (rounds {v['lo']:.2f}-{v['hi']:.2f})", flush=True)
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    main()

## Results

Full layer prefill, chunk 512, W 8 (ms per P tokens; median of round medians, min–max of round medians; seed 1 / seed 2):

| config | P 256 | P 1024 | P 2048 |
|---|---|---|---|
| u32_T256 (2-bit expand) | 55.88 [51.7–58.7] / 61.67 [59.0–68.3] | 214.10 [198.7–223.4] / 199.07 [194.7–215.7] | 450.24 [429.2–488.0] / 423.95 [413.6–491.5] |
| int8_T128 | 47.09 [43.9–50.3] / 51.50 [51.2–58.1] | 197.79 [187.4–270.1] / 194.48 [184.1–203.9] | 406.33 [402.0–474.7] / 389.31 [377.0–404.1] |
| int8_T256 | 47.71 [46.6–82.1] / 54.42 [51.3–55.9] | 198.77 [191.4–205.2] / 193.79 [184.0–225.2] | 402.75 [392.9–450.7] / 396.44 [370.0–423.7] |
| **t5v_T256** | **59.09** [55.0–70.9] / **64.07** [61.9–67.8] | **209.40** [203.9–220.5] / **209.81** [204.9–224.7] | **453.18** [436.9–496.0] / **436.78** [389.0–456.2] |
| t5f_T256 | 62.78 [56.3–69.3] / 65.23 [62.6–71.9] | 224.38 [216.4–252.7] / 213.10 [194.2–237.4] | 458.72 [442.4–502.8] / 433.27 [406.3–545.2] |
| t5v_T512 | 62.34 [57.6–72.9] / 65.45 [59.7–83.7] | 229.97 [212.1–252.3] / 213.58 [206.9–220.1] | 458.70 [428.9–536.2] / 443.12 [393.0–556.5] |

t5v_T256 throughput: 230.8 / 250.3 µs per token at P 256, 204.5 / 204.9 at P 1024, 221.3 / 213.3 at P 2048 (int8_T128: 183.9 / 201.2, 193.1 / 189.9, 198.4 / 190.1).

Paired ratios (seed 1 / seed 2, per-round range in brackets; bar <= 1.1x):

| ratio | P 256 | P 1024 | P 2048 |
|---|---|---|---|
| **t5v_T256 / u32_T256** | **1.057x (1.02–1.30) / 1.039x (0.94–1.10)** | **0.978x (0.95–1.11) / 1.054x (0.95–1.12)** | **1.007x (0.95–1.10) / 1.030x (0.88–1.06)** |
| **t5v_T256 / int8_T128** | **1.255x (1.15–1.49) / 1.244x (1.11–1.28)** | **1.059x (0.79–1.18) / 1.079x (1.01–1.22)** | **1.115x (1.04–1.22) / 1.122x (0.99–1.17)** |
| t5v_T256 / int8_T256 | 1.238x / 1.177x | 1.053x / 1.083x | 1.125x / 1.102x |
| t5f_T256 / u32_T256 | 1.123x / 1.058x | 1.048x / 1.070x | 1.019x / 1.022x |
| t5f_T256 / int8_T128 | 1.333x / 1.267x | 1.134x / 1.096x | 1.129x / 1.113x |
| t5v_T512 / u32_T256 | 1.116x / 1.061x | 1.074x / 1.073x | 1.019x / 1.045x |
| t5v_T512 / int8_T128 | 1.324x / 1.271x | 1.163x / 1.098x | 1.129x / 1.138x |
| u32_T256 / int8_T128 (2-bit's own gap) | 1.187x / 1.197x | 1.083x / 1.024x | 1.108x / 1.089x |

Error (32-token check vs own float64 reference; seed 1 / seed 2): all t5 configs and u32_T256 max abs 1.2e-06 / 1.3e-06, cos 0.99999999999997; int8 4.7e-06 / 4.5e-06, cos 0.9999999999998. t5 vs u32_T256 on a full 512-token chunk: max abs 0.0, cos 1.0 (bit-identical).

Single-thread micro-check (micro.py, one gate|up tile T 256 x in 2560, C 512): void5 take + cast 0.57 ms, f32 axis-0 take 0.68 ms, u64-padded take + strided cast 0.84 ms (all bit-equal), matmul 3.26 ms.

Bytes: weights per layer 13,986,304 B t5 vs 17,458,176 B 2-bit vs 69,559,296 B int8 (int8/t5 4.973x, 2-bit/t5 1.248x). Expand scratch, all 8 workers: t5v_T256 70,809,600 B, t5f_T256 56,647,680 B, t5v_T512 141,619,200 B, u32_T256 70,778,880 B, int8_T128 35,389,440 B (par.scratch also allocates an unused u32 buffer for int8).

## Pair files this wave

No pair files (Trit Pack, Winsor, Side-Commit, Handshake reports) for this wave reached this lane, so none are linked. The first depth-1 clone of JuniorLLM showed this lane's own 8f1ab8a (11-24 18:49 MT, the 18:55 note) as the newest commit. The first push was rejected because an unattributed cloudcover95 commit landed meanwhile: 23ed5da "rails/cleanclone: shallow clone guard" (02:54 MT), which adds only rails/cleanclone/shallow_clone_check.py. This lane re-cloned once and pushed on top of it. Nothing in the last 6 hours touches junior_osai/vault/findings, and no Trit Pack, Handshake or LLC commit appears. This lane did not read JuniorHome this wave.

## Next question

Prefill parity makes t5 the single resident format, so the open regime is between decode and prefill. Small-batch crossover for the t5 pack: at B 2 / 4 / 8 / 16 / 32 tokens per weight pass (speculative-decode verify widths or multi-stream decode, full bitnetd layer, W 8, ctx-free), which t5 path is faster, the T-MAC gather (one 243-entry table per input row, B tables) or the void5 LUT expand + matmul? Where is the crossover B, and does the faster path stay within 1.3x of int8 (T 128) at each B? No earlier note ran t5 between B 1 and prefill; the 11-18 multi-stream note used only the 2-bit u32 expand.

close: c435113f04b212b5
