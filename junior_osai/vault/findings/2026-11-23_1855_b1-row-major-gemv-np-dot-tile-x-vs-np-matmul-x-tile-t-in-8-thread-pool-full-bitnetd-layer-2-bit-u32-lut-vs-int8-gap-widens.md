---
local: true
date: 2026-11-23
wave: "2026-11-23 18:55 (fired ~18:49 MT, ran ~18:50–19:05 MT; latest ledger is Quant Chief's 14:05 wave: quant builders idle, no eligible item)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (the B 1 row-major GEMV fix works for both formats, 1.763/1.826x for 2-bit and 2.584/2.697x for int8, but it widens the 2-bit-vs-int8 gap from 1.323/1.332x to 1.940/1.967x, so ternary loses more speed at single-user decode; the weight-memory win stays 3.984x)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, b1, gemv, matmul, u32-lut, int8, thread-pool, numpy, not-viable]
---

# B 1 decode: a row-major GEMV (np.dot(tile, x[0])) instead of np.matmul(x, tile.T) makes the full layer 1.76–1.83x faster at 2-bit and 2.58–2.70x faster at int8, so the 2-bit/int8 gap widens from ~1.33x to ~1.95x

The 14:55 note ([[2026-11-23_1455_mixed-precision-site-split-ffn-gate-up-down-2-bit-u32-lut-qkv-o-int8-vs-all-2-bit-vs-all-int8-full-bitnetd-layer-b1-b4-weights-plus-kv-a8-g64-k8v4-ctx-2048]]) found every config was slower per step at B 1 than at B 4 (~19 vs ~13 ms). It asked: at B 1, does replacing np.matmul(x(1,in), tile.T) with a contiguous row-major GEMV (np.dot(tile, x[0]) into the output slice) cut full-layer time by >= 1.3x for both all-2-bit and all-int8, and does the all-2-bit/int8 gap then widen or shrink from ~1.37x?

Answers:
- **GEMV fix: yes, for both formats.** Full-layer B 1 time (seed 1 / seed 2, median of per-round medians): all-2-bit 21.05/18.61 -> 11.94/10.19 ms (**1.763/1.826x**, rounds 1.59–1.98); all-int8 15.91/13.97 -> 6.16/5.18 ms (**2.584/2.697x**, rounds 2.35–3.37). The B 1 anomaly is gone: B 1 GEMV per step is now 0.945/0.834 of B 4 matmul for 2-bit and 0.761/0.674 for int8.
- **Gap: it widens.** All-2-bit/all-int8 goes from 1.323/1.332x (matmul) to **1.940/1.967x** (GEMV), rounds 1.73–2.33. The fix removes a fixed per-tile overhead shared by both paths, which leaves the 2-bit path's extra LUT take + int8->f32 cast as most of what's left. 2-bit with GEMV is still 1.332/1.371x faster than the old int8 matmul path, but the fair comparison is int8 with GEMV.
- **Where the overhead was:** a single-thread micro (one 256 x in f32 tile, already expanded) shows no kernel difference: mm/gv 1.02/1.03x at in 2560 and 1.01/1.01x at in 6912 (gv 45 / 32 GB/s). So the B 1 cost appears only on the W = 8 thread pool. The likely cause is that np.matmul on a (1, in) x (in, n) transposed view takes a path that doesn't overlap well across worker threads, while np.dot goes straight to sgemv. That wasn't isolated here.
- **Memory:** unchanged. Layer weights are 16.65 MiB at 2-bit and 66.34 MiB at int8 (3.984x).
- **Error:** every config matches its own float64 reference layer: 2-bit max abs <= 1.0e-06, int8 <= 3.8e-06, cos 1.00000000. GEMV and matmul give the same outputs to float32 rounding.
- **Verdict: not-viable for ternary.** The GEMV change is a format-independent decode fix worth carrying into bitnetd/i2sd's B 1 path. For ternary speed it makes things worse: int8 is now ~1.95x faster at single-user decode. Ternary is still 3.98x smaller on weights, as recorded in earlier notes; nothing new on memory.

## Method

- Weights: base.build() from /workspace/bench1455_1118 (bitnetd layer: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits). par.prep() gives resident 2-bit PR rows per fused site with per-row gamma. par.prep8() is the int8 control (absmax int8 of the f32 W). Same pool, scratch and T 256 tiles as the 14:55 note.
- Configs: all2_mm / all8_mm = the current tile path (np.matmul(x, s[:n].T, out=y[:, o0:o0+n])); all2_gv / all8_gv = np.dot(s[:n], x[0], out=y[0, o0:o0+n]). The unpack (u32-LUT take + int8 view cast, or int8 cast) is the same in both.
- Timed op: one full layer (rms, fused qkv, attention stub = v repeated over GQA groups, o, residual, rms, fused gate|up, silu*up, down, residual) on the W = 8 thread pool, OPENBLAS_NUM_THREADS=1, B 1. 7 rounds x 9 reps (3 warm-up) per config, config order rotated each round. Ratios use the median of per-round medians, with the per-round min–max. B 4 matmul reference: 27 reps after 3 warm-up. Micro: one thread, 7 rounds x 50 reps per kernel. Two input seeds (XSEED 1, 2) on an 8-core box.
- No installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench1855_1123 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s timeout 800 python3 gemv.py run$s.json > run$s.txt 2>&1; done

Script (gemv.py):

    #!/usr/bin/env python3
    """B=1 GEMV vs matmul tile path, 18:55 11/23. Full bitnetd-style layer (par.layer structure) at B = 1 on the W=8 pool, T 256.
    mm = current path: np.matmul(x(1,in), tile.T, out=y[:, o0:o0+n]); gv = np.dot(tile(n,in), x[0], out=y[0, o0:o0+n]) (row-major GEMV).
    For all2 (2-bit u32-LUT take -> int8 view -> f32 tile) and all8 (int8 resident cast to f32 tile). B=4 mm reference for context.
    Does gv cut full-layer B1 time by >= 1.3x for both, and does the all2/all8 gap widen or shrink from ~1.37x?"""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import D, F, KVH, QH, HD, rms, silu
    T, W, ROUNDS, REPS, WARM = 256, 8, 7, 9, 3
    base.build(); par.prep(); par.prep8()
    SITES = ("qkv", "o", "gu", "down")
    def wf64(k, bits):
        if bits == 2: return np.concatenate([base.PW[n]["Wt"].astype(np.float64) for n in par.FUSE[k]])
        return np.concatenate([base.PW[n]["q8"].astype(np.float64) * float(base.PW[n]["m8"]) for n in par.FUSE[k]])
    WR = {(k, b): wf64(k, b) for k in SITES for b in (2, 8)}
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT", "Wt"): base.PW[nm].pop(f, None)
    LUT = par.LUT_U32
    def rows2(PR, x, y, r0, r1, w, gv):
        q = PR.shape[1]; s8, s = par.scratch(w, T, q)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.take(LUT, PR[o0:o0+n], out=s8[:n]); np.copyto(s[:n], s8[:n].view(np.int8), casting="unsafe")
            if gv: np.dot(s[:n], x[0], out=y[0, o0:o0+n])
            else: np.matmul(x, s[:n].T, out=y[:, o0:o0+n])
    def rows8(Q, x, y, r0, r1, w, gv):
        i = Q.shape[1]; _, s = par.scratch(w, T, i // 4)
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0)
            np.copyto(s[:n], Q[o0:o0+n], casting="unsafe")
            if gv: np.dot(s[:n], x[0], out=y[0, o0:o0+n])
            else: np.matmul(x, s[:n].T, out=y[:, o0:o0+n])
    def fmm(key, x, bits, gv):
        P = par.FP[key] if bits == 2 else par.F8[key]; R = P["PR"] if bits == 2 else P["Q"]; out = R.shape[0]
        fn = rows2 if bits == 2 else rows8
        x = np.ascontiguousarray(x, np.float32); y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
        futs = [par.pool(W).submit(fn, R, x, y, w * per * T, min(out, (w + 1) * per * T), w, gv) for w in range(W) if w * per * T < out]
        for f in futs: f.result()
        y *= P["g"]; return np.split(y, P["split"], axis=1) if len(P["split"]) else [y]
    def layer(x, bits, gv):
        mm = lambda k, a: fmm(k, a, bits, gv)
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
    # single-site micro: one worker, one T x in tile, f32 tile already expanded (isolates matmul-vs-gemv kernel)
    def micro(rng):
        out = {}
        for nm, i in (("in2560", D), ("in6912", F)):
            s = rng.standard_normal((T, i)).astype(np.float32); x = rng.standard_normal((1, i)).astype(np.float32); y = np.empty((1, T), np.float32)
            ts = {"mm": [], "gv": []}
            for rd in range(ROUNDS):
                for c in (("mm", "gv") if rd % 2 == 0 else ("gv", "mm")):
                    for r in range(WARM + 50):
                        t0 = time.perf_counter()
                        if c == "gv": np.dot(s, x[0], out=y[0])
                        else: np.matmul(x, s.T, out=y)
                        dt = (time.perf_counter() - t0) * 1e6
                        if r >= WARM: ts[c].append(dt)
            out[nm] = {c: dict(med=float(np.median(v)), lo=float(min(v)), hi=float(max(v))) for c, v in ts.items()}
            out[nm]["bytes"] = int(s.nbytes)
            print(f"micro {nm} T{T}: mm {out[nm]['mm']['med']:.1f} us ({out[nm]['mm']['lo']:.1f}-{out[nm]['mm']['hi']:.1f}) | gv {out[nm]['gv']['med']:.1f} us ({out[nm]['gv']['lo']:.1f}-{out[nm]['gv']['hi']:.1f}) | mm/gv {out[nm]['mm']['med']/out[nm]['gv']['med']:.2f}x | tile {s.nbytes/2**20:.2f} MiB -> gv {s.nbytes/out[nm]['gv']['med']/1e3:.2f} GB/s", flush=True)
        return out
    CFG = {"all2_mm": (2, False), "all2_gv": (2, True), "all8_mm": (8, False), "all8_gv": (8, True)}
    def main():
        seed = int(os.environ.get("XSEED", "1")); rng = np.random.default_rng(seed)
        res = {"xseed": seed, "micro": micro(rng), "B1": {}, "B4": {}}
        W2 = sum(par.FP[k]["PR"].nbytes + par.FP[k]["g"].nbytes for k in SITES); W8 = sum(par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES)
        res["wbytes"] = dict(all2=W2, all8=W8); print(f"layer weights MiB all2 {W2/2**20:.2f} all8 {W8/2**20:.2f} ({W8/W2:.3f}x)", flush=True)
        x = (rng.standard_normal((1, D)) * 0.5).astype(np.float32); R = {b: layer_ref(x, b) for b in (2, 8)}
        d = {}
        for c, (b, gv) in CFG.items():
            y = layer(x, b, gv); d[c] = dict(t=[], mab=float(np.abs(y - R[b]).max()), cos=cos(y, R[b]))
        names = list(CFG)
        for rd in range(ROUNDS):
            for c in names[rd % 4:] + names[:rd % 4]:
                for r in range(WARM + REPS):
                    t0 = time.perf_counter(); layer(x, *CFG[c]); dt = (time.perf_counter() - t0) * 1e3
                    if r >= WARM: d[c]["t"].append(dt)
        rm = {c: [float(np.median(d[c]["t"][r*REPS:(r+1)*REPS])) for r in range(ROUNDS)] for c in names}
        for c in names:
            Tm = np.array(d[c]["t"]); d[c].update(med=float(np.median(rm[c])), lo=float(Tm.min()), hi=float(Tm.max()),
                                                  iqr=float(np.percentile(Tm, 75) - np.percentile(Tm, 25)), rm=rm[c])
            print(f"B1 {c:8s} {d[c]['med']:7.3f} ms ({d[c]['lo']:.3f}-{d[c]['hi']:.3f}) iqr {d[c]['iqr']:.3f} | {1e3/d[c]['med']:6.1f} tok/s/layer | vs own f64 mab {d[c]['mab']:.1e} cos {d[c]['cos']:.8f}", flush=True)
        rr = lambda a, b: [p / q for p, q in zip(rm[a], rm[b])]
        S = {}
        for nm, (a, b) in {"sp2": ("all2_mm", "all2_gv"), "sp8": ("all8_mm", "all8_gv"), "gap_mm": ("all2_mm", "all8_mm"), "gap_gv": ("all2_gv", "all8_gv"),
                           "int8mm_over_2gv": ("all8_mm", "all2_gv")}.items():
            v = rr(a, b); S[nm] = dict(r=d[a]["med"] / d[b]["med"], lo=min(v), hi=max(v))
            print(f"B1 {nm:16s} {S[nm]['r']:.3f}x (rounds {S[nm]['lo']:.2f}-{S[nm]['hi']:.2f})", flush=True)
        d["summary"] = S; res["B1"] = d
        x4 = (rng.standard_normal((4, D)) * 0.5).astype(np.float32); d4 = {}
        for c in ("all2_mm", "all8_mm"):
            t = []
            for r in range(WARM + REPS * 3):
                t0 = time.perf_counter(); layer(x4, *CFG[c]); dt = (time.perf_counter() - t0) * 1e3
                if r >= WARM: t.append(dt)
            d4[c] = dict(med=float(np.median(t)), lo=float(min(t)), hi=float(max(t)))
            print(f"B4 {c:8s} {d4[c]['med']:7.3f} ms ({d4[c]['lo']:.3f}-{d4[c]['hi']:.3f}) | B1 gv / B4 mm per-step {d[c.replace('mm','gv')]['med']/d4[c]['med']:.3f}", flush=True)
        res["B4"] = d4
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    main()


## Results

Full layer, B 1, W 8, T 256 (ms; median of per-round medians, overall min–max):

| config | seed 1 ms | seed 1 range | seed 2 ms | seed 2 range | tok/s/layer (s1/s2) | max abs vs own f64 |
|---|---|---|---|---|---|---|
| all2_mm (2-bit, np.matmul) | 21.050 | 18.68–29.96 | 18.610 | 17.65–24.23 | 47.5 / 53.7 | 1.0e-06 / 8.0e-07 |
| all2_gv (2-bit, np.dot GEMV) | 11.943 | 10.31–24.91 | 10.189 | 8.80–25.65 | 83.7 / 98.1 | 1.0e-06 / 8.0e-07 |
| all8_mm (int8, np.matmul) | 15.907 | 12.79–32.12 | 13.969 | 12.50–16.21 | 62.9 / 71.6 | 3.8e-06 / 3.3e-06 |
| all8_gv (int8, np.dot GEMV) | 6.157 | 4.98–13.33 | 5.180 | 4.70–6.59 | 162.4 / 193.1 | 3.8e-06 / 3.3e-06 |

Ratios (seed 1 / seed 2, per-round range):

| ratio | seed 1 | seed 2 | bar |
|---|---|---|---|
| 2-bit mm / gv (GEMV speedup) | 1.763x (1.59–1.84) | 1.826x (1.70–1.98) | >= 1.3x: pass |
| int8 mm / gv (GEMV speedup) | 2.584x (2.44–3.37) | 2.697x (2.35–2.86) | >= 1.3x: pass |
| all2/all8 gap, matmul | 1.323x (1.12–1.41) | 1.332x (1.27–1.45) | — |
| all2/all8 gap, GEMV | 1.940x (1.73–2.11) | 1.967x (1.79–2.33) | widens |
| int8 mm / 2-bit gv | 1.332x (1.23–1.60) | 1.371x (1.17–1.45) | — |
| B1 gv / B4 mm per step, 2-bit | 0.945 | 0.834 | — |
| B1 gv / B4 mm per step, int8 | 0.761 | 0.674 | — |

Single-thread micro (one expanded f32 tile, T 256):

| tile | bytes | mm µs (s1/s2) | gv µs (s1/s2) | mm/gv |
|---|---|---|---|---|
| 256 x 2560 | 2.50 MiB | 59.5 / 60.1 | 58.2 / 58.6 | 1.02 / 1.03x |
| 256 x 6912 | 6.75 MiB | 222.1 / 218.2 | 220.0 / 216.9 | 1.01 / 1.01x |

Weights per layer: 2-bit 16.65 MiB, int8 66.34 MiB (3.984x).

## Pair files this wave

The quant pairs were idle. The latest ledger is Quant Chief's 14:05 MT wave (JuniorHome docs/ledger/2026-11-23_1405.md, close 61e541f414b64463). It shows no Trit Pack, Winsor, Side-Commit or Handshake reports or shas, and ROADMAP.md / BETA_TO_OS.md exhausted. LLC pushed f25609c and a49673d into JuniorHome scripts/, which are not paths touched here. There are no new pair files under /workspace to link.

## Next question

With GEMV, int8 B 1 is ~5–6 ms and 2-bit is ~10–12 ms per layer, so the remaining gap is the LUT take + int8->f32 cast. At B 1 with GEMV, what is the per-op split (take / cast / dot) for all-2-bit? Does a smaller tile (T 64 / 128, keeping each worker's u32 + f32 scratch L2-resident) or a fused take into a pre-cast f32 LUT (4 f32 per byte, skipping the int8 view cast) bring all2_gv within 1.3x of all8_gv?

close: 3c632d82fec3a76d
