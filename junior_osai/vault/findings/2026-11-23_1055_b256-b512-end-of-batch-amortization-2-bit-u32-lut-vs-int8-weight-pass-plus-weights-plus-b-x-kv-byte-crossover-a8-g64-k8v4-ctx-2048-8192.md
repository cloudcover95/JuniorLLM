---
local: true
date: 2026-11-23
wave: "2026-11-23 10:55 (fired 10:46 MT, ran ~10:47–11:05 MT; Quant Chief 10:05 wave: both pairs no gap, no eligible item; quant builders idle)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (B 256/512 int8 gap stays 1.175/1.112x / 1.212/1.088x, not reliably below 1.1x; and the weights+KV memory edge already falls under 1.5x at B > 31 (A8) / 39 (K8V4) at ctx 2048 and B > 8 / 10 at ctx 8192, so no B gives both)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, weights, u32-lut, batching, b256, b512, kv-bytes, crossover, numpy, not-viable]
---

# End of batch amortization: at B 256 / 512 the 2-bit u32-LUT weight pass is still 1.175/1.112x / 1.212/1.088x slower than int8, and the ternary weights+KV memory edge is gone (< 1.5x) beyond B 31–39 at ctx 2048 (B 8–10 at ctx 8192): there is no batch size where 2-bit wins on both speed parity and memory in this stack

The 05:55 note ([[2026-11-23_0555_batch-amortization-of-shared-2-bit-u32-lut-weight-pass-b8-b32-b64-b128-take-cast-matmul-split-vs-int8-and-f32-resident]]) found batching amortizes the shared 2-bit unpack but int8 stayed 1.19/1.27x faster at B 128, and asked: at B 256 (matmul share past 80%) does the 2-bit/int8 weight-pass gap drop below 1.1x, and at ctx 2048 with A8 g64 KV per stream, at what B does total bytes (weights + KV) fall below 1.5x under int8?

Answers:
- **Speed gap does not close reliably:** int8 over 2-bit is 0.986/1.186x at B 128, 1.175/1.112x at B 256 and 1.212/1.088x at B 512. Two of six seed×B cells are under 1.1x: seed 1 B 128 at 0.986 (per-round range 0.80–1.44, noisy) and seed 2 B 512 at 1.088. The other four are 1.11–1.21x, and per-round ranges span ~0.8–1.5x, so the honest summary is "int8 ~1.1–1.2x faster, flat from B 128 on". The f32-resident no-unpack ceiling is 1.314/1.362x / 1.297/1.522x / 1.405/1.094x.
- **Amortization flattens:** aggregate base256 tok/s B 128 -> 256 is 1.29/1.21x and B 128 -> 512 1.69/1.42x (5883/6003 tok/s at B 512), well down from 2.42/1.93x for B 32 -> 128. rowT128 stays a small, noisy gain at B >= 256 (1.09–1.22x).
- **Memory crossover (the new number):** per stream per layer, A8 g64 KV at ctx 2048 is 2.656 MiB and K8V4 (K int8 g64, V nibble g32) 2.109 MiB, measured from real quantized arrays. With 2-bit weights at 16.65 MiB vs int8 at 66.34 MiB per layer, the (int8 + B·KV)/(2-bit + B·KV) ratio drops below 1.5x for B > 31.1 (A8) / 39.2 (K8V4) at ctx 2048, and for B > 7.8 / 9.8 at ctx 8192. At B 128 / ctx 2048 it is 1.139x (A8) / 1.173x (K8V4). The ratio is the same per layer and for the whole model, because weights and KV both scale with layer count.
- **The trade-off is disjoint:** ternary keeps >= 1.5x total memory only at B <= 31 (A8) / 39 (K8V4) at ctx 2048, where the 05:55 note measured int8 1.32–1.79x faster. Where int8's speed lead shrinks to ~1.1–1.2x (B >= 128), the memory edge is down to 1.04–1.17x. Model-wide constants make this worse for 2-bit: the shared per-worker unpack scratch at T 256 is 67.5 MiB (u32 13.5 + f32 54.0), and int8 needs only the f32 part. Activation outputs reach 27 MiB at B 512 for both paths.
- **Error:** the 2-bit path vs the float64 trit reference has max abs <= 2.3e-06 and cos 1.00000000 at every B. The int8 control's cos ~0.887 reflects different weights (int8 absmax of the f32 W), as in the previous notes. KV quant cos: A8 k/v 0.99998, K8V4 v 0.99533 (random-normal history; the real-history K8V4 numbers are in the K8V4 notes).
- **Verdict: not-viable.** Neither half of the question passes: the gap does not reliably go below 1.1x at B 256/512, and memory falls below the 1.5x bar long before the speed gap narrows. The ternary memory win in this stack is a small-B (B <~ 30 at ctx 2048, B <~ 8 at ctx 8192) / single-user property.

## Method

- Weights: base.build() from /workspace/bench1455_1118 (bitnetd layer: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits). par.prep() fuses q|k|v and gate|up into resident 2-bit PR rows with per-row gamma; par.prep8() is the int8 control; f32res is the float64 trit*gamma reference cast to f32. Same variants and pool as the 05:55 note.
- Timed op: one full weight pass of a decode step (qkv 2560->3840, o 2560->2560, gate|up 2560->13824, down 6912->2560) on the W = 8 thread pool, OPENBLAS_NUM_THREADS=1, at B = 128 / 256 / 512.
- 5 rounds x 5 reps (2 warm-up) per variant per B, variant order rotated each round. base/this is the median of per-round base medians divided by the median of per-round variant medians, with the per-round range shown. Two input seeds (XSEED 1, 2) on an 8-core box.
- KV bytes: one stream's K and V history (5 KV heads x ctx x 128) quantized with the real grouping (A8: int8 g64 + f32 scale for K and V; K8V4: int8 g64 K, signed 4-bit g32 V nibble-packed two per byte + f32 scales). Bytes are nbytes of the resulting arrays. The crossover B solves (W8 + B·kv)/(W2 + B·kv) = 1.5. Seed-independent.
- No installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench1055_1123 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 xover.py run$s.json > run$s.txt 2>&1; done

Script (xover.py):

    #!/usr/bin/env python3
    """Where batching ends, 10:55 11/23. (a) Full decode-step weight pass (qkv 2560->3840, o, gate|up 2560->13824, down 6912->2560)
    on the W=8 pool at B = 128 / 256 / 512 streams sharing one unpack: base256 (current u32-LUT 2-bit, par.fmm T256), rowT128,
    int8_256 (int8 resident, cast only), f32res (f32 trit*gamma resident, matmul only). Does the 2-bit/int8 gap drop below 1.1x?
    (b) Byte crossover: per-layer weights + B per-stream KV caches at ctx 2048 / 8192 (A8 g64 measured from real q8 arrays, and
    K8V4 = K int8 g64 + V nibble g32), 2-bit vs int8 weights. At what B does (int8 total)/(2-bit total) fall below 1.5x?
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=... python3 xover.py out.json"""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import D, F, KVH, HD
    W = 8; ROUNDS, REPS, WARM = 5, 5, 2
    BS = [128, 256, 512]
    base.build(); WB = par.prep(); WB8 = par.prep8()
    SITES = [("qkv", D), ("o", D), ("gu", D), ("down", F)]
    def wref(key): return np.concatenate([base.PW[n]["Wt"].astype(np.float64) for n in par.FUSE[key]])
    WF = {k: np.ascontiguousarray(wref(k).astype(np.float32)) for k, _ in SITES}; WBF = sum(a.nbytes for a in WF.values())
    WR = {k: wref(k) for k, _ in SITES}
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT"): base.PW[nm].pop(f, None)
    def run_f(Wf, x, y, r0, r1, T):
        for o0 in range(r0, r1, T):
            n = min(T, r1 - o0); np.matmul(x, Wf[o0:o0+n].T, out=y[:, o0:o0+n])
    def fmm_f(key, x, T=256):
        Wf = WF[key]; out = Wf.shape[0]; y = np.empty((x.shape[0], out), np.float32); nt = -(-out // T); per = -(-nt // W)
        for f in [par.pool(W).submit(run_f, Wf, x, y, w*per*T, min(out, (w+1)*per*T), T) for w in range(W) if w*per*T < out]: f.result()
        return y
    V = {"base256": lambda k, x: np.concatenate(par.fmm(k, x, 256, W), axis=1),
         "rowT128": lambda k, x: np.concatenate(par.fmm(k, x, 128, W), axis=1),
         "int8_256": lambda k, x: np.concatenate(par.fmm8(k, x, 256, W), axis=1),
         "f32res": lambda k, x: fmm_f(k, x)}
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    # ---- KV bytes, measured from real quantized arrays for one stream ----
    def qg(x, gs, bits):
        g = x.reshape(*x.shape[:-1], -1, gs); qm = 2 ** (bits - 1) - 1; m = (np.abs(g).max(-1) / qm + 1e-12).astype(np.float32)
        c = np.clip(np.rint(g / m[..., None]), -qm, qm).astype(np.int8).reshape(x.shape)
        if bits == 4:  # nibble pack two codes per byte
            u = (c.astype(np.int16) & 0xF).astype(np.uint8).reshape(*x.shape[:-1], -1, 2); c = (u[..., 0] | (u[..., 1] << 4)).astype(np.uint8)
        return c, m
    def deq(c, m, gs, bits, shape):
        if bits == 4:
            lo = (c & 0xF).astype(np.int8); hi = (c >> 4).astype(np.int8); lo[lo > 7] -= 16; hi[hi > 7] -= 16
            c = np.stack([lo, hi], -1).reshape(shape)
        return (c.reshape(*shape[:-1], -1, gs).astype(np.float32) * m[..., None]).reshape(shape)
    def kv_stream(ctx, rng):
        k = rng.standard_normal((KVH, ctx, HD)).astype(np.float32); v = rng.standard_normal((KVH, ctx, HD)).astype(np.float32)
        out = {}
        for nm, (kb, kg, vb, vg) in {"a8g64": (8, 64, 8, 64), "k8v4": (8, 64, 4, 32)}.items():
            kc, km = qg(k, kg, kb); vc, vm = qg(v, vg, vb)
            nb = kc.nbytes + km.nbytes + vc.nbytes + vm.nbytes
            ck = cos(deq(kc, km, kg, kb, k.shape), k); cv = cos(deq(vc, vm, vg, vb, v.shape), v)
            out[nm] = dict(bytes=int(nb), cos_k=ck, cos_v=cv)
        out["f32"] = dict(bytes=int(k.nbytes + v.nbytes), cos_k=1.0, cos_v=1.0)
        return out
    def main():
        seed = int(os.environ.get("XSEED", "1")); rng = np.random.default_rng(seed)
        out = {"xseed": seed, "w_bytes_2bit": WB, "w_bytes_int8": WB8, "w_bytes_f32": WBF, "B": {}, "kv": {}}
        print(f"xseed {seed} | resident 2-bit {WB/2**20:.2f} MiB int8 {WB8/2**20:.2f} MiB ({WB8/WB:.3f}x) f32 {WBF/2**20:.2f} MiB | W {W} rounds {ROUNDS} x reps {REPS}", flush=True)
        for ctx in (2048, 8192):
            kv = kv_stream(ctx, rng); out["kv"][str(ctx)] = kv
            for nm, d in kv.items():
                if nm == "f32": continue
                b = d["bytes"]; bx = (WB8 - 1.5 * WB) / (0.5 * b)
                rows = {B: (WB8 + B*b) / (WB + B*b) for B in (1, 8, 16, 32, 64, 128, 256, 512)}
                d["xover_B"] = bx; d["ratio"] = rows
                print(f"ctx {ctx} {nm}: {b/2**20:.3f} MiB/stream/layer cos k {d['cos_k']:.5f} v {d['cos_v']:.5f} | int8/2bit total <1.5x for B > {bx:.1f} | " +
                      " ".join(f"B{B}:{r:.3f}" for B, r in rows.items()), flush=True)
        for B in BS:
            X = {k: base.rms(rng.standard_normal((B, i)).astype(np.float32)) for k, i in SITES}
            REF = {k: X[k].astype(np.float64) @ WR[k].T for k, _ in SITES}
            act = max(B * WF[k].shape[0] * 4 for k, _ in SITES)
            res = {}
            for nm, fn in V.items():
                Y = {k: fn(k, X[k]) for k, _ in SITES}
                res[nm] = dict(t=[], mab_vs_f64=max(float(np.abs(Y[k]-REF[k]).max()) for k, _ in SITES), cos_f64=min(cos(Y[k], REF[k]) for k, _ in SITES))
            del Y
            names = list(V)
            for rd in range(ROUNDS):
                for nm in names[rd % len(names):] + names[:rd % len(names)]:
                    fn = V[nm]
                    for r in range(WARM + REPS):
                        t0 = time.perf_counter()
                        for k, _ in SITES: fn(k, X[k])
                        dt = (time.perf_counter() - t0) * 1e3
                        if r >= WARM: res[nm]["t"].append(dt)
            bm = [float(np.median(res["base256"]["t"][r*REPS:(r+1)*REPS])) for r in range(ROUNDS)]
            for nm in names:
                d = res[nm]; T = np.array(d["t"]); rm = [float(np.median(T[r*REPS:(r+1)*REPS])) for r in range(ROUNDS)]
                gap = [a/b for a, b in zip(bm, rm)]
                d.update(med=float(np.median(T)), lo=float(T.min()), hi=float(T.max()), iqr=float(np.percentile(T, 75)-np.percentile(T, 25)),
                         tps=B*1e3/float(np.median(T)), base_over=float(np.median(bm)/np.median(rm)), bo_lo=min(gap), bo_hi=max(gap), n=len(T))
                print(f"B{B:<4d}{nm:9s} {d['med']:7.2f} ms ({d['lo']:.2f}-{d['hi']:.2f}) iqr {d['iqr']:5.2f} | {d['tps']:8.0f} tok/s | base/this {d['base_over']:.3f}x (rounds {d['bo_lo']:.2f}-{d['bo_hi']:.2f}) | mab vs f64 {d['mab_vs_f64']:.1e} cos {d['cos_f64']:.8f}", flush=True)
            res["act_peak_bytes"] = act; res["scratch_bytes"] = par.live_scratch_bytes()
            print(f"B{B:<4d}peak activation out {act/2**20:.2f} MiB, unpack scratch {res['scratch_bytes']/2**20:.2f} MiB", flush=True)
            out["B"][str(B)] = res
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Results: full weight pass, ms median (min–max) iqr; seed 1 / seed 2

| B | variant | seed 1 ms | seed 2 ms | agg tok/s s1 / s2 | base/this (per-round) s1 / s2 | max abs vs f64 | min cos vs f64 |
|---|---|---|---|---|---|---|---|
| 128 | base256 | 36.74 (31.51–46.88) iqr 3.48 | 30.23 (27.67–39.34) iqr 3.15 | 3484 / 4235 | 1.000 (1.00–1.00) / 1.000 (1.00–1.00) | 2.0e-06 | 1.00000000 |
| 128 | rowT128 | 37.66 (31.01–44.70) iqr 6.32 | 28.33 (25.10–40.84) iqr 5.57 | 3399 / 4519 | 0.882 (0.84–1.11) / 1.073 (0.83–1.17) | 2.0e-06 | 1.00000000 |
| 128 | int8_256 | 36.39 (26.63–59.14) iqr 9.76 | 25.49 (23.23–36.68) iqr 5.10 | 3518 / 5021 | 0.986 (0.80–1.44) / 1.186 (0.95–1.31) | 4.1e+00 | 0.88673008 |
| 128 | f32res | 27.89 (22.85–39.89) iqr 3.96 | 22.16 (18.43–29.11) iqr 2.56 | 4589 / 5777 | 1.314 (1.10–1.50) / 1.362 (1.21–1.53) | 2.0e-06 | 1.00000000 |
| 256 | base256 | 57.10 (48.75–68.07) iqr 8.30 | 50.09 (44.59–65.98) iqr 13.50 | 4484 / 5111 | 1.000 (1.00–1.00) / 1.000 (1.00–1.00) | 2.0e-06 | 1.00000000 |
| 256 | rowT128 | 49.35 (44.87–71.39) iqr 4.83 | 49.20 (41.92–61.77) iqr 9.31 | 5187 / 5204 | 1.106 (0.99–1.26) / 1.223 (1.00–1.24) | 2.0e-06 | 1.00000000 |
| 256 | int8_256 | 46.28 (41.31–70.74) iqr 10.34 | 50.75 (37.92–57.77) iqr 6.22 | 5532 / 5044 | 1.175 (0.94–1.35) / 1.112 (0.84–1.22) | 4.3e+00 | 0.88640673 |
| 256 | f32res | 42.16 (35.00–49.72) iqr 4.54 | 38.56 (33.16–48.10) iqr 7.72 | 6072 / 6639 | 1.297 (1.12–1.49) / 1.522 (1.25–1.69) | 1.9e-06 | 1.00000000 |
| 512 | base256 | 87.03 (74.38–114.52) iqr 14.48 | 85.29 (76.97–119.09) iqr 16.90 | 5883 / 6003 | 1.000 (1.00–1.00) / 1.000 (1.00–1.00) | 2.3e-06 | 1.00000000 |
| 512 | rowT128 | 81.29 (74.35–91.17) iqr 6.74 | 79.83 (73.80–108.09) iqr 5.34 | 6299 / 6414 | 1.146 (0.93–1.38) / 1.089 (0.92–1.50) | 2.3e-06 | 1.00000000 |
| 512 | int8_256 | 76.48 (73.49–86.12) iqr 5.16 | 78.11 (71.15–91.25) iqr 8.73 | 6694 / 6555 | 1.212 (0.98–1.32) / 1.088 (1.06–1.53) | 4.3e+00 | 0.88681890 |
| 512 | f32res | 65.96 (59.48–89.96) iqr 7.95 | 77.39 (61.97–111.86) iqr 13.47 | 7762 / 6616 | 1.405 (1.12–1.59) / 1.094 (1.09–1.32) | 2.1e-06 | 1.00000000 |

Variants: base256 = current u32-LUT 2-bit path (T 256); rowT128 = same, row tile 128; int8_256 = int8 resident, cast only (4x bytes); f32res = f32 resident trit*gamma, matmul only (16x bytes). Resident weight bytes per layer: 2-bit 16.65 MiB, int8 66.34 MiB (3.984x), f32 265.00 MiB.

## Results: weights + B·KV byte ratio, int8 total / 2-bit total (per layer = whole model)

| ctx | KV fmt | KV MiB/stream/layer | cos k / v | < 1.5x when | B1 | B8 | B16 | B32 | B64 | B128 | B256 | B512 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | a8g64 | 2.656 | 0.99998 / 0.99998 | B > 31.1 | 3.574 | 2.311 | 1.840 | 1.489 | 1.266 | 1.139 | 1.071 | 1.036 |
| 2048 | k8v4 | 2.109 | 0.99998 / 0.99533 | B > 39.2 | 3.649 | 2.482 | 1.986 | 1.590 | 1.328 | 1.173 | 1.089 | 1.045 |
| 8192 | a8g64 | 10.625 | 0.99998 / 0.99998 | B > 7.8 | 2.822 | 1.489 | 1.266 | 1.139 | 1.071 | 1.036 | 1.018 | 1.009 |
| 8192 | k8v4 | 8.438 | 0.99998 / 0.99533 | B > 9.8 | 2.981 | 1.590 | 1.328 | 1.173 | 1.089 | 1.045 | 1.023 | 1.011 |

## Pair files this wave

Quant pairs idle this wave. Quant Chief's 10:05 MT assignment was "no gap" for both pairs (roadmap/beta blobs unchanged, no eligible item), and no Trit Pack or Handshake pair files are new, so there is nothing to link.

## Next question

Speed parity at large B is out of reach in numpy, and the memory edge only exists at small B. So the open question is the small-B single-user path, where weight bytes dominate: at B = 1 / 4 and ctx 2048, does keeping only the FFN (gate|up + down, ~77% of weight bytes) at 2-bit and holding qkv/o at int8 recover >= 1.3x of the full-2-bit-vs-int8 speed gap while keeping >= 1.5x total memory vs all-int8 (mixed-precision site split)?

close: 2fd3c9c7df5dcb0e
