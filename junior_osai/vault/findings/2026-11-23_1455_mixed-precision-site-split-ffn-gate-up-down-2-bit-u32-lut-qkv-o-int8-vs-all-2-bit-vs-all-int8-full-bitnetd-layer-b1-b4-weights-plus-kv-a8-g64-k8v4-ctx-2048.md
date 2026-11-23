---
local: true
date: 2026-11-23
wave: "2026-11-23 14:55 (fired ~14:40 MT, ran ~14:41–14:55 MT; Quant Chief 10:05 wave: both pairs no gap, no eligible item; quant builders idle)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "not-viable (mixed split keeps 1.97–2.25x total memory vs all-int8, but is only 1.084/1.087x (B 1) / 1.038/1.088x (B 4) faster than all-2-bit and recovers 0.29/0.29 / 0.10/0.22 of the all-2-bit-vs-int8 gap, far from the >= 1.3x bar)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, weights, u32-lut, mixed-precision, site-split, ffn, int8, b1, b4, kv-bytes, k8v4, numpy, not-viable]
---

# Mixed-precision site split (FFN gate|up + down at 2-bit u32-LUT, qkv/o at int8): 1.97–2.25x total memory vs all-int8 at ctx 2048, but only 1.084/1.087x (B 1) / 1.038/1.088x (B 4) faster than all-2-bit, recovering 0.29/0.29 / 0.10/0.22 of the gap to int8; no reading of the 1.3x bar passes

The 10:55 note ([[2026-11-23_1055_b256-b512-end-of-batch-amortization-2-bit-u32-lut-vs-int8-weight-pass-plus-weights-plus-b-x-kv-byte-crossover-a8-g64-k8v4-ctx-2048-8192]]) found that the ternary weights+KV memory edge only exists at small B, and asked: at B = 1 / 4 and ctx 2048, does keeping only the FFN (gate|up + down, ~77% of weight bytes) at 2-bit and holding qkv/o at int8 recover >= 1.3x of the full-2-bit-vs-int8 speed gap while keeping >= 1.5x total memory vs all-int8?

Answers:
- **Speed: no.** On the full layer, all-2-bit / mixed / all-int8 run at 19.23/19.35 / 17.75/17.90 / 14.14/14.26 ms at B 1 and 12.56/13.53 / 12.00/12.47 / 8.17/8.70 ms at B 4 (seed 1/seed 2). The all-2-bit-vs-int8 gap is 1.360/1.374x (B 1) and 1.549/1.579x (B 4). Mixed over all-2-bit is 1.084/1.087x and 1.038/1.088x, which recovers 0.29/0.29 (B 1) and 0.10/0.22 (B 4) of the gap; per-round ranges are wide (about −0.04 to 0.58). Both readings of "recover >= 1.3x" fail: mixed is nowhere near 1.3x faster than all-2-bit, and it does not even close a third of the gap. That fits a bytes-proportional cost: the int8 sites (qkv + o) are only 23.6% of int8 weight bytes, so moving them off 2-bit can win back at most about that share of the unpack overhead.
- **Memory: yes, but all-2-bit already has more.** Mixed weights are 28.37 MiB/layer (vs 16.65 all-2-bit and 66.34 all-int8). With measured per-stream KV (A8 g64 2.656, K8V4 2.109 MiB/stream/layer at ctx 2048), all-int8/mixed total is 1.974–2.246x at B 1–4 (1.910–2.146x for a 30-layer model with the per-worker unpack scratch counted), so mixed clears the 1.5x bar. All-2-bit is 2.822–3.649x, so the split gives up about a third of the memory win for a ~4–9% speedup.
- **Error:** all-2-bit vs the float64 trit reference layer has max abs <= 1.2e-06, cos 1.00000000. Every config matches its own float64 weights (max abs <= 4.4e-06). Against the trit reference, mixed has cos 0.772–0.783 and all-int8 has 0.660–0.679. That is because the int8 control is absmax int8 of the original f32 W, a different model, not quant noise on the trits, as in previous notes. So a real mixed deployment would be a different model at qkv/o, and its accuracy cost can't be measured with random weights.
- **Side observation:** B 1 is slower per step than B 4 (about 19 vs 12.5–13.5 ms) for all three configs. It shows up in both seeds and does not depend on the weight format, so the likely cause (not isolated here) is the (1, in) @ (T, in).T matmul path on the f32 tile.
- **Verdict: not-viable.** The memory half passes, but the speed half, the point of the split, fails by a wide margin: 1.084/1.087x / 1.038/1.088x against a 1.3x bar.

## Method

- Weights: base.build() from /workspace/bench1455_1118 (bitnetd layer: hidden 2560, 20 q heads x 128, 5 KV heads, FFN 6912, absmean trits). par.prep() gives resident 2-bit PR rows per fused site (q|k|v, gate|up) with per-row gamma. par.prep8() is the int8 control (int8 absmax of the f32 W). Same pool, tiles and paths as the 05:55 and 10:55 notes.
- Configs: all2 = every site par.fmm (u32-LUT take -> int8 view -> f32 tile -> matmul, T 256); all8 = every site par.fmm8 (int8 resident cast to f32 tile, T 256); mixed = qkv and o via fmm8, gate|up and down via fmm.
- Timed op: one full layer as par.layer (rms, fused qkv, attention stub = v repeated over GQA groups, o, residual, rms, fused gate|up, silu*up, down, residual) on the W = 8 thread pool, OPENBLAS_NUM_THREADS=1, B = 1 and 4. 7 rounds x 9 reps (3 warm-up) per config, config order rotated each round. Gap and speedup use the median of per-round medians, with per-round min–max. Gap fraction recovered = (t_all2 − t_mixed)/(t_all2 − t_all8). Two input seeds (XSEED 1, 2) on an 8-core box.
- Error: float64 reference layer with trit*gamma weights at every site (trit ref), and float64 reference with each config's own weights (q8*m8 at int8 sites).
- Bytes: resident nbytes per site (2-bit PR + gamma; int8 Q + scale). KV is one stream's K/V (5 heads x 2048 x 128) quantized for real (A8: int8 g64 + f32 scale; K8V4: int8 g64 K + signed nibble g32 V), nbytes. Total = weights + B·KV per layer, which also gives the whole-model ratio. The "+scratch" column is 30 layers plus per-worker scratch at W 8 / T 256 (2-bit and mixed u32+f32 67.5 MiB, int8 f32 only 54.0 MiB).
- No installs, no network, no model download, nothing bound.

Command:

    cd /workspace/bench1455_1123 && for s in 1 2; do OPENBLAS_NUM_THREADS=1 XSEED=$s python3 split.py run$s.json > run$s.txt 2>&1; done

Script (split.py):

    #!/usr/bin/env python3
    """Mixed-precision site split, 14:55 11/23. Full bitnetd-style layer (rms -> fused qkv 2560->3840 -> o 2560->2560 -> rms ->
    fused gate|up 2560->13824 -> silu*up -> down 6912->2560, par.layer's attention stub) at B = 1 / 4 on the W=8 pool, T 256:
    all2 (every site 2-bit u32-LUT, par.fmm), all8 (every site int8 resident, par.fmm8), mixed (qkv/o int8, gate|up + down 2-bit).
    Does mixed recover >= 1.3x of the all2-vs-all8 speed gap while weights + B x KV (ctx 2048, A8 g64 / K8V4) stay >= 1.5x
    under all8? Error vs the float64 trit reference layer, and vs each config's own float64 weights.
    Usage: OPENBLAS_NUM_THREADS=1 XSEED=... python3 split.py out.json"""
    import json, os, sys, time
    import numpy as np
    sys.path.insert(0, "/workspace/bench1455_1118")
    import base, par
    from base import D, F, KVH, QH, HD, rms, silu
    W, T = 8, 256; ROUNDS, REPS, WARM = 7, 9, 3
    BS = [1, 4]; CTX = 2048; NL = 30   # NL only for the model-wide scratch line
    base.build(); par.prep(); par.prep8()
    SITES = ["qkv", "o", "gu", "down"]
    B2 = {k: par.FP[k]["PR"].nbytes + par.FP[k]["g"].nbytes for k in SITES}
    B8 = {k: par.F8[k]["Q"].nbytes + par.F8[k]["g"].nbytes for k in SITES}
    CFG = {"all2": {k: 2 for k in SITES}, "mixed": {"qkv": 8, "o": 8, "gu": 2, "down": 2}, "all8": {k: 8 for k in SITES}}
    WBY = {c: sum(B2[k] if b == 2 else B8[k] for k, b in m.items()) for c, m in CFG.items()}
    def wf64(k, bits):
        if bits == 2: return np.concatenate([base.PW[n]["Wt"].astype(np.float64) for n in par.FUSE[k]])
        return np.concatenate([base.PW[n]["q8"].astype(np.float64) * float(base.PW[n]["m8"]) for n in par.FUSE[k]])
    WR = {(k, b): wf64(k, b) for k in SITES for b in (2, 8)}
    for nm in base.PW:
        for f in ("Wd", "wm", "ws", "wm8", "ws8", "PT", "Wt"): base.PW[nm].pop(f, None)
    def mm(k, bits, x): return par.fmm(k, x, T, W) if bits == 2 else par.fmm8(k, x, T, W)
    def layer(x, m):
        h1 = rms(x); q, kk, v = mm("qkv", m["qkv"], h1)
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        o = mm("o", m["o"], att)[0]; x2 = x + o; h2 = rms(x2)
        g, u = mm("gu", m["gu"], h2); z = (silu(g) * u).astype(np.float32)
        return x2 + mm("down", m["down"], z)[0]
    def layer_ref(x, m):   # float64 everywhere, weights per m (2 -> trit*gamma, 8 -> q8*m8)
        x = x.astype(np.float64); r = lambda a: a / np.sqrt(np.mean(a ** 2, axis=-1, keepdims=True) + 1e-6)
        sg = lambda a: a / (1 + np.exp(-a))
        qkv = r(x) @ WR[("qkv", m["qkv"])].T; v = qkv[:, D + KVH * HD:]
        att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
        x2 = x + att @ WR[("o", m["o"])].T; gu = r(x2) @ WR[("gu", m["gu"])].T
        z = sg(gu[:, :F]) * gu[:, F:]
        return x2 + z @ WR[("down", m["down"])].T
    def cos(a, b): a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    def qg(x, gs, bits):
        g = x.reshape(*x.shape[:-1], -1, gs); qm = 2 ** (bits - 1) - 1; s = (np.abs(g).max(-1) / qm + 1e-12).astype(np.float32)
        c = np.clip(np.rint(g / s[..., None]), -qm, qm).astype(np.int8).reshape(x.shape)
        if bits == 4:
            u = (c.astype(np.int16) & 0xF).astype(np.uint8).reshape(*x.shape[:-1], -1, 2); c = (u[..., 0] | (u[..., 1] << 4)).astype(np.uint8)
        return c, s
    def kv_bytes(rng):
        k = rng.standard_normal((KVH, CTX, HD)).astype(np.float32); v = rng.standard_normal((KVH, CTX, HD)).astype(np.float32); out = {}
        for nm, (kb, kg, vb, vg) in {"a8g64": (8, 64, 8, 64), "k8v4": (8, 64, 4, 32)}.items():
            kc, km = qg(k, kg, kb); vc, vm = qg(v, vg, vb); out[nm] = int(kc.nbytes + km.nbytes + vc.nbytes + vm.nbytes)
        return out
    def main():
        seed = int(os.environ.get("XSEED", "1")); rng = np.random.default_rng(seed); MiB = 2 ** 20
        out = {"xseed": seed, "site_bytes_2bit": B2, "site_bytes_int8": B8, "w_bytes": WBY, "B": {}, "mem": {}}
        print(f"xseed {seed} | W {W} T {T} rounds {ROUNDS} x reps {REPS} | site bytes MiB 2bit " + " ".join(f"{k}:{B2[k]/MiB:.2f}" for k in SITES) +
              " | int8 " + " ".join(f"{k}:{B8[k]/MiB:.2f}" for k in SITES), flush=True)
        ffn = (B8["gu"] + B8["down"]) / sum(B8.values())
        print(f"FFN share of int8 weight bytes {ffn:.3f} | layer weights MiB " + " ".join(f"{c}:{b/MiB:.2f}" for c, b in WBY.items()), flush=True)
        kv = kv_bytes(rng); out["kv"] = kv; out["ffn_share"] = ffn
        sc2 = par.scratch_bytes(W, T); sc8 = W * T * F * 4
        for nm, b in kv.items():
            for B in BS:
                tot = {c: WBY[c] + B * b for c in CFG}
                tsc = {c: NL * tot[c] + (sc8 if c == "all8" else sc2) for c in CFG}
                out["mem"][f"{nm}_B{B}"] = dict(kv=b, tot=tot, r=tot["all8"] / tot["mixed"], r2=tot["all8"] / tot["all2"],
                                                r_sc=tsc["all8"] / tsc["mixed"], r2_sc=tsc["all8"] / tsc["all2"])
                d = out["mem"][f"{nm}_B{B}"]
                print(f"mem ctx {CTX} {nm} B{B}: kv {b/MiB:.3f} MiB/stream/layer | total/layer all2 {tot['all2']/MiB:.2f} mixed {tot['mixed']/MiB:.2f} all8 {tot['all8']/MiB:.2f} MiB"
                      f" | all8/mixed {d['r']:.3f}x all8/all2 {d['r2']:.3f}x | {NL}L+scratch all8/mixed {d['r_sc']:.3f}x all8/all2 {d['r2_sc']:.3f}x", flush=True)
        for B in BS:
            x = (rng.standard_normal((B, D)) * 0.5).astype(np.float32)
            REF2 = layer_ref(x, CFG["all2"]); res = {}
            for c, m in CFG.items():
                y = layer(x, m); own = layer_ref(x, m)
                res[c] = dict(t=[], mab_trit=float(np.abs(y - REF2).max()), cos_trit=cos(y, REF2), mab_own=float(np.abs(y - own).max()), cos_own=cos(y, own))
            names = list(CFG)
            for rd in range(ROUNDS):
                for c in names[rd % 3:] + names[:rd % 3]:
                    for r in range(WARM + REPS):
                        t0 = time.perf_counter(); layer(x, CFG[c]); dt = (time.perf_counter() - t0) * 1e3
                        if r >= WARM: res[c]["t"].append(dt)
            rm = {c: [float(np.median(res[c]["t"][r*REPS:(r+1)*REPS])) for r in range(ROUNDS)] for c in names}
            for c in names:
                d = res[c]; Tm = np.array(d["t"])
                d.update(med=float(np.median(Tm)), lo=float(Tm.min()), hi=float(Tm.max()), iqr=float(np.percentile(Tm, 75) - np.percentile(Tm, 25)),
                         tps=B * 1e3 / float(np.median(Tm)), n=len(Tm))
            t2, tm, t8 = (float(np.median(rm[c])) for c in names)
            sp = [a / b for a, b in zip(rm["all2"], rm["mixed"])]; gp = [a / b for a, b in zip(rm["all2"], rm["all8"])]
            fr = [(a - b) / (a - e) if a > e else float("nan") for a, b, e in zip(rm["all2"], rm["mixed"], rm["all8"])]
            res["summary"] = dict(t2=t2, tm=tm, t8=t8, gap=t2 / t8, gap_lo=min(gp), gap_hi=max(gp), speedup=t2 / tm, sp_lo=min(sp), sp_hi=max(sp),
                                  frac=(t2 - tm) / (t2 - t8) if t2 > t8 else float("nan"), fr_lo=float(np.nanmin(fr)), fr_hi=float(np.nanmax(fr)))
            for c in names:
                d = res[c]
                print(f"B{B} {c:6s} {d['med']:7.3f} ms ({d['lo']:.3f}-{d['hi']:.3f}) iqr {d['iqr']:.3f} | {d['tps']:7.1f} tok/s | vs trit f64 mab {d['mab_trit']:.1e} cos {d['cos_trit']:.8f} | vs own f64 mab {d['mab_own']:.1e} cos {d['cos_own']:.8f}", flush=True)
            s = res["summary"]
            print(f"B{B} gap all2/all8 {s['gap']:.3f}x (rounds {s['gap_lo']:.2f}-{s['gap_hi']:.2f}) | all2/mixed {s['speedup']:.3f}x (rounds {s['sp_lo']:.2f}-{s['sp_hi']:.2f}) | gap fraction recovered {s['frac']:.3f} (rounds {s['fr_lo']:.2f}-{s['fr_hi']:.2f})", flush=True)
            out["B"][str(B)] = res
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    main()

## Results: full layer, ms median (min–max) iqr; seed 1 / seed 2

| B | config | seed 1 ms | seed 2 ms | tok/s s1 / s2 | vs trit f64 max abs / min cos | vs own f64 max abs / min cos |
|---|---|---|---|---|---|---|
| 1 | all2 | 19.23 (17.97–21.63) iqr 0.83 | 19.35 (17.68–25.27) iqr 1.18 | 52.0 / 51.7 | 9.4e-07 / 1.0000 | 9.4e-07 / 1.00000000 |
| 1 | mixed | 17.75 (16.38–31.43) iqr 0.86 | 17.90 (16.50–25.00) iqr 1.22 | 56.3 / 55.9 | 2.8e+00 / 0.7742 | 1.9e-06 / 1.00000000 |
| 1 | all8 | 14.14 (13.27–18.52) iqr 0.93 | 14.26 (12.74–18.77) iqr 1.06 | 70.7 / 70.1 | 4.5e+00 / 0.6621 | 3.9e-06 / 1.00000000 |
| 4 | all2 | 12.56 (11.21–23.91) iqr 1.49 | 13.53 (12.10–16.58) iqr 1.10 | 318.4 / 295.6 | 1.2e-06 / 1.0000 | 1.2e-06 / 1.00000000 |
| 4 | mixed | 12.00 (10.59–19.75) iqr 0.90 | 12.47 (10.80–28.05) iqr 1.09 | 333.3 / 320.7 | 3.1e+00 / 0.7718 | 2.7e-06 / 1.00000000 |
| 4 | all8 | 8.17 (7.01–11.97) iqr 0.93 | 8.70 (7.33–13.09) iqr 0.88 | 489.4 / 459.7 | 4.9e+00 / 0.6605 | 4.4e-06 / 1.00000000 |

Site bytes per layer (MiB), 2-bit / int8: qkv 2.36 / 9.39, o 1.57 / 6.26, gate|up 8.49 / 33.80, down 4.23 / 16.88. FFN share of int8 bytes 0.764.

## Results: gap and recovery (per-round range); seed 1 / seed 2

| B | all2/all8 gap | all2/mixed speedup | gap fraction recovered |
|---|---|---|---|
| 1 | 1.360 (1.33–1.45) / 1.374 (1.29–1.41) | 1.084 (1.03–1.14) / 1.087 (0.99–1.19) | 0.291 (0.11–0.40) / 0.295 (-0.04–0.58) |
| 4 | 1.549 (1.45–1.66) / 1.579 (1.42–1.68) | 1.038 (1.00–1.12) / 1.088 (1.02–1.18) | 0.102 (-0.00–0.34) / 0.221 (0.06–0.38) |

## Results: weights + B·KV per layer at ctx 2048 (MiB) and all-int8 / config ratios

| KV fmt | B | KV MiB/stream/layer | all2 | mixed | all8 | all8/mixed | all8/all2 | 30L+scratch all8/mixed | 30L+scratch all8/all2 |
|---|---|---|---|---|---|---|---|---|---|
| a8g64 | 1 | 2.656 | 19.31 | 31.02 | 68.99 | 2.224 | 3.574 | 2.128 | 3.284 |
| a8g64 | 4 | 2.656 | 27.27 | 38.99 | 76.96 | 1.974 | 2.822 | 1.910 | 2.668 |
| k8v4 | 1 | 2.109 | 18.76 | 30.48 | 68.45 | 2.246 | 3.649 | 2.146 | 3.344 |
| k8v4 | 4 | 2.109 | 25.09 | 36.81 | 74.77 | 2.032 | 2.981 | 1.961 | 2.801 |

## Pair files this wave

Quant pairs idle this wave. Quant Chief's 10:05 MT assignment (latest ledger, JuniorHome docs/ledger/2026-11-23_1005.md) was "no gap" for both pairs: roadmap/beta blobs unchanged, no eligible item, no quant-builder shas. No new Trit Pack or Handshake pair files appear under /workspace today, so there is nothing to link.

## Next question

Every config is slower per step at B 1 than at B 4 (~19 vs ~13 ms), so single-user decode in this stack is held back by the tile matmul itself, not just the weight format. At B 1, does replacing np.matmul(x(1,in), tile.T) with a contiguous row-major GEMV (np.dot(tile, x[0]) into the output slice) cut full-layer time by >= 1.3x for both all-2-bit and all-int8, and does the all-2-bit/int8 gap then widen or shrink from ~1.37x?

close: 816ed80c308b8784
