---
local: true
date: 2026-11-13
wave: "2026-11-13 10:55 (ran ~10:52–10:54 MT; Chief last_run 2026-11-13T17:10:00Z ~10:10 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, bitplane-activation, down-proj, tmac-g8, numpy, not-viable]
---

# A5 per-row bitplane activations keep a 1.66–1.83x speed win over `tmac_g8` on the full bitnetd GQA layer at B=1, but the down projection's silu·up input drops to cos 0.959 (layer delta cos 0.980, 10/32 tokens under 0.98), and routing down through `tmac_g8` leaves only ~1.44x (est.), so not viable

## Question

The prior note [[2026-11-13_0555_a4-a5-a6-bitplane-activation-per-row-vs-g128-scales-packed-ternary-gemv-b1-b4-vs-tmac-g8]] found A5 per-row (5 activation bitplanes, c=0.26) viable on a single 2560×2560 GEMV: cos 0.988 vs the weight-only W1.58 path and 1.61–1.81x over `tmac_g8` at B=1. Its next question: does the win survive the full bitnetd layer (q/k/v/o + gate/up/down, GQA, B=1), i.e. the down projection's 6912-wide silu·up input and the per-site activation re-quantisation? Quant Chief `last_tips.json` `last_run` 2026-11-13T17:10:00Z (~10:10 MT) wrote pair_A/pair_B **no gap** (74th idle wave), so the lane's own next question is used.

## Method

One decoder-layer step, B=1, numpy 2.5.4, single process, 8-core virtualised Xeon (load avg 0.60 → 1.99 across the runs). Shapes: q/o 2560×2560, k/v 640×2560 (5 KV heads × 128), gate/up 6912×2560, down 2560×6912. Weights N(0, 0.02), absmean-ternarised per matrix. Context length 1, so attention output is exactly v repeated over the 4 q heads per KV head (softmax over one key = 1); RMSNorm before attention and FFN, SiLU·up into down, both residuals. A5 path: one per-row absmean quant + 5-plane packing per input site (attn-in shared by q/k/v, o-in, ffn-in shared by gate/up, down-in), all **inside** the timed path; mask/sign packed weights, 2 popcounts per plane, output tile T=1024. c is calibrated per site on a 16-token draw (seed+1) over a 0.14–0.40 grid; the 05:55 fixed c=0.26 is reported too. Accuracy on a separate 32-token t(4)·0.05 draw (seed+2) vs the same layer with weight-only f32-activation × trit products (what `tmac_g8` computes; its own max rel err 6.0e-07). Baselines run the identical layer: `tmac_g8`, `int8_scratch` (reused scratch, int8→f32 then GEMV), `f32_resident`. 9 timed repeats × 5 layer steps after 3 warmups; three full runs; plus a per-projection split for q, k, gate, down.

Command (offline, no network, nothing bound):

```
cd /workspace/bench1055_1113 && for i in 1 2 3; do python3 bench.py results_run$i.json | tee run$i.txt; done
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Full bitnetd decoder-layer step at B=1 (GQA: hidden 2560, 20 q heads x 128, 5 KV heads -> k/v 640,
FFN 6912): A5 per-row bitplane activations x packed mask/sign ternary weights (activation quant +
plane packing inside the timed path, one quant shared by q/k/v and by gate/up) vs weight-only tmac_g8
(f32 activations, same 2-bit packed bytes), int8_scratch and f32_resident. Context length 1, so
attention output is exactly v repeated over the 4 q heads per KV head (softmax over one key = 1).
Accuracy = cos of every projection and of the final layer output vs the weight-only W1.58 path, on a
32-token eval draw, with c calibrated per input site on a separate 16-token draw.
numpy-only, offline, single process. Usage: python3 bench.py results_runN.json"""
import json, statistics, sys, time
import numpy as np
assert hasattr(np, "bitwise_count")
D, F, HD, QH, KVH = 2560, 6912, 128, 20, 5
KV = KVH * HD
SHAPES = {"q": (D, D), "k": (KV, D), "v": (KV, D), "o": (D, D), "gate": (F, D), "up": (F, D), "down": (D, F)}
K = 5; L = (1 << K) - 1; T = 1024
CGRID = [round(0.14 + 0.02 * i, 2) for i in range(14)]
REPEATS, WARM, PASSES = 9, 3, 5
SEED = 202611131055
u64 = lambda p: np.ascontiguousarray(p).view(np.uint64)
def absmean_trits(w):
    g = float(np.mean(np.abs(w))) + 1e-7
    return np.clip(np.rint(w / g), -1, 1).astype(np.int8), g
def absmax_int8(w):
    m = float(np.max(np.abs(w))) / 127.0
    return np.clip(np.rint(w / m), -127, 127).astype(np.int8), m
def rms(x): return x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-6).astype(np.float32)
def silu(x): return x / (1 + np.exp(-x))
def quant5(x, c):
    h = (c * np.mean(np.abs(x), axis=1, keepdims=True) + 1e-7).astype(np.float32)
    q = np.clip(np.rint((x / h + L) / 2), 0, L).astype(np.uint8)
    return [u64(np.packbits((q >> j) & 1, axis=-1)) for j in range(K)], h
def a5gemv(planes, h, P):
    wm, ws, sumw, gw = P["wm"], P["ws"], P["sumw"], P["gw"]
    out = wm.shape[0]; B = h.shape[0]
    acc = np.zeros((B, out), np.int32)
    for o0 in range(0, out, T):
        wmt = wm[None, o0:o0+T]; wst = ws[None, o0:o0+T]
        for j, p in enumerate(planes):
            a = p[:, None, :] & wmt
            s = np.bitwise_count(a).sum(-1, dtype=np.int32)
            a &= wst
            s -= 2 * np.bitwise_count(a).sum(-1, dtype=np.int32)
            acc[:, o0:o0+T] += s << (j + 1)
    acc -= L * sumw[None, :]
    return acc.astype(np.float32) * (np.float32(gw) * h)
LUT = np.array([[((c >> (2 * j)) & 3) for j in range(4)] for c in range(256)], np.int64)
LUTF_T = np.where(LUT == 2, -1, LUT).astype(np.float32).T
def pack2(t):
    c = np.where(t < 0, 2, t).astype(np.uint8).reshape(t.shape[0], -1, 4)
    p = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return np.ascontiguousarray(p.T)
G = 8
OFFS = (np.arange(G, dtype=np.uint16) * 256)[:, None]
SCR_T = {o: (np.empty((G, o), np.uint16), np.empty((G, o), np.float32), np.empty(o, np.float32)) for o in (D, KV, F)}
def tmac_g8(P, x1):
    PT = P["PT"]; q, out = PT.shape
    idx, tf, ta = SCR_T[out]
    Tf = (x1.reshape(q, 4) @ LUTF_T).ravel()
    y = np.zeros(out, np.float32)
    for c in range(0, q, G):
        np.add(PT[c:c+G], OFFS, out=idx, dtype=np.uint16)
        np.take(Tf[c*256:(c+G)*256], idx, out=tf)
        np.add.reduce(tf, axis=0, out=ta); y += ta
    y *= np.float32(P["gw"])
    return y[None, :]
SCR = np.empty(F * D, np.float32)
def i8scr(P, x):
    o, i = P["q8"].shape; s = SCR[:o * i].reshape(o, i)
    np.copyto(s, P["q8"], casting='unsafe'); y = x @ s.T; y *= np.float32(P["m8"]); return y
def f32res(P, x): return x @ P["Wd"].T
def layer(x, mm, qmode=None, cs=None):
    """mm(name, xin, shared_quant) -> y. qmode: A5 path gets pre-quantised planes per site."""
    h1 = rms(x)
    if qmode:
        pl, hh = quant5(h1, cs["attn"]); q, k, v = (a5gemv(pl, hh, PW[n]) for n in ("q", "k", "v"))
    else:
        q, k, v = (mm(PW[n], h1) for n in ("q", "k", "v"))
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)   # ctx=1 -> softmax=1
    if qmode:
        pl, hh = quant5(att, cs["o"]); o = a5gemv(pl, hh, PW["o"])
    else:
        o = mm(PW["o"], att)
    x2 = x + o; h2 = rms(x2)
    if qmode:
        pl, hh = quant5(h2, cs["ffn"]); g, u = a5gemv(pl, hh, PW["gate"]), a5gemv(pl, hh, PW["up"])
    else:
        g, u = mm(PW["gate"], h2), mm(PW["up"], h2)
    z = (silu(g) * u).astype(np.float32)
    if qmode:
        pl, hh = quant5(z, cs["down"]); dn = a5gemv(pl, hh, PW["down"])
    else:
        dn = mm(PW["down"], z)
    return x2 + dn, dict(q=q, k=k, v=v, o=o, gate=g, up=u, down=dn), dict(attn=h1, o=att, ffn=h2, down=z)
def timed(fn):
    for _ in range(WARM): fn()
    s = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(PASSES): fn()
        s.append((time.perf_counter() - t0) / PASSES * 1e3)
    return {"median_ms": statistics.median(s), "min_ms": min(s), "max_ms": max(s)}
def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
def rowcos(a, b): return [cos(a[i], b[i]) for i in range(a.shape[0])]
PW = {}
def build():
    rng = np.random.default_rng(SEED)
    by = {"packed_ms": 0, "sumw": 0, "tmac_PT": 0, "int8": 0, "f32": 0}
    for n, (o, i) in SHAPES.items():
        W = (rng.standard_normal((o, i)) * 0.02).astype(np.float32)
        tw, gw = absmean_trits(W); q8, m8 = absmax_int8(W)
        P = dict(gw=gw, wm=u64(np.packbits(tw != 0, axis=-1)), ws=u64(np.packbits(tw < 0, axis=-1)),
                 sumw=tw.astype(np.int32).sum(1), PT=pack2(tw), q8=q8, m8=m8, Wd=q8.astype(np.float32) * np.float32(m8),
                 Wt=tw.astype(np.float32) * np.float32(gw))
        PW[n] = P
        by["packed_ms"] += P["wm"].nbytes + P["ws"].nbytes; by["sumw"] += P["sumw"].nbytes
        by["tmac_PT"] += P["PT"].nbytes; by["int8"] += q8.nbytes; by["f32"] += P["Wd"].nbytes
    return rng, by
wo_mm = lambda P, x: x @ P["Wt"].T   # weight-only W1.58 reference (what tmac_g8 computes)
def main():
    rng, by = build()
    tok = lambda r, n: (r.standard_t(4, (n, D)) * 0.05).astype(np.float32)
    # calibrate c per input site on 16 tokens (SEED+1): maximise pooled cos of that site's outputs
    xc = tok(np.random.default_rng(SEED + 1), 16)
    _, ycw, sites = layer(xc, wo_mm)
    site_proj = {"attn": ("q", "k", "v"), "o": ("o",), "ffn": ("gate", "up"), "down": ("down",)}
    cs, calib = {}, {}
    for s, projs in site_proj.items():
        best = None; calib[s] = {}
        for c in CGRID:
            pl, hh = quant5(sites[s], c)
            cc = statistics.mean(cos(a5gemv(pl, hh, PW[p]), ycw[p]) for p in projs)
            calib[s][str(c)] = cc
            if best is None or cc > best[1]: best = (c, cc)
        cs[s] = best[0]
    print("c per site", cs, flush=True)
    # eval on 32 tokens (SEED+2): weight-only reference vs A5 path, full layer composition
    xe = tok(np.random.default_rng(SEED + 2), 32)
    yw, pw, _ = layer(xe, wo_mm)
    ya, pa, _ = layer(xe, None, qmode=True, cs=cs)
    # also the A5 path with the 05:55 fixed c=0.26 everywhere
    yf, pf, _ = layer(xe, None, qmode=True, cs={s: 0.26 for s in site_proj})
    ev = {"c": cs, "calib": calib, "proj_cos": {p: cos(pa[p], pw[p]) for p in SHAPES},
          "proj_cos_c026": {p: cos(pf[p], pw[p]) for p in SHAPES}}
    rc = rowcos(ya - xe, yw - xe)   # layer residual delta (what the layer adds)
    ev["delta_cos_pooled"] = cos(ya - xe, yw - xe); ev["delta_cos_row_median"] = statistics.median(rc)
    ev["delta_cos_row_min"] = min(rc); ev["delta_rows_below_098"] = sum(r < 0.98 for r in rc)
    ev["out_cos_pooled"] = cos(ya, yw); ev["delta_cos_pooled_c026"] = cos(yf - xe, yw - xe)
    ev["delta_relmae"] = float(np.mean(np.abs((ya - xe) - (yw - xe))) / np.mean(np.abs(yw - xe)))
    yt, _, _ = layer(xe[:1], tmac_g8); ev["tmac_relerr_vs_wo"] = float(np.max(np.abs(yt - yw[:1])) / np.max(np.abs(yw[:1])))
    print(json.dumps({k: v for k, v in ev.items() if k != "calib"}), flush=True)
    # B=1 timing, full layer step
    x1 = tok(rng, 1)
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "eval32": ev, "time": {}}
    fns = {"a5_packed": lambda: layer(x1, None, qmode=True, cs=cs), "tmac_g8": lambda: layer(x1, tmac_g8),
           "int8_scratch": lambda: layer(x1, i8scr), "f32_resident": lambda: layer(x1, f32res)}
    for n, fn in fns.items():
        res["time"][n] = timed(fn); print(n, res["time"][n], flush=True)
    # per-projection split for A5 vs tmac (single GEMV incl. its quant for A5)
    split = {}
    h1 = rms(x1); z = (silu(f32res(PW["gate"], h1)) * f32res(PW["up"], h1)).astype(np.float32)
    for p, xin, s in (("q", h1, "attn"), ("k", h1, "attn"), ("gate", h1, "ffn"), ("down", z, "down")):
        ta = timed(lambda: a5gemv(*quant5(xin, cs[s]), PW[p])); tt = timed(lambda: tmac_g8(PW[p], xin[0]))
        split[p] = {"a5_ms": ta["median_ms"], "tmac_ms": tt["median_ms"], "speedup": tt["median_ms"] / ta["median_ms"]}
        tq = timed(lambda: quant5(xin, cs[s])); split[p]["quant_ms"] = tq["median_ms"]
    res["split"] = split; print(json.dumps(split), flush=True)
    t = res["time"]; a = t["a5_packed"]
    res["ratios"] = {b: t[b]["median_ms"] / a["median_ms"] for b in ("tmac_g8", "int8_scratch", "f32_resident")}
    res["ratios"]["tmac_worst"] = t["tmac_g8"]["min_ms"] / a["max_ms"]
    print(json.dumps(res["ratios"]), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
if __name__ == "__main__": main()
```
</details>

## Results

Resident weight bytes per layer: packed mask/sign 17,367,040 B (+ Σw 182,272 B), `tmac_g8` PT 17,367,040 B, int8 69,468,160 B (4.00x), f32 277,872,640 B (16.00x).

Calibrated c per site: attn 0.26, o 0.14, ffn 0.14, down 0.36 (identical every run).

Accuracy on 32 eval tokens (cos vs weight-only path, pooled):

| projection | A5 calibrated c | A5 fixed c=0.26 |
|---|---|---|
| q | 0.9905 | 0.9905 |
| k | 0.9905 | 0.9905 |
| v | 0.9906 | 0.9906 |
| o | 0.9884 | 0.9834 |
| gate | 0.9865 | 0.9768 |
| up | 0.9865 | 0.9767 |
| down | 0.9589 | 0.9343 |

Layer residual delta (what the layer adds to x): pooled cos 0.9803, token median 0.9833, min 0.9493, rel mae 0.195, 10/32 tokens under 0.98; fixed c=0.26 gives 0.9700. Layer output (x + delta) pooled cos 0.9806.

Full layer step, ms, median (min–max) over 9 repeats. Ratios are baseline / A5; "worst" is `tmac_g8` min / A5 max.

| run | A5 packed | tmac_g8 | int8_scratch | f32_resident | vs tmac_g8 (worst) | vs int8_scratch | vs f32_resident |
|---|---|---|---|---|---|---|---|
| 1 | 13.06 (13.00–13.41) | 22.40 (22.11–23.08) | 39.58 (37.83–47.28) | 1.69 (1.37–2.97) | **1.72x** (1.65x) | 3.03x | 0.13x |
| 2 | 13.27 (13.03–14.99) | 22.06 (21.74–23.23) | 29.13 (27.76–31.57) | 1.43 (1.34–2.68) | **1.66x** (1.45x) | 2.19x | 0.11x |
| 3 | 14.71 (14.35–15.27) | 26.88 (24.91–27.88) | 31.81 (29.72–39.14) | 1.35 (1.32–2.25) | **1.83x** (1.63x) | 2.16x | 0.09x |

Per-projection split, ms median (runs 1 / 2 / 3); A5 includes its own quant+pack:

| proj | A5 | tmac_g8 | speedup | quant+pack only |
|---|---|---|---|---|
| q | 1.33 / 1.42 / 1.40 | 2.14 / 2.13 / 2.26 | 1.61x / 1.51x / 1.61x | 0.026 / 0.027 / 0.027 |
| k | 0.34 / 0.34 / 0.38 | 0.87 / 0.78 / 0.89 | 2.58x / 2.31x / 2.34x | 0.024 / 0.025 / 0.028 |
| gate | 3.38 / 3.35 / 3.62 | 5.62 / 4.95 / 5.36 | 1.66x / 1.48x / 1.48x | 0.027 / 0.027 / 0.024 |
| down | 3.24 / 3.01 / 3.19 | 5.78 / 5.79 / 6.08 | 1.78x / 1.93x / 1.91x | 0.035 / 0.031 / 0.032 |

Hybrid estimate (A5 everywhere, down via `tmac_g8`), from the split above, not separately timed: 1.44x / 1.37x / 1.53x vs all-`tmac_g8`.

## Verdict

**not-viable.**

1. Speed survives: the full layer is 1.66–1.83x faster than `tmac_g8` at B=1, at the same 2-bit packed bytes. The per-site quant+pack is ~0.03 ms (<1% of the step), and down (6912 wide) speeds up as well as gate (~1.7–1.8x).
2. Accuracy does not: q/k/v/o/gate/up all hold cos ≥ 0.986, but down's SiLU·up input is much heavier-tailed and A5 gives only cos 0.959 even with its own c=0.36 (0.934 at the 05:55 c=0.26). The layer delta sits right on the bar (pooled 0.980) with 10/32 tokens under 0.98 and a worst token at 0.949, compounding over 30 layers.
3. The obvious fix, sending down through `tmac_g8`, drops the estimated layer speedup to 1.37–1.53x (median 1.44x), at or under the 1.5x bar and not separately timed.
4. Same caveat as before: `f32_resident` is ~9x faster than A5 at 16x the memory; the win is only versus the packed fallback.

The 05:55 single-GEMV result does not carry over to the full layer. Keep `tmac_g8` as the packed fallback; A5 per-row is acceptable only for the attention and gate/up sites.

## Pair files this wave

Quant Chief 10:10 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) both **no gap**, so no pair files were produced this wave. Roadmap blobs unchanged (JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923). JuniorLLM head at clone was bc40188 (the previous findings note); this note touches only `junior_osai/vault/findings/`.

## Next question

Does A6 or A8 on the down-projection input only (A5 elsewhere), or a per-row Winsor clip of the SiLU·up activations before A5, bring down to cos ≥ 0.98 and the layer delta to ≥ 0.98 for every token while keeping the full layer ≥ 1.5x over `tmac_g8` at B=1?

close: 869a40edf54ce911 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-13_0555)
