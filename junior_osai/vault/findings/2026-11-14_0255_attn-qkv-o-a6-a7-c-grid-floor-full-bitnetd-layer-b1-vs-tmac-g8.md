---
local: true
date: 2026-11-14
wave: "2026-11-14 02:55 (ran ~02:57–03:01 MT; Chief last_run 2026-11-14T09:15:00Z ~02:15 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: inconclusive
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, a6, a7, bitplane-activation, attn-qkv, o-proj, c-grid, tmac-g8, numpy, inconclusive]
---

# The A5 full-layer error comes from the q/k/v input site: q/k/v at A6 lifts the layer delta from 0.9790 to 0.9877 (17/32 → 1/32 tokens under 0.98) and stays 1.54–1.56x over `tmac_g8` at B=1, but the one config with every token ≥ 0.98 (q/k/v + down A6) is only 1.47–1.50x, so this is inconclusive

## Question

The prior note [[2026-11-13_1855_ffn-in-gate-up-a6-a7-a8-vs-tmac-full-bitnetd-layer-b1-error-upstream-attn-o]] ruled out the FFN sites: even exact `tmac_g8` on gate/up and down left 7/32 tokens under 0.98, so the error had to come from the attn/o A5 sites. It also flagged that the o and ffn calibrated c sat on the floor (0.14) of the c grid. Its next question: does extending the c grid down to 0.04, or raising only q/k/v and o to A6, bring every token to ≥ 0.98 while the full layer stays ≥ 1.5x over `tmac_g8` at B=1? Quant Chief `last_tips.json` `last_run` 2026-11-14T09:15:00Z (~02:15 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same bitnetd GQA layer as the last three notes (hidden 2560, 20 q heads × 128, 5 KV heads, FFN 6912, ctx 1, RMSNorm, SiLU·up, both residuals; absmean-ternarised N(0, 0.02) weights; mask/sign packed GEMV, 2 popcounts per plane, T=1024), new seed 202611140255. Each input site (q/k/v shared, o, gate/up shared, down) gets its own spec: Ak per-row absmean bitplanes with c calibrated per site and bit width on a 16-token draw (A5 on the old 0.14–0.40 grid and on an extended 0.04–0.40 grid; A6 on 0.02–0.20; A7 on 0.01–0.10), or exact f32-input `tmac_g8`. Accuracy on a separate 32-token t(4)·0.05 draw vs the full weight-only W1.58 layer. Timing: full layer step at B=1, 9 repeats × 5 steps after 3 warmups, three full runs; numpy 2.5.4, single process, 8-core virtualised Xeon (load avg 0.3–0.95). A first pass of three runs without the q/k/v + down A6 variant gave the same picture (q/k/v A6 1.57–1.61x, q/k/v + o A6 1.53–1.58x, q/k/v + o + down A6 1.47–1.52x); the tables below are the second pass.

Command (offline, no network, nothing bound):

```
cd /workspace/bench0255_1114 && for i in 1 2 3; do python3 bench.py results_run$i.json > run$i.txt 2>&1; done
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""attn (q/k/v) and o site bit width / c-grid floor for the A5 full-layer path (02:55 11/14). Full bitnetd decoder-layer step at B=1 (GQA: hidden 2560, 20 q heads x 128, 5 KV heads -> k/v 640,
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
SEED = 202611140255
u64 = lambda p: np.ascontiguousarray(p).view(np.uint64)
def absmean_trits(w):
    g = float(np.mean(np.abs(w))) + 1e-7
    return np.clip(np.rint(w / g), -1, 1).astype(np.int8), g
def absmax_int8(w):
    m = float(np.max(np.abs(w))) / 127.0
    return np.clip(np.rint(w / m), -127, 127).astype(np.int8), m
def rms(x): return x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-6).astype(np.float32)
def silu(x): return x / (1 + np.exp(-x))
def quantk(x, c, k=5, mode="absmean"):
    Lk = (1 << k) - 1
    if mode == "absmean":
        h = (c * np.mean(np.abs(x), axis=1, keepdims=True) + 1e-7).astype(np.float32)
    else:  # Winsor: range = per-row |x| quantile c, step h = range / Lk
        h = (np.quantile(np.abs(x), c, axis=1, keepdims=True) / Lk + 1e-7).astype(np.float32)
    q = np.clip(np.rint((x / h + Lk) / 2), 0, Lk).astype(np.uint8)
    return [u64(np.packbits((q >> j) & 1, axis=-1)) for j in range(k)], h
def quant5(x, c): return quantk(x, c, 5)
def a5gemv(planes, h, P):
    wm, ws, sumw, gw = P["wm"], P["ws"], P["sumw"], P["gw"]
    out = wm.shape[0]; B = h.shape[0]; Lk = (1 << len(planes)) - 1
    acc = np.zeros((B, out), np.int64 if len(planes) > 7 else np.int32)
    for o0 in range(0, out, T):
        wmt = wm[None, o0:o0+T]; wst = ws[None, o0:o0+T]
        for j, p in enumerate(planes):
            a = p[:, None, :] & wmt
            s = np.bitwise_count(a).sum(-1, dtype=np.int32)
            a &= wst
            s -= 2 * np.bitwise_count(a).sum(-1, dtype=np.int32)
            acc[:, o0:o0+T] += s.astype(acc.dtype) << (j + 1)
    acc -= Lk * sumw[None, :]
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
def site_mm(spec, x, names):
    """spec ("tmac",) -> exact f32-input tmac_g8; (k, c) -> Ak per-row absmean bitplanes."""
    if spec[0] == "tmac": return [np.concatenate([tmac_g8(PW[n], x[i]) for i in range(x.shape[0])]) for n in names]
    pl, hh = quantk(x, spec[1], spec[0]); return [a5gemv(pl, hh, PW[n]) for n in names]
def layer(x, mm, cfg=None):
    """mm(P, x) -> y for the reference/baseline paths; cfg = {site: spec} for the quantised path."""
    h1 = rms(x)
    if cfg: q, k, v = site_mm(cfg["attn"], h1, ("q", "k", "v"))
    else: q, k, v = (mm(PW[n], h1) for n in ("q", "k", "v"))
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)   # ctx=1 -> softmax=1
    o = site_mm(cfg["o"], att, ("o",))[0] if cfg else mm(PW["o"], att)
    x2 = x + o; h2 = rms(x2)
    if cfg: g, u = site_mm(cfg["ffn"], h2, ("gate", "up"))
    else: g, u = mm(PW["gate"], h2), mm(PW["up"], h2)
    z = (silu(g) * u).astype(np.float32)
    dn = site_mm(cfg["down"], z, ("down",))[0] if cfg else mm(PW["down"], z)
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
GRIDS = {"old5": (5, CGRID), "ext5": (5, [round(0.04 + 0.02 * i, 2) for i in range(19)]),
         6: (6, [round(0.02 + 0.01 * i, 2) for i in range(19)]), 7: (7, [round(0.01 + 0.005 * i, 3) for i in range(19)])}
def main():
    rng, by = build()
    tok = lambda r, n: (r.standard_t(4, (n, D)) * 0.05).astype(np.float32)
    xc = tok(np.random.default_rng(SEED + 1), 16)
    _, ycw, sites = layer(xc, wo_mm)
    site_proj = {"attn": ("q", "k", "v"), "o": ("o",), "ffn": ("gate", "up"), "down": ("down",)}
    cal = {}
    for g, (k, grid) in GRIDS.items():
        for s, projs in site_proj.items():
            if g in (6, 7) and s in ("ffn",): continue
            score = {c: statistics.mean(cos(a5gemv(*quantk(sites[s], c, k), PW[p]), ycw[p]) for p in projs) for c in grid}
            c = max(score, key=score.get)
            cal[f"{s}_{g}"] = (k, c, score[c])
    print("cal", cal, flush=True)
    sp = lambda s, g: (cal[f"{s}_{g}"][0], cal[f"{s}_{g}"][1])
    TM = ("tmac",)
    base = lambda g: {s: sp(s, g) for s in site_proj}
    variants = {"allA5_oldgrid": base("old5"), "allA5_extgrid": base("ext5"),
                "oA6": {**base("ext5"), "o": sp("o", 6)}, "attnA6": {**base("ext5"), "attn": sp("attn", 6)},
                "attnoA6": {**base("ext5"), "attn": sp("attn", 6), "o": sp("o", 6)},
                "attnoA7": {**base("ext5"), "attn": sp("attn", 7), "o": sp("o", 7)},
                "attnA6_downA6": {**base("ext5"), "attn": sp("attn", 6), "down": sp("down", 6)},
                "attnoA6_downA6": {**base("ext5"), "attn": sp("attn", 6), "o": sp("o", 6), "down": sp("down", 6)},
                "attnoTmac": {**base("ext5"), "attn": TM, "o": TM},
                "attnoTmac_downTmac": {**base("ext5"), "attn": TM, "o": TM, "down": TM}}
    xe = tok(np.random.default_rng(SEED + 2), 32)
    yw, pw, _ = layer(xe, wo_mm)
    ev = {"cal": {k: list(v) for k, v in cal.items()}, "variants": {k: {s: list(v) for s, v in c.items()} for k, c in variants.items()}, "acc": {}}
    for n, cfg in variants.items():
        ya, pa, _ = layer(xe, None, cfg)
        rc = rowcos(ya - xe, yw - xe)
        ev["acc"][n] = {**{p + "_cos": cos(pa[p], pw[p]) for p in ("q", "v", "o", "gate", "down")},
                        "delta_pooled": cos(ya - xe, yw - xe), "delta_min": min(rc), "rows_below_098": sum(r < 0.98 for r in rc),
                        "delta_relmae": float(np.mean(np.abs(ya - yw)) / np.mean(np.abs(yw - xe)))}
        print(n, ev["acc"][n], flush=True)
    x1 = tok(rng, 1)
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "eval32": ev, "time": {}}
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, cfg in variants.items():
        fns["q_" + n] = (lambda cfg=cfg: layer(x1, None, cfg))
    for n, fn in fns.items():
        res["time"][n] = timed(fn); print(n, res["time"][n], flush=True)
    t = res["time"]; tm = t["tmac_g8"]
    res["ratios"] = {n: {"vs_tmac": tm["median_ms"] / t[n]["median_ms"], "worst": tm["min_ms"] / t[n]["max_ms"]}
                     for n in t if n.startswith("q_")}
    print(json.dumps(res["ratios"]), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
if __name__ == "__main__": main()
```
</details>

## Results

Resident weight bytes per layer are the same for every variant: packed mask/sign 17,367,040 B (+ Σw 182,272 B), `tmac_g8` PT 17,367,040 B, int8 69,468,160 B (4.00x), f32 277,872,640 B (16.00x). Extra bitplanes cost only activation scratch.

Calibrated c (identical in all runs, so accuracy is identical across runs): A5 old grid q/k/v 0.3, o 0.14, gate/up 0.14, down 0.38; A5 extended grid q/k/v 0.3, o 0.12, gate/up 0.12, down 0.38; A6 q/k/v 0.2, o 0.07, down 0.2; A7 q/k/v 0.1, o 0.035. The extended grid moves o and gate/up only one step (0.14 → 0.12), so the grid floor was not the issue.

Accuracy on 32 eval tokens (cos vs weight-only layer):

| config (others A5, extended grid) | q cos | o cos | gate cos | down cos | layer delta pooled | worst token | tokens < 0.98 |
|---|---|---|---|---|---|---|---|
| all A5, old c grid 0.14–0.40 | 0.9885 | 0.9865 | 0.9848 | 0.9571 | 0.9785 | 0.9652 | 18/32 |
| all A5, c grid 0.04–0.40 | 0.9885 | 0.9868 | 0.9855 | 0.9584 | 0.9790 | 0.9660 | 17/32 |
| o A6 | 0.9885 | 0.9880 | 0.9866 | 0.9604 | 0.9804 | 0.9678 | 15/32 |
| q/k/v A6 | 0.9955 | 0.9937 | 0.9921 | 0.9719 | 0.9877 | 0.9788 | 1/32 |
| q/k/v + o A6 | 0.9955 | 0.9948 | 0.9933 | 0.9738 | 0.9891 | 0.9798 | 1/32 |
| q/k/v + o A7 | 0.9980 | 0.9979 | 0.9962 | 0.9803 | 0.9931 | 0.9845 | 0/32 |
| q/k/v A6 + down A6 | 0.9955 | 0.9937 | 0.9921 | 0.9794 | 0.9898 | 0.9815 | 0/32 |
| q/k/v + o + down A6 | 0.9955 | 0.9948 | 0.9933 | 0.9813 | 0.9912 | 0.9822 | 0/32 |
| q/k/v + o tmac_g8 (f32) | 1.0000 | 1.0000 | 0.9982 | 0.9841 | 0.9956 | 0.9936 | 0/32 |
| q/k/v + o + down tmac_g8 (only gate/up A5) | 1.0000 | 1.0000 | 0.9982 | 0.9963 | 0.9990 | 0.9983 | 0/32 |

Full layer step at B=1, ms median (min–max), runs 1 / 2 / 3; ratio = all-`tmac_g8` median / variant median ("worst" = `tmac_g8` min / variant max):

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| all A5, old c grid 0.14–0.40 | 13.32 (13.18–13.51) / 13.10 (13.06–13.13) / 13.01 (12.95–13.18) | 1.58x (1.55x) / 1.60x (1.60x) / 1.60x (1.57x) |
| all A5, c grid 0.04–0.40 | 13.36 (13.30–13.84) / 13.09 (13.06–13.38) / 12.96 (12.94–14.12) | 1.58x (1.51x) / 1.61x (1.57x) / 1.61x (1.47x) |
| o A6 | 13.53 (13.46–13.56) / 13.34 (13.30–13.45) / 13.22 (13.18–13.27) | 1.56x (1.54x) / 1.58x (1.56x) / 1.58x (1.56x) |
| q/k/v A6 | 13.69 (13.60–14.13) / 13.46 (13.43–13.51) / 13.35 (13.29–14.30) | 1.54x (1.48x) / 1.56x (1.55x) / 1.56x (1.45x) |
| q/k/v + o A6 | 13.91 (13.85–13.99) / 13.78 (13.65–13.89) / 13.68 (13.57–14.24) | 1.52x (1.49x) / 1.53x (1.51x) / 1.52x (1.45x) |
| q/k/v + o A7 | 14.51 (14.44–14.78) / 14.34 (14.27–14.53) / 14.25 (14.18–14.70) | 1.45x (1.41x) / 1.47x (1.44x) / 1.46x (1.41x) |
| q/k/v A6 + down A6 | 14.19 (14.18–14.75) / 14.05 (14.00–14.14) / 14.21 (14.07–14.81) | 1.49x (1.42x) / 1.50x (1.48x) / 1.47x (1.40x) |
| q/k/v + o + down A6 | 14.42 (14.38–14.60) / 14.29 (14.26–14.40) / 14.38 (14.30–14.42) | 1.46x (1.43x) / 1.47x (1.46x) / 1.45x (1.44x) |
| q/k/v + o tmac_g8 (f32) | 15.72 (15.64–16.30) / 15.67 (15.56–16.00) / 15.80 (15.62–16.09) | 1.34x (1.28x) / 1.34x (1.31x) / 1.32x (1.29x) |
| q/k/v + o + down tmac_g8 (only gate/up A5) | 18.12 (18.05–18.89) / 18.16 (18.03–18.36) / 18.23 (18.15–18.63) | 1.16x (1.11x) / 1.16x (1.14x) / 1.14x (1.11x) |
| all-`tmac_g8` baseline | 21.08 (20.89–21.54) / 21.02 (20.95–21.10) / 20.85 (20.70–21.91) | 1.00x |
| `f32_resident` | 2.87 (2.10–4.35) / 1.44 (1.37–1.56) / 1.47 (1.44–1.81) | 7.3x / 14.6x / 14.2x |

## Verdict

**inconclusive** (closes most of the accuracy gap, but does not clear both bars at once).

1. The source is q/k/v, not o and not the c grid. The extended grid adds only +0.0006 pooled; o A6 alone adds +0.0014; q/k/v A6 alone adds +0.0087 (q cos 0.9885 → 0.9955) and pulls gate cos up to 0.9921, which confirms the prior note: the FFN inputs were off because the residual carried the q/k/v→v→o error.
2. q/k/v A6 is cheap (q/k/v is 2560 × 3840 of the layer's weights): 1.54–1.56x vs `tmac_g8`, pooled 0.9877, but 1/32 token still sits at 0.9788. q/k/v + o A6 is 1.52–1.53x with worst 0.9798 (1/32 under).
3. Every-token ≥ 0.98 needs one more fix: q/k/v + down A6 (worst 0.9815, 0/32 under) is 1.47–1.50x; q/k/v + o A7 (worst 0.9845) is 1.45–1.47x. Both are just under the 1.5x bar.
4. As before, `f32_resident` is ~13–14x faster than `tmac_g8` at 16x the memory; any win is only versus the packed fallback.

Keep `tmac_g8` as the packed fallback and keep A5 off the default `juniorctl ask` path for now. Of all the mixes tried since 11-13 10:55, q/k/v A6 with the rest at A5 is the best candidate: about 1.55x at pooled 0.988.

## Pair files this wave

Quant Chief 02:15 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) both **no gap**, so this wave produced no pair files. Roadmap blobs are unchanged (JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923). JuniorLLM head at clone was 3339153 (cloudcover95, rails/scratchfirst, 02:48 MT; not a findings path); this note touches only `junior_osai/vault/findings/`.

## Next question

Is the last token a draw artefact or a real tail? With q/k/v A6 and everything else A5, does a larger eval (128+ tokens, two seeds) keep ≥ 97% of tokens ≥ 0.98? And can the down site be fixed for less than one full extra plane (e.g. A6 only on the top-|z| 25% of down input channels, or a per-row Winsor clip at A5) so that q/k/v A6 + cheap down clears every token at ≥ 1.5x?

close: 5d77a3733766edf6 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-13_1855)
