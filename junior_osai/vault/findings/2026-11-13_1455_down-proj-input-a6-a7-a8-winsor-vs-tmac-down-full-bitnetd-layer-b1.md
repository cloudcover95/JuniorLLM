---
local: true
date: 2026-11-13
wave: "2026-11-13 14:55 (ran ~14:53–14:55 MT; Chief last_run 2026-11-13T21:14:00Z ~14:14 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, a6, a8, winsor, bitplane-activation, down-proj, tmac-g8, numpy, not-viable]
---

# Raising the down-projection input to A6/A7/A8 or Winsor-clipping it does not fix the A5 full-layer accuracy gap: even an exact f32-input `tmac_g8` down only reaches down cos 0.961 and 9/32 tokens under 0.98, so the error comes from the gate/up (ffn-in) A5 site, not the down site; measured hybrid is 1.32–1.40x, so not viable

## Question

The prior note [[2026-11-13_1055_a5-bitplane-activation-full-bitnetd-gqa-layer-b1-down-proj-vs-tmac-g8]] found the A5 per-row bitplane path 1.66–1.83x faster than `tmac_g8` on the full bitnetd GQA layer at B=1, but the down projection reached only cos 0.959 (layer delta 0.980, 10/32 tokens under 0.98). Its next question: does A6 or A8 on the down input only (A5 elsewhere), or a per-row Winsor clip of the SiLU·up activations before A5, bring down to cos ≥ 0.98 and every token's layer delta to ≥ 0.98 while keeping the layer ≥ 1.5x over `tmac_g8`? Quant Chief `last_tips.json` `last_run` 2026-11-13T21:14:00Z (~14:14 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same layer as the prior note (hidden 2560, 20 q heads × 128, 5 KV heads, FFN 6912, ctx 1, RMSNorm, SiLU·up, both residuals; absmean-ternarised N(0, 0.02) weights; mask/sign packed GEMV, 2 popcounts per plane, T=1024), new seed 202611131455. Attn/o/ffn sites stay A5 with their per-site calibrated c; only the **down** input changes. Variants: A5/A6/A7/A8 per-row absmean (c calibrated per bit width on a 16-token draw), A5/A6 Winsor (step = per-row |x| quantile / L, quantile calibrated over 0.98–1.0, `np.quantile` inside the timed path), and down through `tmac_g8` on the f32 SiLU·up input (the exact weight-only down, measured, not estimated). Accuracy on a separate 32-token t(4)·0.05 draw vs the full weight-only W1.58 layer. Timing: full layer step at B=1, 9 repeats × 5 steps after 3 warmups, three full runs; numpy 2.5.4, single process, 8-core virtualised Xeon (load avg ~0.3 at start).

Command (offline, no network, nothing bound):

```
cd /workspace/bench1455_1113 && for i in 1 2 3; do python3 bench.py results_run$i.json > run$i.txt 2>&1; done
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Down-site fix for the A5 full-layer path (14:55 11/13). Full bitnetd decoder-layer step at B=1 (GQA: hidden 2560, 20 q heads x 128, 5 KV heads -> k/v 640,
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
SEED = 202611131455
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
def layer(x, mm, qmode=None, cs=None, dcfg=None):
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
        if dcfg is None: pl, hh = quant5(z, cs["down"]); dn = a5gemv(pl, hh, PW["down"])
        elif dcfg[0] == "tmac": dn = np.concatenate([tmac_g8(PW["down"], z[i]) for i in range(z.shape[0])])
        else: pl, hh = quantk(z, dcfg[2], dcfg[0], dcfg[1]); dn = a5gemv(pl, hh, PW["down"])
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
DGRID = {5: CGRID, 6: [round(0.04 + 0.01 * i, 2) for i in range(15)], 7: [round(0.02 + 0.005 * i, 3) for i in range(15)],
         8: [round(0.008 + 0.002 * i, 3) for i in range(15)]}
WGRID = [0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0]
def main():
    rng, by = build()
    tok = lambda r, n: (r.standard_t(4, (n, D)) * 0.05).astype(np.float32)
    xc = tok(np.random.default_rng(SEED + 1), 16)
    _, ycw, sites = layer(xc, wo_mm)
    site_proj = {"attn": ("q", "k", "v"), "o": ("o",), "ffn": ("gate", "up"), "down": ("down",)}
    cs = {}
    for s, projs in site_proj.items():
        cs[s] = max(CGRID, key=lambda c: statistics.mean(cos(a5gemv(*quant5(sites[s], c), PW[p]), ycw[p]) for p in projs))
    # down-site variants, each calibrated on the 16-token draw
    variants = {}
    z = sites["down"]
    for k in (5, 6, 7, 8):
        variants[f"A{k}"] = (k, "absmean", max(DGRID[k], key=lambda c: cos(a5gemv(*quantk(z, c, k), PW["down"]), ycw["down"])))
    for k in (5, 6):
        variants[f"A{k}w"] = (k, "winsor", max(WGRID, key=lambda c: cos(a5gemv(*quantk(z, c, k, "winsor"), PW["down"]), ycw["down"])))
    variants["tmac"] = ("tmac",)
    print("c", cs, "variants", variants, flush=True)
    xe = tok(np.random.default_rng(SEED + 2), 32)
    yw, pw, _ = layer(xe, wo_mm)
    ev = {"c": cs, "variants": {k: list(v) for k, v in variants.items()}, "acc": {}}
    for n, dc in variants.items():
        ya, pa, _ = layer(xe, None, qmode=True, cs=cs, dcfg=dc)
        rc = rowcos(ya - xe, yw - xe)
        ev["acc"][n] = {"down_cos": cos(pa["down"], pw["down"]), "delta_pooled": cos(ya - xe, yw - xe),
                        "delta_min": min(rc), "rows_below_098": sum(r < 0.98 for r in rc),
                        "delta_relmae": float(np.mean(np.abs(ya - yw)) / np.mean(np.abs(yw - xe)))}
        print(n, ev["acc"][n], flush=True)
    x1 = tok(rng, 1)
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "eval32": ev, "time": {}}
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, dc in variants.items():
        fns["a5_down_" + n] = (lambda dc=dc: layer(x1, None, qmode=True, cs=cs, dcfg=dc))
    for n, fn in fns.items():
        res["time"][n] = timed(fn); print(n, res["time"][n], flush=True)
    t = res["time"]; tm = t["tmac_g8"]
    res["ratios"] = {n: {"vs_tmac": tm["median_ms"] / t[n]["median_ms"], "worst": tm["min_ms"] / t[n]["max_ms"]}
                     for n in t if n.startswith("a5_down_")}
    print(json.dumps(res["ratios"]), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
if __name__ == "__main__": main()
```
</details>

## Results

Resident weight bytes per layer unchanged for every variant: packed mask/sign 17,367,040 B (+ Σw 182,272 B), `tmac_g8` PT 17,367,040 B, int8 69,468,160 B (4.00x), f32 277,872,640 B (16.00x). Extra bitplanes cost only activation scratch (≤ 8 × 864 B per token).

Calibrated c: attn 0.26, o 0.14, ffn 0.14, down (A5) 0.38; identical in all three runs, so accuracy is identical across runs.

Accuracy on 32 eval tokens (cos vs weight-only layer):

| down input | c / quantile | down cos | layer delta pooled | worst token | tokens < 0.98 |
|---|---|---|---|---|---|
| A5 absmean | 0.38 | 0.9491 | 0.9737 | 0.8807 | 14/32 |
| A6 absmean | 0.18 | 0.9553 | 0.9755 | 0.8822 | 12/32 |
| A7 absmean | 0.09 | 0.9569 | 0.9760 | 0.8826 | 11/32 |
| A8 absmean | 0.036 | 0.9525 | 0.9748 | 0.8818 | 13/32 |
| A5 Winsor (|x| quantile) | 0.998 | 0.9491 | 0.9738 | 0.8807 | 14/32 |
| A6 Winsor (|x| quantile) | 0.9995 | 0.9569 | 0.9758 | 0.8829 | 12/32 |
| down via tmac_g8 (f32 input) | — | 0.9611 | 0.9770 | 0.8835 | 9/32 |

Full layer step at B=1, ms median (min–max), runs 1 / 2 / 3; ratio = all-`tmac_g8` median / variant median ("worst" = `tmac_g8` min / variant max):

| down input | ms | vs all-`tmac_g8` |
|---|---|---|
| A5 absmean | 13.26 (13.10–14.11) / 13.95 (13.18–14.86) / 13.05 (12.96–13.86) | 1.61x (1.48x) / 1.56x (1.43x) / 1.69x (1.58x) |
| A6 absmean | 13.63 (13.56–14.26) / 13.88 (13.61–14.26) / 13.73 (13.40–14.31) | 1.57x (1.47x) / 1.57x (1.49x) / 1.61x (1.53x) |
| A7 absmean | 16.49 (15.18–17.66) / 14.36 (14.26–14.83) / 15.90 (14.40–20.74) | 1.30x (1.18x) / 1.52x (1.43x) / 1.39x (1.06x) |
| A8 absmean | 15.74 (14.89–18.33) / 15.32 (15.00–15.55) / 14.52 (14.45–17.90) | 1.36x (1.14x) / 1.42x (1.37x) / 1.52x (1.23x) |
| A5 Winsor (|x| quantile) | 13.34 (13.27–15.57) / 13.42 (13.35–13.74) / 13.00 (12.99–13.30) | 1.60x (1.34x) / 1.62x (1.55x) / 1.70x (1.65x) |
| A6 Winsor (|x| quantile) | 13.90 (13.81–14.00) / 14.10 (14.00–14.46) / 13.82 (13.77–14.17) | 1.54x (1.49x) / 1.55x (1.47x) / 1.60x (1.55x) |
| down via tmac_g8 (f32 input) | 15.97 (15.85–16.15) / 16.57 (16.17–17.24) / 15.77 (15.70–16.26) | 1.34x (1.30x) / 1.32x (1.23x) / 1.40x (1.35x) |
| all-`tmac_g8` baseline | 21.36 (20.92–21.91) / 21.80 (21.25–23.33) / 22.10 (21.96–22.33) | 1.00x |
| `f32_resident` | 1.50 (1.42–2.49) / 1.63 (1.38–3.18) / 1.71 (1.67–1.92) | 14.2x / 13.4x / 12.9x |

## Verdict

**not-viable.**

1. The down site is not where the error lives. With the down GEMV made exact on the f32 SiLU·up input (`tmac_g8`), down cos is still only 0.961 and 9/32 tokens stay under 0.98 (worst 0.883). The remaining loss comes from the A5 gate/up outputs being multiplied together in SiLU·up, which amplifies their relative error before down sees it.
2. More down bitplanes barely help: A6 0.955, A7 0.957, A8 0.952 vs A5 0.949 (A8 is not better than A7, because its calibrated range clips harder). Winsor quantile scaling matches absmean (A5w 0.949, A6w 0.957). No variant reaches 0.98 on down or on every token.
3. Speed: A5/A6 on down keep 1.56–1.69x / 1.57–1.61x over `tmac_g8`; A7/A8 drop to ~1.3–1.5x (more planes, int64 accumulator at A8). The measured hybrid with `tmac_g8` down is 1.32–1.40x, lower than the prior note's 1.44x estimate and under the 1.5x bar.
4. As before, `f32_resident` is ~13–14x faster than `tmac_g8` at 16x the memory; any win is only versus the packed fallback.

Keep `tmac_g8` as the packed fallback. Spending bits on the down input is not worth it; if anything gets more bits, it should be the ffn-in (gate/up) site.

## Pair files this wave

Quant Chief 14:14 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) both **no gap**, so no pair files were produced this wave. Roadmap blobs unchanged (JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923). JuniorLLM head at clone was fae3bcf (the previous findings note); this note touches only `junior_osai/vault/findings/`.

## Next question

Does raising only the ffn-in (gate/up) site to A6 or A7, with attn/o/down at A5, bring the layer delta to ≥ 0.98 on every token, and does the full layer then stay ≥ 1.5x over `tmac_g8` at B=1 (gate+up are about half of the A5 step)?

close: be61e1123c623db7 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-13_1055)
