---
local: true
date: 2026-11-14
wave: "2026-11-14 05:55 (ran ~05:57–06:05 MT; Chief last_run 2026-11-14T12:08:00Z ~05:08 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: inconclusive
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, a6, bitplane-activation, outlier-split, winsor, attn-qkv, down-proj, tmac-g8, numpy, inconclusive]
---

# A 32-channel outlier split at q/k/v (A6) and down (A5) fixes the accuracy tail: 0/256 tokens under 0.98 (worst 0.9927) on a 2 × 128-token eval, with no extra resident bytes, but the speed sits right on the bar: 1.52–1.58x vs `tmac_g8` in three runs and 1.49x in an interleaved check, so this is inconclusive

## Question

The prior note [[2026-11-14_0255_attn-qkv-o-a6-a7-c-grid-floor-full-bitnetd-layer-b1-vs-tmac-g8]] found that the A5 full-layer error comes from the q/k/v input site; q/k/v A6 reached pooled 0.988 at ~1.55x but left 1/32 tokens under 0.98. Its next question: is that last token a draw artefact or a real tail on a larger eval (128+ tokens, two seeds), and can the down site be fixed for less than one full extra plane (top-|z| channels, or a per-row Winsor clip at A5) so that q/k/v A6 + cheap down clears every token at ≥ 1.5x? Quant Chief `last_tips.json` `last_run` 2026-11-14T12:08:00Z (~05:08 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same bitnetd GQA layer as the last four notes (hidden 2560, 20 q heads × 128, 5 KV heads, FFN 6912, ctx 1, RMSNorm, SiLU·up, both residuals; absmean-ternarised N(0, 0.02) weights; mask/sign packed GEMV, 2 popcounts per plane, T=1024), new seed 202611140555. New down/q/k/v options:

- **Winsor A5**: per-row step = |z| quantile q / 31 (q calibrated over 0.90–1.0).
- **Outlier split k**: per token, the k largest-|x| input channels are zeroed before the Ak quant (so the scale fits the bulk), and added back exactly as Σ x_i · trit(W[:, i]) · γ. The trit columns are unpacked on the fly from the same mask/sign bytes, so resident bytes stay the same. argpartition, column unpack and the small f32 matvec are all inside the timed path.

c (or q) is calibrated per site and option on a 16-token draw. Accuracy uses **two 128-token t(4)·0.05 eval draws** (seeds +2, +3), with per-token cos of the layer delta vs the weight-only W1.58 layer. Timing: full layer step at B=1, 9 repeats × 5 steps after 3 warmups, three full runs, plus an interleaved check (7 rounds alternating order, same process, `tinter.py`). numpy 2.5.4, single process, 8-core virtualised Xeon (load avg 0.5–2.2). A first pass without the split-at-q/k/v and tmac diagnostic rows gave the same accuracy and q/k/v A6 + down split 32 at 1.55x.

Command (offline, no network, nothing bound):

```
cd /workspace/bench0555_1114 && for i in 1 2 3; do python3 bench.py results_run$i.json > run$i.txt 2>&1; done && python3 tinter.py tinter.json
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Cheap down-site fix + larger eval for the q/k/v-A6 full-layer path (05:55 11/14). Full bitnetd
decoder-layer step at B=1 (GQA: hidden 2560, 20 q heads x 128, 5 KV heads -> k/v 640, FFN 6912),
q/k/v input at A6 per-row bitplanes, o and gate/up at A5, and the down input (silu(g)*u) handled by:
A5, A6, A5 Winsor (per-row |z| quantile clip), or A5 + top-k outlier split (k largest |z| channels
per token zeroed before A5 quant and added back exactly through ternary weight columns unpacked
from the same mask/sign planes; no extra resident bytes). vs weight-only tmac_g8 (f32 activations,
same 2-bit packed bytes) and f32_resident. ctx 1, so attention output = v repeated per KV group.
Accuracy = per-token cos of the layer delta vs the weight-only W1.58 layer on 2 x 128 eval tokens
(two token seeds), c calibrated per site on a separate 16-token draw.
numpy-only, offline, single process. Usage: python3 bench.py results_runN.json [noeval]"""
import json, statistics, sys, time
import numpy as np
assert hasattr(np, "bitwise_count")
D, F, HD, QH, KVH = 2560, 6912, 128, 20, 5
KV = KVH * HD
SHAPES = {"q": (D, D), "k": (KV, D), "v": (KV, D), "o": (D, D), "gate": (F, D), "up": (F, D), "down": (D, F)}
K = 5; L = (1 << K) - 1; T = 1024
CGRID = [round(0.14 + 0.02 * i, 2) for i in range(14)]
REPEATS, WARM, PASSES = 9, 3, 5
SEED = 202611140555
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

def split_mm(z, c, kout, P, kb=5):
    """Ak (default A5) on z with the kout largest-|z| channels per row zeroed, plus their exact ternary contribution."""
    B, Fi = z.shape
    idx = np.argpartition(np.abs(z), Fi - kout, axis=1)[:, Fi - kout:]
    zr = z.copy(); np.put_along_axis(zr, idx, 0.0, axis=1)
    y = a5gemv(*quantk(zr, c, kb), P)
    wm8, ws8 = P["wm8"], P["ws8"]
    for b in range(B):
        ix = idx[b]; byt = ix >> 3; sh = (7 - (ix & 7)).astype(np.uint8)
        m = (wm8[:, byt] >> sh) & 1; s = (ws8[:, byt] >> sh) & 1
        t = m.astype(np.float32) * (1 - 2 * s.astype(np.float32))
        y[b] += (t @ z[b, ix]) * np.float32(P["gw"])
    return y
def site_mm(spec, x, names):
    """("tmac",) exact f32 tmac_g8; (k, c) Ak absmean; ("w", c) A5 Winsor quantile c; ("s", c, kout[, k]) Ak + outlier split."""
    if spec[0] == "tmac": return [np.concatenate([tmac_g8(PW[n], x[i]) for i in range(x.shape[0])]) for n in names]
    if spec[0] == "s": return [split_mm(x, spec[1], spec[2], PW[n], spec[3] if len(spec) > 3 else 5) for n in names]
    if spec[0] == "w": pl, hh = quantk(x, spec[1], 5, "winsor")
    else: pl, hh = quantk(x, spec[1], spec[0])
    return [a5gemv(pl, hh, PW[n]) for n in names]
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
        P["wm8"] = P["wm"].view(np.uint8); P["ws8"] = P["ws"].view(np.uint8)
        PW[n] = P
        by["packed_ms"] += P["wm"].nbytes + P["ws"].nbytes; by["sumw"] += P["sumw"].nbytes
        by["tmac_PT"] += P["PT"].nbytes; by["int8"] += q8.nbytes; by["f32"] += P["Wd"].nbytes
    return rng, by
wo_mm = lambda P, x: x @ P["Wt"].T   # weight-only W1.58 reference (what tmac_g8 computes)
GRID5 = [round(0.04 + 0.02 * i, 2) for i in range(19)]; GRID6 = [round(0.02 + 0.01 * i, 2) for i in range(19)]
WGRID = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0]
KOUT = (32, 64, 128)
def main():
    rng, by = build()
    tok = lambda r, n: (r.standard_t(4, (n, D)) * 0.05).astype(np.float32)
    xc = tok(np.random.default_rng(SEED + 1), 16)
    _, ycw, sites = layer(xc, wo_mm)
    site_proj = {"attn": ("q", "k", "v"), "o": ("o",), "ffn": ("gate", "up"), "down": ("down",)}
    def calib(s, mk, grid):
        sc = {c: statistics.mean(cos(site_mm(mk(c), sites[s], (p,))[0], ycw[p]) for p in site_proj[s]) for c in grid}
        c = max(sc, key=sc.get); return mk(c), sc[c]
    cal = {f"{s}_5": calib(s, lambda c: (5, c), GRID5) for s in site_proj}
    cal["attn_6"] = calib("attn", lambda c: (6, c), GRID6); cal["down_6"] = calib("down", lambda c: (6, c), GRID6)
    cal["down_w"] = calib("down", lambda c: ("w", c), WGRID)
    for k in KOUT: cal[f"down_s{k}"] = calib("down", lambda c, k=k: ("s", c, k), GRID5)
    cal["attn_s6"] = calib("attn", lambda c: ("s", c, 32, 6), GRID6)
    print("cal", cal, flush=True)
    sp = lambda key: cal[key][0]
    base = {s: sp(f"{s}_5") for s in site_proj}
    qa = {**base, "attn": sp("attn_6")}
    variants = {"allA5": base, "attnA6": qa, "attnA6_downA6": {**qa, "down": sp("down_6")},
                "attnA6_downW5": {**qa, "down": sp("down_w")}}
    for k in KOUT: variants[f"attnA6_downS{k}"] = {**qa, "down": sp(f"down_s{k}")}
    TM = ("tmac",); ds = sp("down_s32")
    variants["attnA6S32_downS32"] = {**qa, "attn": sp("attn_s6"), "down": ds}
    variants["attnTmac_downS32"] = {**base, "attn": TM, "down": ds}
    variants["attnoTmac_downS32"] = {**base, "attn": TM, "o": TM, "down": ds}
    variants["attnoTmac_downTmac"] = {**base, "attn": TM, "o": TM, "down": TM}
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "cal": {k: [list(v[0]), v[1]] for k, v in cal.items()},
           "variants": {k: {s: list(v) for s, v in c.items()} for k, c in variants.items()}, "acc": {}, "time": {}}
    if "noeval" not in sys.argv:
        for es in (2, 3):
            xe = tok(np.random.default_rng(SEED + es), 128)
            yw = np.concatenate([layer(xe[i:i+32], wo_mm)[0] for i in range(0, 128, 32)])
            for n, cfg in variants.items():
                ya = np.concatenate([layer(xe[i:i+32], None, cfg)[0] for i in range(0, 128, 32)])
                rc = rowcos(ya - xe, yw - xe)
                res["acc"].setdefault(n, {})[f"seed{es}"] = {"delta_pooled": cos(ya - xe, yw - xe), "delta_min": min(rc),
                    "p05": float(np.percentile(rc, 5)), "rows_below_098": int(sum(r < 0.98 for r in rc)), "n": 128}
                print(n, es, res["acc"][n][f"seed{es}"], flush=True)
    x1 = tok(rng, 1)
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, cfg in variants.items(): fns["q_" + n] = (lambda cfg=cfg: layer(x1, None, cfg))
    for n, fn in fns.items():
        res["time"][n] = timed(fn); print(n, res["time"][n], flush=True)
    t = res["time"]; tm = t["tmac_g8"]
    res["ratios"] = {n: {"vs_tmac": tm["median_ms"] / t[n]["median_ms"], "worst": tm["min_ms"] / t[n]["max_ms"]} for n in t if n.startswith("q_")}
    print(json.dumps(res["ratios"]), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
if __name__ == "__main__": main()
```
</details>

<details><summary>tinter.py</summary>

```python
#!/usr/bin/env python3
"""Interleaved timing check (same process, alternating order) for tmac_g8 vs the best variants. Usage: python3 tinter.py out.json"""
import json, sys, statistics, numpy as np
import bench as b
b.build(); cal = json.load(open("results_run1.json"))["variants"]
cfg = {n: {s: tuple(v) for s, v in cal[n].items()} for n in ("attnA6", "attnA6S32_downS32", "allA5")}
x1 = (np.random.default_rng(b.SEED + 9).standard_t(4, (1, b.D)) * 0.05).astype(np.float32)
fns = {"tmac_g8": lambda: b.layer(x1, b.tmac_g8), **{n: (lambda c=c: b.layer(x1, None, c)) for n, c in cfg.items()}}
med = {n: [] for n in fns}
for r in range(7):
    order = list(fns) if r % 2 == 0 else list(fns)[::-1]
    for n in order: med[n].append(b.timed(fns[n])["median_ms"])
out = {n: {"median_of_medians": statistics.median(v), "min": min(v), "max": max(v), "rounds": v} for n, v in med.items()}
for n in cfg: out[n]["vs_tmac"] = out["tmac_g8"]["median_of_medians"] / out[n]["median_of_medians"]
print(json.dumps({n: {k: v for k, v in d.items() if k != "rounds"} for n, d in out.items()}, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
```
</details>

## Results

Resident weight bytes per layer are the same for every variant: packed mask/sign 17,367,040 B (+ Σw 182,272 B), `tmac_g8` PT 17,367,040 B, int8 69,468,160 B (4.00x), f32 277,872,640 B (16.00x). The split adds only per-token scratch (k indices plus a 2560 × k or 640 × k f32 column block).

Calibrated (identical in all runs): A5 q/k/v c=0.28, o 0.12, gate/up 0.12, down 0.38; A6 q/k/v 0.2, down 0.2; down Winsor q=0.998; down A5 split 32/64/128 c=0.26/0.22/0.2; q/k/v A6 split 32 c=0.07.

Accuracy, layer delta vs weight-only layer, eval seed +2 / seed +3 (128 tokens each; o and gate/up A5 unless stated otherwise):

| config | pooled cos | p5 token | worst token | tokens < 0.98 |
|---|---|---|---|---|
| all A5 | 0.9797 / 0.9782 | 0.9629 / 0.9605 | 0.8828 / 0.8533 | 37/128 / 44/128 |
| q/k/v A6, down A5 | 0.9871 / 0.9863 | 0.9808 / 0.9791 | 0.9159 / 0.8825 | 5/128 / 10/128 |
| q/k/v A6, down A6 | 0.9892 / 0.9883 | 0.9827 / 0.9808 | 0.9178 / 0.8843 | 4/128 / 6/128 |
| q/k/v A6, down A5 Winsor | 0.9871 / 0.9864 | 0.9807 / 0.9791 | 0.9147 / 0.8827 | 6/128 / 10/128 |
| q/k/v A6, down A5 + split 32 | 0.9895 / 0.9886 | 0.9827 / 0.9808 | 0.9191 / 0.8848 | 4/128 / 6/128 |
| q/k/v A6, down A5 + split 64 | 0.9898 / 0.9889 | 0.9830 / 0.9811 | 0.9195 / 0.8857 | 4/128 / 6/128 |
| q/k/v A6, down A5 + split 128 | 0.9901 / 0.9892 | 0.9836 / 0.9813 | 0.9196 / 0.8854 | 4/128 / 6/128 |
| **q/k/v A6 + split 32, down A5 + split 32** | 0.9952 / 0.9952 | 0.9943 / 0.9944 | 0.9927 / 0.9929 | 0/128 / 0/128 |
| q/k/v tmac_g8, down A5 + split 32 | 0.9957 / 0.9957 | 0.9947 / 0.9950 | 0.9934 / 0.9934 | 0/128 / 0/128 |
| q/k/v + o tmac_g8, down A5 + split 32 | 0.9979 / 0.9979 | 0.9976 / 0.9976 | 0.9972 / 0.9973 | 0/128 / 0/128 |
| q/k/v + o + down tmac_g8 (gate/up A5) | 0.9990 / 0.9990 | 0.9988 / 0.9988 | 0.9985 / 0.9986 | 0/128 / 0/128 |

Full layer step at B=1, ms median (min–max), runs 1 / 2 / 3; ratio = all-`tmac_g8` median / variant median ("worst" = `tmac_g8` min / variant max):

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| all A5 | 15.08 (14.54–18.66) / 13.14 (13.05–13.22) / 13.22 (13.10–13.79) | 1.55x (1.20x) / 1.67x (1.61x) / 1.72x (1.61x) |
| q/k/v A6, down A5 | 13.61 (13.51–14.14) / 13.47 (13.42–13.67) / 13.64 (13.52–13.92) | 1.72x (1.58x) / 1.63x (1.55x) / 1.67x (1.60x) |
| q/k/v A6, down A6 | 14.40 (14.16–14.62) / 14.01 (13.96–14.25) / 14.14 (14.00–14.42) | 1.63x (1.53x) / 1.57x (1.49x) / 1.61x (1.54x) |
| q/k/v A6, down A5 Winsor | 14.03 (13.68–15.05) / 13.63 (13.59–14.23) / 13.76 (13.65–14.29) | 1.67x (1.49x) / 1.61x (1.49x) / 1.65x (1.56x) |
| q/k/v A6, down A5 + split 32 | 14.80 (13.93–15.09) / 13.87 (13.74–14.28) / 14.10 (13.88–14.51) | 1.58x (1.48x) / 1.58x (1.49x) / 1.61x (1.53x) |
| q/k/v A6, down A5 + split 64 | 14.18 (14.09–14.57) / 14.45 (13.96–14.91) / 14.15 (14.04–14.40) | 1.65x (1.54x) / 1.52x (1.42x) / 1.61x (1.55x) |
| q/k/v A6, down A5 + split 128 | 14.56 (14.46–16.69) / 14.91 (14.60–15.48) / 14.87 (14.43–15.09) | 1.61x (1.34x) / 1.47x (1.37x) / 1.53x (1.48x) |
| **q/k/v A6 + split 32, down A5 + split 32** | 14.82 (14.45–15.93) / 14.40 (14.30–14.51) / 14.60 (14.42–14.87) | 1.58x (1.41x) / 1.52x (1.46x) / 1.56x (1.50x) |
| q/k/v tmac_g8, down A5 + split 32 | 15.68 (15.46–16.03) / 15.23 (15.19–15.32) / 15.50 (15.34–16.00) | 1.49x (1.40x) / 1.44x (1.39x) / 1.47x (1.39x) |
| q/k/v + o tmac_g8, down A5 + split 32 | 16.77 (16.36–18.57) / 16.16 (16.11–16.36) / 16.31 (16.18–16.82) | 1.40x (1.21x) / 1.36x (1.30x) / 1.39x (1.32x) |
| q/k/v + o + down tmac_g8 (gate/up A5) | 19.98 (18.82–20.80) / 18.27 (18.23–18.42) / 19.18 (18.78–22.39) | 1.17x (1.08x) / 1.20x (1.15x) / 1.19x (0.99x) |
| all-`tmac_g8` baseline | 23.44 (22.40–24.32) / 21.95 (21.22–24.41) / 22.75 (22.25–23.81) | 1.00x |
| `f32_resident` | 1.65 (1.46–2.80) / 1.47 (1.36–1.55) / 1.49 (1.41–2.60) | 14.2x / 14.9x / 15.3x |

Interleaved check (median of 7 per-round medians, min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| tmac_g8 | 21.70 (21.47–23.56) | 1.00x |
| all A5 | 13.14 (12.95–13.30) | 1.65x |
| q/k/v A6, down A5 | 13.46 (13.32–14.08) | 1.61x |
| **q/k/v A6 + split 32, down A5 + split 32** | 14.55 (14.22–14.63) | 1.49x |

## Verdict

**inconclusive**. The accuracy bar is met. The speed sits on the 1.5x line.

1. The tail is real. At 128 tokens, q/k/v A6 has 5/128 and 10/128 tokens under 0.98, with worst 0.9159 / 0.8825. That fails the ≥ 97% bar on one seed, and the 32-token 0.9788 was optimistic.
2. Down-only fixes cannot fix it. Down A6, Winsor and split 32/64/128 move the pooled cos by up to +0.003, but the worst token stays at 0.88–0.92 and 4–6/128 tokens stay under 0.98. Winsor (q=0.998) does nothing.
3. The tail tokens are heavy-tailed q/k/v inputs. Exact `tmac_g8` at q/k/v alone takes the worst token to 0.9934. The same result comes for almost no cost from a **32-channel outlier split at the q/k/v input** (A6 on the rest): worst 0.9927, p5 0.9943, pooled 0.9952 / 0.9952, 0/256 under 0.98.
4. Speed: that config runs at 1.52–1.58x in the three runs. All three medians are ≥ 1.5x, but the worst-case ratios are below 1.5x. In the interleaved run it is 1.49x, against 1.61x for q/k/v A6 without the split. The `tmac_g8` baseline ran 21.7–23.4 ms here, against ~21.0 ms in the prior note. Each split site costs about 0.5 ms of argpartition plus column unpack.
5. As before, `f32_resident` is ~14x faster than `tmac_g8` at 16x the memory. Any win is only against the packed fallback.

This is the first mix since 11-13 10:55 that passes the every-token accuracy bar while staying near 1.5x. Keep `tmac_g8` as the packed default until the split is cheaper.

## Pair files this wave

Quant Chief 05:08 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so no pair files this wave. Roadmap blobs are unchanged (JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923). JuniorLLM head at clone was 2956628 (the prior findings note). This note touches only `junior_osai/vault/findings/`.

## Next question

Can the outlier split get cheap enough to clear 1.5x with margin? Options: replace argpartition with one fixed per-row threshold (|x| > t·absmean); drop q/k/v to A5 + split (one plane fewer); or split only at q/k/v and leave down at plain A5. Does any of these keep 0/256 under 0.98 at ≥ 1.6x interleaved?

close: 5dfcfe101b402078 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-14_0255)
