---
local: true
date: 2026-11-14
wave: "2026-11-14 10:55 (ran ~10:42–10:50 MT; Chief last_run 2026-11-14T17:06:00Z ~10:06 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, a6, bitplane-activation, outlier-split, shared-split, threshold-split, attn-qkv, tmac-g8, numpy, viable]
---

# Q/k/v at A5 with one shared 32-channel outlier split (down left at plain A5) passes every token, with 0/256 under 0.98 and the worst at 0.9868. It runs 1.57–1.65x faster than `tmac_g8` in three runs and 1.57x in an interleaved check, on the same 2-bit resident bytes. That makes it viable against the packed fallback, though it misses the ≥ 1.6x interleaved margin

## Question

The prior note [[2026-11-14_0555_outlier-split-qkv-a6-down-a5-larger-eval-2x128-full-bitnetd-layer-b1-vs-tmac-g8]] found that a 32-channel outlier split at q/k/v (A6) and down (A5) clears every token. Its speed was 1.49–1.58x, right on the bar. Its next question was whether the split can be made cheap enough to clear 1.5x with margin. It proposed three ways:

- a fixed per-row threshold (|x| > t·absmean) instead of argpartition,
- q/k/v at A5 + split (one plane fewer),
- splitting only at q/k/v, with down at plain A5.

The goal was 0/256 under 0.98 at ≥ 1.6x interleaved. Quant Chief `last_tips.json` `last_run` 2026-11-14T17:06:00Z (~10:06 MT) wrote pair_A/pair_B **no gap**, so this note uses the lane's own next question.

## Method

The layer is the same bitnetd GQA layer as the last five notes, with a new seed, 202611141055:

- hidden 2560, 20 q heads × 128, 5 KV heads, FFN 6912, ctx 1
- RMSNorm, SiLU·up, both residuals
- absmean-ternarised N(0, 0.02) weights
- mask/sign packed GEMV with 2 popcounts per plane, T=1024
- o and gate/up at A5 everywhere

New this run:

- **Shared split** (`split_shared`). The 05:55 code ran argpartition, zeroing and the Ak quant once per projection, three times at q/k/v. Here it runs once and feeds all three GEMVs. Each projection then adds back its own exact Σ x_i · trit(W[:, i]) · γ from columns unpacked out of the same mask/sign bytes. The result is numerically identical to the per-projection split (see the accuracy table).
- **Threshold split**. Select = `flatnonzero(|x| > t · mean|x|)` per row, for t = 4 and 6, so the count varies per token (reported below).
- Q/k/v at A5 + split, and down left at plain A5.

c is calibrated per site and option on a 16-token draw. Accuracy uses two 128-token t(4)·0.05 eval draws (seeds +2, +3), measured as per-token cos of the layer delta vs the weight-only W1.58 layer. Timing:

- a full layer step at B=1, 9 repeats × 5 steps after 3 warmups, over three full runs;
- an interleaved check, `tinter.py`: 7 rounds in alternating order in the same process.

Environment: numpy 2.5.4, single process, 8-core virtualised Xeon, load avg 0.8–2.1.

Command (offline, no network, nothing bound):

```
cd /workspace/bench1055_1114 && for i in 1 2 3; do python3 bench.py results_run$i.json > run$i.txt 2>&1; done && python3 tinter.py tinter.json
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Cheaper outlier split for the full-layer path (10:55 11/14): shared q/k/v split (one select + one quant
for all three projections instead of three), fixed |x| > t*absmean threshold select instead of argpartition,
q/k/v A5 + split (one plane fewer), and split at q/k/v only with plain A5 down. Base layer as in 05:55 11/14. Full bitnetd
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
SEED = 202611141055
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
STAT = {"thr_k": []}
def split_shared(x, c, sel, kb, names):
    """One select + one Ak quant shared by every projection in names. sel=("k", kout) argpartition top-k,
    sel=("t", t) fixed threshold |x| > t*mean|x| per row (variable count)."""
    B, Fi = x.shape
    if sel[0] == "k":
        I = np.argpartition(np.abs(x), Fi - sel[1], axis=1)[:, Fi - sel[1]:]; idxs = [I[b] for b in range(B)]
    else:
        ax = np.abs(x); lim = sel[1] * ax.mean(1)
        idxs = [np.flatnonzero(ax[b] > lim[b]) for b in range(B)]
    xr = x.copy()
    for b in range(B): xr[b, idxs[b]] = 0.0
    pl, hh = quantk(xr, c, kb)
    outs = []
    for n in names:
        P = PW[n]; y = a5gemv(pl, hh, P); wm8, ws8 = P["wm8"], P["ws8"]
        for b in range(B):
            ix = idxs[b]
            if ix.size == 0: continue
            byt = ix >> 3; sh = (7 - (ix & 7)).astype(np.uint8)
            m = (wm8[:, byt] >> sh) & 1; s = (ws8[:, byt] >> sh) & 1
            t = m.astype(np.float32) * (1 - 2 * s.astype(np.float32))
            y[b] += (t @ x[b, ix]) * np.float32(P["gw"])
        outs.append(y)
    if sel[0] == "t" and B > 1: STAT["thr_k"].extend(int(i.size) for i in idxs)
    return outs
def site_mm(spec, x, names):
    """("tmac",) exact f32 tmac_g8; (k, c) Ak absmean; ("w", c) A5 Winsor quantile c; ("s", c, kout[, k]) Ak + outlier split."""
    if spec[0] == "tmac": return [np.concatenate([tmac_g8(PW[n], x[i]) for i in range(x.shape[0])]) for n in names]
    if spec[0] == "S": return split_shared(x, spec[1], ("k", spec[2]), spec[3], names)
    if spec[0] == "T": return split_shared(x, spec[1], ("t", spec[2]), spec[3], names)
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
THR = (4, 6)
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
    cal["attn_6"] = calib("attn", lambda c: (6, c), GRID6)
    cal["down_s32"] = calib("down", lambda c: ("s", c, 32), GRID5)
    cal["attn_s6"] = calib("attn", lambda c: ("s", c, 32, 6), GRID6)
    cal["attn_S6"] = calib("attn", lambda c: ("S", c, 32, 6), GRID6)
    cal["attn_S5"] = calib("attn", lambda c: ("S", c, 32, 5), GRID5 + [0.02, 0.03])
    for t in THR: cal[f"attn_T{t}_6"] = calib("attn", lambda c, t=t: ("T", c, t, 6), GRID6)
    cal["attn_T4_5"] = calib("attn", lambda c: ("T", c, 4, 5), GRID5 + [0.02, 0.03])
    print("cal", cal, flush=True)
    sp = lambda key: cal[key][0]
    base = {s: sp(f"{s}_5") for s in site_proj}
    qa = {**base, "attn": sp("attn_6")}; ds = sp("down_s32")
    variants = {"allA5": base, "attnA6": qa,
                "old_attnA6s32_downS32": {**qa, "attn": sp("attn_s6"), "down": ds},
                "sh_attnA6S32_downS32": {**qa, "attn": sp("attn_S6"), "down": ds},
                "sh_attnA6S32_downA5": {**qa, "attn": sp("attn_S6")},
                "sh_attnA5S32_downS32": {**base, "attn": sp("attn_S5"), "down": ds},
                "sh_attnA5S32_downA5": {**base, "attn": sp("attn_S5")}}
    for t in THR:
        variants[f"thr{t}_attnA6_downS32"] = {**qa, "attn": sp(f"attn_T{t}_6"), "down": ds}
        variants[f"thr{t}_attnA6_downA5"] = {**qa, "attn": sp(f"attn_T{t}_6")}
    variants["thr4_attnA5_downA5"] = {**base, "attn": sp("attn_T4_5")}
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
                if n.startswith("thr"): res["acc"][n][f"seed{es}"]["thr_k_mean"] = statistics.mean(STAT["thr_k"]); res["acc"][n][f"seed{es}"]["thr_k_max"] = max(STAT["thr_k"])
                STAT["thr_k"].clear()
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
NAMES = ("allA5", "attnA6", "old_attnA6s32_downS32", "sh_attnA6S32_downA5", "sh_attnA5S32_downA5", "thr6_attnA6_downA5")
cfg = {n: {s: tuple(v) for s, v in cal[n].items()} for n in NAMES}
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

Every variant has the same resident weight bytes per layer:

- packed mask/sign: 17,367,040 B (+ Σw 182,272 B)
- `tmac_g8` PT: 17,367,040 B
- int8: 69,468,160 B (4.00x)
- f32: 277,872,640 B (16.00x)

The split only adds per-token scratch.

Calibrated values, identical in all three runs:

- A5: q/k/v c=0.28, o 0.12, gate/up 0.12, down 0.38
- A6 q/k/v: 0.2
- down A5 split 32: 0.26
- q/k/v A6 split 32: 0.07 (shared = per-projection)
- q/k/v A5 shared split 32: 0.14
- threshold A6: t=4 0.07, t=6 0.09
- threshold A5: t=4 0.12

Accuracy is the layer delta vs the weight-only layer, given as eval seed +2 / seed +3 (128 tokens each). "split ch" is the number of channels split per token at q/k/v.

| config | split ch | pooled cos | p5 token | worst token | tokens < 0.98 |
|---|---|---|---|---|---|
| all A5 (no split) | 0 | 0.9769 / 0.9783 | 0.9520 / 0.9579 | 0.8025 / 0.8558 | 43/128 / 48/128 |
| q/k/v A6, down A5 (no split) | 0 | 0.9851 / 0.9864 | 0.9752 / 0.9763 | 0.8408 / 0.8815 | 10/128 / 10/128 |
| 05:55 best: q/k/v A6 + split 32 per projection, down A5 + split 32 | 32 | 0.9952 / 0.9952 | 0.9943 / 0.9941 | 0.9937 / 0.9932 | 0/128 / 0/128 |
| q/k/v A6 + shared split 32, down A5 + split 32 | 32 | 0.9952 / 0.9952 | 0.9943 / 0.9941 | 0.9937 / 0.9932 | 0/128 / 0/128 |
| q/k/v A6 + shared split 32, down A5 | 32 | 0.9928 / 0.9927 | 0.9912 / 0.9904 | 0.9883 / 0.9896 | 0/128 / 0/128 |
| q/k/v A5 + shared split 32, down A5 + split 32 | 32 | 0.9939 / 0.9938 | 0.9929 / 0.9927 | 0.9926 / 0.9917 | 0/128 / 0/128 |
| **q/k/v A5 + shared split 32, down A5** | 32 | 0.9914 / 0.9913 | 0.9901 / 0.9893 | 0.9868 / 0.9875 | 0/128 / 0/128 |
| q/k/v A6 + threshold split t=4, down A5 + split 32 | 41.0 / 40.8 (max 53) | 0.9953 / 0.9952 | 0.9944 / 0.9942 | 0.9941 / 0.9935 | 0/128 / 0/128 |
| q/k/v A6 + threshold split t=4, down A5 | 41.0 / 40.8 (max 53) | 0.9928 / 0.9927 | 0.9914 / 0.9908 | 0.9883 / 0.9897 | 0/128 / 0/128 |
| q/k/v A6 + threshold split t=6, down A5 + split 32 | 9.8 / 9.9 (max 19) | 0.9949 / 0.9948 | 0.9940 / 0.9936 | 0.9933 / 0.9927 | 0/128 / 0/128 |
| q/k/v A6 + threshold split t=6, down A5 | 9.8 / 9.9 (max 19) | 0.9924 / 0.9923 | 0.9908 / 0.9903 | 0.9874 / 0.9895 | 0/128 / 0/128 |
| q/k/v A5 + threshold split t=4, down A5 | 41.0 / 40.8 (max 53) | 0.9915 / 0.9915 | 0.9900 / 0.9895 | 0.9862 / 0.9882 | 0/128 / 0/128 |

The full layer step at B=1 is in ms, median (min–max), for runs 1 / 2 / 3. Ratio = all-`tmac_g8` median / variant median; "worst" = `tmac_g8` min / variant max.

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| all A5 (no split) | 12.81 (12.66–13.17) / 12.71 (12.65–12.98) / 12.68 (12.62–12.74) | 1.71x (1.61x) / 1.63x (1.59x) / 1.70x (1.63x) |
| q/k/v A6, down A5 (no split) | 12.99 (12.96–13.08) / 13.07 (12.99–13.26) / 13.00 (12.98–13.05) | 1.68x (1.63x) / 1.59x (1.56x) / 1.65x (1.59x) |
| 05:55 best: q/k/v A6 + split 32 per projection, down A5 + split 32 | 14.22 (14.03–14.73) / 14.13 (14.09–14.45) / 14.09 (14.06–14.20) | 1.54x (1.44x) / 1.47x (1.43x) / 1.53x (1.46x) |
| q/k/v A6 + shared split 32, down A5 + split 32 | 14.02 (13.95–14.34) / 14.01 (13.91–14.38) / 13.93 (13.90–14.02) | 1.56x (1.48x) / 1.48x (1.44x) / 1.54x (1.48x) |
| q/k/v A6 + shared split 32, down A5 | 13.68 (13.59–13.96) / 13.63 (13.57–14.04) / 13.61 (13.55–13.78) | 1.60x (1.52x) / 1.52x (1.48x) / 1.58x (1.50x) |
| q/k/v A5 + shared split 32, down A5 + split 32 | 13.66 (13.59–14.62) / 13.64 (13.56–14.06) / 13.56 (13.53–13.67) | 1.60x (1.45x) / 1.52x (1.47x) / 1.59x (1.52x) |
| **q/k/v A5 + shared split 32, down A5** | 13.28 (13.21–13.46) / 13.27 (13.23–13.76) / 13.24 (13.14–14.54) | 1.65x (1.58x) / 1.57x (1.51x) / 1.62x (1.42x) |
| q/k/v A6 + threshold split t=4, down A5 + split 32 | 13.91 (13.86–13.95) / 13.90 (13.86–14.38) / 13.79 (13.76–13.85) | 1.57x (1.52x) / 1.49x (1.44x) / 1.56x (1.50x) |
| q/k/v A6 + threshold split t=4, down A5 | 13.54 (13.45–14.99) / 13.54 (13.51–13.96) / 13.47 (13.44–13.63) | 1.62x (1.42x) / 1.53x (1.48x) / 1.60x (1.52x) |
| q/k/v A6 + threshold split t=6, down A5 + split 32 | 13.70 (13.68–13.84) / 13.78 (13.70–14.14) / 13.68 (13.67–13.72) | 1.60x (1.54x) / 1.51x (1.46x) / 1.57x (1.51x) |
| q/k/v A6 + threshold split t=6, down A5 | 13.40 (13.31–14.75) / 13.61 (13.39–15.95) / 13.30 (13.29–13.38) | 1.63x (1.44x) / 1.53x (1.30x) / 1.62x (1.55x) |
| q/k/v A5 + threshold split t=4, down A5 | 13.79 (13.18–15.06) / 13.72 (13.38–16.27) / 13.12 (13.05–13.25) | 1.59x (1.41x) / 1.51x (1.27x) / 1.64x (1.56x) |
| all-`tmac_g8` baseline | 21.88 (21.26–22.63) / 20.77 (20.71–21.92) / 21.51 (20.72–22.12) | 1.00x |
| `f32_resident` | 1.44 (1.37–2.28) / 1.37 (1.36–1.61) / 1.50 (1.35–1.55) | 15.2x / 15.2x / 14.3x |

Interleaved check, median of 7 per-round medians (min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 20.83 (20.74–21.93) | 1.00x |
| all A5 (no split) | 12.71 (12.65–13.06) | 1.64x |
| q/k/v A6, down A5 (no split) | 13.10 (12.99–13.51) | 1.59x |
| 05:55 best: q/k/v A6 + split 32 per projection, down A5 + split 32 | 14.06 (13.97–14.44) | 1.48x |
| q/k/v A6 + shared split 32, down A5 | 13.57 (13.51–13.88) | 1.53x |
| **q/k/v A5 + shared split 32, down A5** | 13.28 (13.13–13.51) | 1.57x |
| q/k/v A6 + threshold split t=6, down A5 | 13.63 (13.25–14.65) | 1.53x |

## Verdict

**Viable**, against the packed `tmac_g8` fallback at equal resident bytes.

1. **The win comes from dropping a plane and the down split, not from the split mechanics.** Sharing the split across q/k/v gives bit-identical accuracy. It saves only ~0.1–0.2 ms (1.47–1.54x → 1.48–1.56x).
2. **The threshold select is no cheaper than argpartition.** t=6 averages ~10 channels and is slightly faster, but noisier (worst-case ratios down to ~1.3x).
3. **Down needs no split once q/k/v is split.** Every variant with a q/k/v split, including A5 at q/k/v, keeps 0/256 tokens under 0.98.
4. **The cheapest passing mix is **q/k/v A5 + shared split 32, down A5**.**
   - Accuracy: pooled 0.9914 / 0.9913, p5 0.9901 / 0.9893, worst 0.9868.
   - Speed: 1.57–1.65x in three runs and 1.57x interleaved, against 1.48x for the 05:55 config in the same interleaved run.
   - Overhead: all A5 with no split ran 1.64x interleaved, so the split now costs only ~0.6 ms per layer.
   - That clears 1.5x on every median. The worst-case ratios (`tmac_g8` min / variant max) still dip to 1.42–1.58x, and it misses the ≥ 1.6x interleaved stretch target.
5. **The caveat stands.** `f32_resident` is still ~15x faster than `tmac_g8`, at 16x the memory. This is a win for the memory-bound packed path only. The test uses synthetic Gaussian weights and t(4) activations; it does not use real BitNet tensors.

Recommendation: q/k/v A5 + shared 32-channel split, with o, gate/up and down at A5, should replace `tmac_g8` as the packed B=1 default in the bitnetd/i2sd prototype path.

## Pair files this wave

Quant Chief 10:06 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave. Roadmap blobs are unchanged: JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923. This note touches only `junior_osai/vault/findings/`.

## Next question

Does the q/k/v A5 + shared split 32 mix hold on real BitNet-b1.58-2B-4T-shaped statistics? The test would use a local, non-downloaded activation trace or per-channel outlier profile, if one exists under /workspace, or else a harder synthetic draw: t(3), plus a few fixed hot channels shared across tokens. Separately, can the o / gate-up sites drop to A4 + split for more speed while keeping 0/256 under 0.98?

close: 30c7b0ff0a0f516c (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-14_0555)
