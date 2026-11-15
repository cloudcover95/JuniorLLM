---
local: true
date: 2026-11-14
wave: "2026-11-14 18:55 (ran ~18:45–19:00 MT; Chief last_run 2026-11-15T01:06:00Z ~18:06 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: inconclusive
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, a4, bitplane-activation, outlier-split, static-hot-list, index-reuse, transposed-planes, bytes-vs-speed, tmac-g8, numpy, inconclusive]
---

# The all-sites split reaches the 14:55 target (0/1024 tokens under 0.98 at 1.63x interleaved vs `tmac_g8`) only by gathering split columns from a transposed copy of the planes, which doubles packed weight bytes (17.55 → 34.92 MB). Static hot-channel lists (1.70x) and residual-index reuse lose tokens on heavy tails

## Question

The prior note [[2026-11-14_1455_harder-activations-t3-hot-channels-qkv-split-robustness-o-ffn-a4-split-full-bitnetd-layer-b1-vs-tmac-g8]] found that splitting every site (q/k/v A5 + shared split 32, o and gate/up A4 + split 32, down A5 + split 32) was the only mix with 0/1024 tokens under 0.98 across t4/t3/t3hot8/t3hot48, but it ran only 1.50–1.53x vs `tmac_g8`. Its next question: can a cheaper split reach **0/1024 at ≥ 1.6x interleaved**? Options: (a) a fixed calibrated hot-channel list at o and gate/up with no per-token select; (b) reuse the residual top-32 index set from q/k/v at gate/up. Added here: (c) keep the per-token argpartition but gather the 32 exact ternary columns from a transposed copy of the mask/sign planes (contiguous rows instead of a strided byte gather). Quant Chief `last_tips.json` `last_run` 2026-11-15T01:06:00Z (~18:06 MT) wrote pair_A/pair_B **no gap**, so this note uses the lane's own next question.

## Method

- **Layer, weights, kernels, draws, metric:** unchanged from 14:55. `base.py` is copied byte for byte (sha256[:16] `a38c8245cd13d800`). One full bitnetd decoder layer at B=1 (hidden 2560, GQA 20 q / 5 KV heads, FFN 6912), absmean ternary weights as 2-bit mask/sign planes. Accuracy is per-token cosine of the layer delta against the weight-only W1.58 layer, on 2 × 128 tokens per draw (1,024 tokens total).
- **Static list (F):** per site, the top-k channels by mean |x| on the 16-token calibration draw. Those channels are zeroed before quantising and added back exactly through ternary columns cached as f32 (gw-scaled). The cache is extra resident bytes and is counted below.
- **Reuse (R):** gate/up use the top-32 index set picked on the q/k/v input (h1 = rms(x)), applied to h2 = rms(x + o).
- **Transposed (X):** same per-token argpartition split as 14:55, but columns come from `wmT/wsT` = packbits of the transposed mask/sign planes, one contiguous row per input channel (`np.unpackbits(wmT[ix], count=out)`). The row-major planes stay for the bitplane GEMV, so packed bytes double. The math is identical to 14:55, so accuracy matches the reference exactly.
- **Timing:** B=1, 3 runs (9 repeats × 5 passes, 3 warm), plus a 7-round interleaved check. Run 1 (with eval) took 2m59s. Setup: 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench1855_1114`, next to `base.py`):

```
python3 bench.py results_run1.json          # calibration + 4 draws x 2x128 eval + timing
python3 bench.py results_run2.json noeval   # timing only
python3 bench.py results_run3.json noeval
python3 tinter.py                           # interleaved timing
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Cheaper all-sites outlier split for the packed B=1 bitnetd layer (18:55 11/14). Same layer, weights, kernels,
token draws (t4, t3, t3hot8, t3hot48) and metric as the 14:55 note (base.py copied verbatim from 10:55).
Reference = 14:55 all-sites split: q/k/v A5 + shared argpartition split 32, o and gate/up A4 + split 32, down A5 + split 32.
Cheaper options tested:
  F  static calibrated channel list (top-k mean|x| on the 16-token calibration draw), no per-token select; the k exact
     ternary columns are cached as f32 (extra resident bytes, counted).
  R  gate/up reuse the residual top-32 index set chosen for q/k/v (rms is a per-token scale, x2 = x + o).
  X  per-token argpartition split, but columns gathered from a transposed copy of the mask/sign planes
     (contiguous rows; +100% packed bytes, counted).
numpy-only, offline. Usage: python3 bench.py results_runN.json [noeval]"""
import json, statistics, sys
import numpy as np
import base
from base import site_mm, wo_mm, tmac_g8, f32res, timed, cos, rowcos, rms, silu, quantk, a5gemv, PW, GRID5, D, KVH, QH, HD
SEED = base.SEED
GRID4 = [round(0.06 + 0.03 * i, 2) for i in range(19)]
HOT = {"t4": (4, 0), "t3": (3, 0), "t3hot8": (3, 8), "t3hot48": (3, 48)}
def tokf(df, nh):
    hot = np.random.default_rng(777).choice(D, nh, replace=False) if nh else None
    def f(r, n):
        x = r.standard_t(df, (n, D)) * 0.05
        if nh: x[:, hot] *= 15.0
        return x.astype(np.float32)
    return f
site_proj = {"attn": ("q", "k", "v"), "o": ("o",), "ffn": ("gate", "up"), "down": ("down",)}
STATIC = {}
def add_transposed():
    for n, P in PW.items():
        t = P["wm8"].shape  # (out, in/8)
        m = np.unpackbits(P["wm8"], axis=1); s = np.unpackbits(P["ws8"], axis=1)
        P["wmT"] = np.packbits(m.T, axis=1); P["wsT"] = np.packbits(s.T, axis=1)
def cols_rowmajor(P, ix):
    byt = ix >> 3; sh = (7 - (ix & 7)).astype(np.uint8)
    m = (P["wm8"][:, byt] >> sh) & 1; s = (P["ws8"][:, byt] >> sh) & 1
    return (m.astype(np.float32) * (1 - 2 * s.astype(np.float32))).T  # (k, out)
def cols_T(P, ix):
    out = P["wm8"].shape[0]
    m = np.unpackbits(P["wmT"][ix], axis=1, count=out); s = np.unpackbits(P["wsT"][ix], axis=1, count=out)
    return m.astype(np.float32) - 2 * (m & s).astype(np.float32)
def dyn(x, c, kout, kb, names, how="row", idxs=None):
    B, Fi = x.shape
    if idxs is None:
        I = np.argpartition(np.abs(x), Fi - kout, axis=1)[:, Fi - kout:]; idxs = [I[b] for b in range(B)]
    xr = x.copy()
    for b in range(B): xr[b, idxs[b]] = 0.0
    pl, hh = quantk(xr, c, kb); outs = []
    for n in names:
        P = PW[n]; y = a5gemv(pl, hh, P); g = np.float32(P["gw"])
        for b in range(B):
            ix = idxs[b]; t = cols_T(P, ix) if how == "T" else cols_rowmajor(P, ix)
            y[b] += (x[b, ix] @ t) * g
        outs.append(y)
    return outs, idxs
def static(x, c, key, kb, names):
    S = STATIC[key]; ix = S["idx"]
    xr = x.copy(); xr[:, ix] = 0.0
    pl, hh = quantk(xr, c, kb)
    return [a5gemv(pl, hh, PW[n]) + x[:, ix] @ S[n] for n in names]
def make_static(key, xcal, k, names):
    ix = np.sort(np.argsort(np.abs(xcal).mean(0))[-k:])
    S = {"idx": ix}
    for n in names: S[n] = np.ascontiguousarray(cols_rowmajor(PW[n], ix) * np.float32(PW[n]["gw"]))
    STATIC[key] = S
def smm(spec, x, names, ctx):
    k = spec[0]
    if k == "S": o, ctx["idx"] = dyn(x, spec[1], spec[2], spec[3], names); return o
    if k == "X": o, ctx["idx"] = dyn(x, spec[1], spec[2], spec[3], names, "T"); return o
    if k == "R": return dyn(x, spec[1], 32, spec[2], names, idxs=ctx["idx"])[0]
    if k == "F": return static(x, spec[1], spec[2], spec[3], names)
    if k == "s": return dyn(x, spec[1], spec[2], 5, names)[0]
    if k == "sX": return dyn(x, spec[1], spec[2], 5, names, "T")[0]
    return site_mm(spec, x, names)
def layer(x, mm, cfg=None):
    ctx = {}; h1 = rms(x)
    if cfg: q, k, v = smm(cfg["attn"], h1, ("q", "k", "v"), ctx)
    else: q, k, v = (mm(PW[n], h1) for n in ("q", "k", "v"))
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)
    o = smm(cfg["o"], att, ("o",), ctx)[0] if cfg else mm(PW["o"], att)
    x2 = x + o; h2 = rms(x2)
    if cfg: g, u = smm(cfg["ffn"], h2, ("gate", "up"), ctx)
    else: g, u = mm(PW["gate"], h2), mm(PW["up"], h2)
    z = (silu(g) * u).astype(np.float32)
    dn = smm(cfg["down"], z, ("down",), ctx)[0] if cfg else mm(PW["down"], z)
    return x2 + dn, dict(q=q, k=k, v=v, o=o, gate=g, up=u, down=dn), dict(attn=h1, o=att, ffn=h2, down=z)
def calibrate(dn, tok):
    xc = tok(np.random.default_rng(SEED + 1), 16)
    _, ycw, sites = layer(xc, wo_mm)
    for s, kk in (("attn", 32), ("o", 32), ("ffn", 32), ("ffn", 64), ("down", 32)):
        make_static(f"{dn}_{s}_{kk}", sites[s], kk, site_proj[s])
    def calib(s, mk, grid, cfg_attn=None):
        sc = {}
        for c in grid:
            ctx = {}
            if cfg_attn is not None: smm(cfg_attn, sites["attn"], ("q",), ctx)
            sc[c] = statistics.mean(cos(smm(mk(c), sites[s], (p,), ctx)[0], ycw[p]) for p in site_proj[s])
        c = max(sc, key=sc.get); return [mk(c), sc[c]]
    cal = {}
    cal["attn_S5"] = calib("attn", lambda c: ("S", c, 32, 5), GRID5 + [0.02, 0.03])
    cal["attn_F5"] = calib("attn", lambda c: ("F", c, f"{dn}_attn_32", 5), GRID5 + [0.02, 0.03])
    cal["o_S4"] = calib("o", lambda c: ("S", c, 32, 4), GRID4)
    cal["o_F4"] = calib("o", lambda c: ("F", c, f"{dn}_o_32", 4), GRID4)
    cal["ffn_S4"] = calib("ffn", lambda c: ("S", c, 32, 4), GRID4)
    cal["ffn_F4"] = calib("ffn", lambda c: ("F", c, f"{dn}_ffn_32", 4), GRID4)
    cal["ffn_F4_64"] = calib("ffn", lambda c: ("F", c, f"{dn}_ffn_64", 4), GRID4)
    cal["ffn_R4"] = calib("ffn", lambda c: ("R", c, 4), GRID4, cfg_attn=tuple(cal["attn_S5"][0]))
    cal["down_s32"] = calib("down", lambda c: ("s", c, 32), GRID5)
    cal["down_F5"] = calib("down", lambda c: ("F", c, f"{dn}_down_32", 5), GRID5)
    return cal
def variants(cal):
    sp = lambda k: tuple(cal[k][0])
    ref = {"attn": sp("attn_S5"), "o": sp("o_S4"), "ffn": sp("ffn_S4"), "down": sp("down_s32")}
    X = lambda s: ("X",) + s[1:]
    return {"ref_allsplit_1455": ref,
            "rec_1455_downA5": {**ref, "down": ("5x",)},  # filled below
            "static_o_ffn32": {**ref, "o": sp("o_F4"), "ffn": sp("ffn_F4")},
            "static_o32_ffn64": {**ref, "o": sp("o_F4"), "ffn": sp("ffn_F4_64")},
            "static_all": {"attn": sp("attn_F5"), "o": sp("o_F4"), "ffn": sp("ffn_F4_64"), "down": sp("down_F5")},
            "reuse_ffn": {**ref, "ffn": sp("ffn_R4")},
            "transposed_allsplit": {"attn": X(sp("attn_S5")), "o": X(sp("o_S4")), "ffn": X(sp("ffn_S4")), "down": ("sX",) + sp("down_s32")[1:]},
            "transposed_reuse": {"attn": X(sp("attn_S5")), "o": X(sp("o_S4")), "ffn": sp("ffn_R4"), "down": ("sX",) + sp("down_s32")[1:]}}
def fix(v, cal):
    v["rec_1455_downA5"]["down"] = tuple(cal["down_5"][0]); return v
def main():
    _, by = base.build(); add_transposed()
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "cal": {}, "variants": {}, "acc": {}, "time": {}}
    V = {}
    for dn, (df, nh) in HOT.items():
        tok = tokf(df, nh); cal = calibrate(dn, tok)
        xc = tok(np.random.default_rng(SEED + 1), 16); z = layer(xc, wo_mm)[2]["down"]
        sc = {c: cos(site_mm((5, c), z, ("down",))[0], z @ PW["down"]["Wt"].T) for c in GRID5}
        cal["down_5"] = [[5, max(sc, key=sc.get)], max(sc.values())]
        res["cal"][dn] = {k: [list(v[0]), v[1]] for k, v in cal.items()}; V[dn] = fix(variants(cal), cal)
        res["variants"][dn] = {k: {s: list(v) for s, v in c.items()} for k, c in V[dn].items()}
        print("cal", dn, res["cal"][dn], flush=True)
    sb = {}
    for key, S in STATIC.items():
        if key.startswith("t4_"): sb[key] = sum(v.nbytes for k, v in S.items() if k != "idx")
    res["bytes"]["static_cols_f32_t4"] = sb
    res["bytes"]["transposed_planes"] = sum(P["wmT"].nbytes + P["wsT"].nbytes for P in PW.values())
    print("bytes", res["bytes"], flush=True)
    if "noeval" not in sys.argv:
        for dn, (df, nh) in HOT.items():
            tok = tokf(df, nh)
            for es in (2, 3):
                xe = tok(np.random.default_rng(SEED + es), 128)
                yw = np.concatenate([layer(xe[i:i+32], wo_mm)[0] for i in range(0, 128, 32)])
                for n, cfg in V[dn].items():
                    ya = np.concatenate([layer(xe[i:i+32], None, cfg)[0] for i in range(0, 128, 32)])
                    d1, d0 = ya - xe, yw - xe; rc = rowcos(d1, d0)
                    r = {"delta_pooled": cos(d1, d0), "delta_min": min(rc), "rows_below_098": int(sum(v < 0.98 for v in rc)), "n": 128,
                         "rel_l2": float(np.linalg.norm((d1 - d0).astype(np.float64)) / np.linalg.norm(d0.astype(np.float64)))}
                    res["acc"].setdefault(dn, {}).setdefault(n, {})[f"seed{es}"] = r
                    print(dn, n, es, r, flush=True)
    x1 = tokf(4, 0)(np.random.default_rng(SEED + 9), 1)
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, cfg in V["t4"].items(): fns["q_" + n] = (lambda cfg=cfg: layer(x1, None, cfg))
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
"""Interleaved timing: 7 rounds, each round times every config (REPEATS=5 x PASSES=5) in rotated order; reports median of round medians."""
import json, statistics
import numpy as np
import base, bench
base.build(); bench.add_transposed(); base.REPEATS = 5
cal = bench.calibrate("t4", bench.tokf(4, 0)); cal["down_5"] = [[5, 0.38], 0]
V = bench.fix(bench.variants(cal), cal)
x1 = bench.tokf(4, 0)(np.random.default_rng(base.SEED + 9), 1)
fns = {"tmac_g8": lambda: bench.layer(x1, base.tmac_g8)}
for n in ("ref_allsplit_1455", "rec_1455_downA5", "static_o_ffn32", "static_o32_ffn64", "transposed_allsplit", "transposed_reuse"):
    fns[n] = (lambda cfg=V[n]: bench.layer(x1, None, cfg))
names = list(fns); rounds = {n: [] for n in names}
for r in range(7):
    for n in names[r % len(names):] + names[:r % len(names)]: rounds[n].append(base.timed(fns[n])["median_ms"])
out = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
for n in names: out[n]["vs_tmac"] = out["tmac_g8"]["median_ms"] / out[n]["median_ms"]
print(json.dumps(out, indent=1)); json.dump(out, open("tinter.json", "w"), indent=1)
```
</details>

## Results

Calibrated c per site (h = c·mean|x|):

| draw | c |
|---|---|
| t4 | attn_S5 0.14, attn_F5 0.24, o_S4 0.18, o_F4 0.21, ffn_S4 0.18, ffn_F4 0.21, ffn_F4_64 0.21, ffn_R4 0.21, down_s32 0.26, down_F5 0.36, down_5 0.38 |
| t3 | attn_S5 0.16, attn_F5 0.32, o_S4 0.18, o_F4 0.21, ffn_S4 0.18, ffn_F4 0.21, ffn_F4_64 0.21, ffn_R4 0.21, down_s32 0.26, down_F5 0.38, down_5 0.4 |
| t3hot8 | attn_S5 0.16, attn_F5 0.32, o_S4 0.18, o_F4 0.21, ffn_S4 0.18, ffn_F4 0.21, ffn_F4_64 0.21, ffn_R4 0.21, down_s32 0.26, down_F5 0.36, down_5 0.36 |
| t3hot48 | attn_S5 0.24, attn_F5 0.4, o_S4 0.18, o_F4 0.21, ffn_S4 0.18, ffn_F4 0.21, ffn_F4_64 0.21, ffn_R4 0.21, down_s32 0.26, down_F5 0.36, down_5 0.38 |

Accuracy per draw: pooled delta cos (seed +2 / +3); worst token over both seeds; **tokens under 0.98 out of 256**:

| config | t4 | t3 | t3hot8 | t3hot48 |
|---|---|---|---|---|
| 14:55 all-sites split (q/k/v A5 S32, o + gate/up A4 S32, down A5 s32) | 0.9901/0.9900; worst 0.9876; **0/256** | 0.9898/0.9900; worst 0.9858; **0/256** | 0.9905/0.9904; worst 0.9878; **0/256** | 0.9910/0.9909; worst 0.9866; **0/256** |
| 14:55 recommended (same, down plain A5) | 0.9876/0.9875; worst 0.9831; **0/256** | 0.9873/0.9875; worst 0.9824; **0/256** | 0.9878/0.9881; worst 0.9813; **0/256** | 0.9884/0.9887; worst 0.9792; **1/256** |
| static 32-ch list at o + gate/up (cached f32 cols); q/k/v, down argpartition | 0.9863/0.9864; worst 0.9809; **0/256** | 0.9859/0.9860; worst 0.9490; **3/256** | 0.9868/0.9867; worst 0.9530; **3/256** | 0.9845/0.9840; worst 0.9274; **30/256** |
| static list o 32 + gate/up 64; q/k/v, down argpartition | 0.9863/0.9864; worst 0.9807; **0/256** | 0.9859/0.9860; worst 0.9490; **3/256** | 0.9867/0.9866; worst 0.9524; **3/256** | 0.9876/0.9877; worst 0.9479; **2/256** |
| static lists at every site (no per-token select) | 0.9686/0.9707; worst 0.7915; **233/256** | 0.9511/0.9527; worst 0.4070; **256/256** | 0.9612/0.9617; worst 0.4747; **227/256** | 0.9191/0.9162; worst 0.5283; **248/256** |
| gate/up reuse the q/k/v residual top-32 set | 0.9888/0.9888; worst 0.9854; **0/256** | 0.9882/0.9885; worst 0.9509; **2/256** | 0.9875/0.9868; worst 0.8966; **10/256** | 0.9778/0.9785; worst 0.8602; **98/256** |
| **all-sites split, columns gathered from transposed planes** | 0.9901/0.9900; worst 0.9876; **0/256** | 0.9898/0.9900; worst 0.9858; **0/256** | 0.9905/0.9904; worst 0.9878; **0/256** | 0.9910/0.9909; worst 0.9866; **0/256** |
| transposed planes + gate/up reuse | 0.9888/0.9888; worst 0.9854; **0/256** | 0.9882/0.9885; worst 0.9509; **2/256** | 0.9875/0.9868; worst 0.8966; **10/256** | 0.9778/0.9785; worst 0.8602; **98/256** |

Mean relative L2 error of the layer delta (t4 / t3 / t3hot8 / t3hot48):

| config | rel L2 |
|---|---|
| 14:55 all-sites split (q/k/v A5 S32, o + gate/up A4 S32, down A5 s32) | 0.141 / 0.142 / 0.138 / 0.135 |
| 14:55 recommended (same, down plain A5) | 0.157 / 0.158 / 0.155 / 0.151 |
| static 32-ch list at o + gate/up (cached f32 cols); q/k/v, down argpartition | 0.165 / 0.167 / 0.162 / 0.177 |
| static list o 32 + gate/up 64; q/k/v, down argpartition | 0.165 / 0.167 / 0.163 / 0.157 |
| static lists at every site (no per-token select) | 0.244 / 0.306 / 0.275 / 0.397 |
| gate/up reuse the q/k/v residual top-32 set | 0.149 / 0.152 / 0.160 / 0.208 |
| **all-sites split, columns gathered from transposed planes** | 0.141 / 0.142 / 0.138 / 0.135 |
| transposed planes + gate/up reuse | 0.149 / 0.152 / 0.160 / 0.208 |

Full layer step at B=1 in ms, median (min–max), for runs 1 / 2 / 3. Ratio = all-`tmac_g8` median / variant median, worst case (`tmac_g8` min / variant max) in parentheses:

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| 14:55 all-sites split (q/k/v A5 S32, o + gate/up A4 S32, down A5 s32) | 13.67 (13.64–13.71) / 13.67 (13.66–14.34) / 14.02 (14.01–14.53) | 1.51x (1.50x) / 1.55x (1.47x) / 1.51x (1.45x) |
| 14:55 recommended (same, down plain A5) | 13.34 (13.32–13.42) / 13.36 (13.34–14.34) / 13.69 (13.68–13.76) | 1.55x (1.54x) / 1.58x (1.47x) / 1.55x (1.53x) |
| static 32-ch list at o + gate/up (cached f32 cols); q/k/v, down argpartition | 12.09 (12.08–12.10) / 12.08 (12.07–12.22) / 12.34 (12.33–12.37) | 1.71x (1.70x) / 1.75x (1.73x) / 1.72x (1.71x) |
| static list o 32 + gate/up 64; q/k/v, down argpartition | 12.16 (12.14–12.41) / 12.15 (12.14–12.17) / 12.41 (12.40–12.75) | 1.70x (1.66x) / 1.74x (1.73x) / 1.71x (1.66x) |
| static lists at every site (no per-token select) | 11.53 (11.52–11.56) / 11.44 (11.43–11.61) / 11.70 (11.68–12.05) | 1.79x (1.78x) / 1.85x (1.82x) / 1.81x (1.75x) |
| gate/up reuse the q/k/v residual top-32 set | 13.48 (13.46–13.50) / 13.48 (13.47–13.56) / 13.81 (13.79–14.09) | 1.53x (1.53x) / 1.57x (1.55x) / 1.53x (1.50x) |
| **all-sites split, columns gathered from transposed planes** | 12.73 (12.72–13.11) / 12.74 (12.72–12.80) / 13.05 (13.02–13.91) | 1.62x (1.57x) / 1.66x (1.65x) / 1.62x (1.52x) |
| transposed planes + gate/up reuse | 13.17 (13.16–13.47) / 13.31 (13.14–13.56) / 13.48 (13.47–13.80) | 1.57x (1.53x) / 1.59x (1.55x) / 1.57x (1.53x) |
| all-`tmac_g8` | 20.65 (20.61–20.73) / 21.12 (21.09–21.18) / 21.19 (21.12–21.28) | 1.00x |
| `f32_resident` | 1.32 (1.27–3.07) / 1.23 (1.22–1.89) / 1.39 (1.30–3.29) | 15.7x / 17.2x / 15.2x |

Interleaved check: median of 7 per-round medians (min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 20.83 (20.82–20.99) | 1.00x |
| 14:55 all-sites split (q/k/v A5 S32, o + gate/up A4 S32, down A5 s32) | 13.69 (13.64–13.74) | 1.52x |
| 14:55 recommended (same, down plain A5) | 13.37 (13.31–13.40) | 1.56x |
| static 32-ch list at o + gate/up (cached f32 cols); q/k/v, down argpartition | 12.26 (12.20–12.27) | 1.70x |
| static list o 32 + gate/up 64; q/k/v, down argpartition | 12.29 (12.26–12.33) | 1.70x |
| **all-sites split, columns gathered from transposed planes** | 12.76 (12.72–12.78) | 1.63x |
| transposed planes + gate/up reuse | 13.30 (13.26–13.35) | 1.57x |

Resident weight bytes for the whole layer:

| layout | bytes | vs int8 |
|---|---|---|
| 2-bit mask/sign planes + row sums (14:55 path, = `tmac_g8` bytes) | 17.55 MB | 3.96x |
| + static f32 cols, o 32 + gate/up 32 | 19.65 MB | 3.54x |
| + static f32 cols, o 32 + gate/up 64 | 21.42 MB | 3.24x |
| + transposed mask/sign planes (all projections) | 34.92 MB | 1.99x |
| int8 weights | 69.47 MB | 1.00x |
| f32 weights | 277.87 MB | 0.25x |

## Verdict

**Inconclusive.** The speed target is met, but only by giving up half of the memory win.

1. **Transposed planes hit the target on speed and accuracy.**
   - Accuracy: identical to the 14:55 all-sites split, **0/1024** under 0.98 (worst 0.9858).
   - Speed: 1.62–1.66x in three runs and **1.63x interleaved**. The reference runs 1.51–1.55x (1.52x interleaved).
   - Cost: packed weights go from 17.55 MB to 34.92 MB. That is still 1.99x below int8, but `tmac_g8` does the same work in half the bytes. Decode on this path is meant to be memory-bound, and numpy here is compute-bound, so the 1.63x is not a clean efficiency win. Not called viable.
2. **Static lists are fast but not robust.**
   - o + gate/up static 32 runs 1.71–1.75x (1.70x interleaved) for +2.10 MB of cached columns. It stays clean on t4 (0/256), but loses 3/256 on t3 and t3hot8 and 30/256 on t3hot48.
   - Widening gate/up to 64 static channels brings t3hot48 down to 2/256, but t3 and t3hot8 still lose 3/256 each. Those are per-token tails that a fixed list cannot catch.
   - Static lists at every site fail outright: 227–256/256 under 0.98.
3. **Reuse does not help.** The q/k/v index set does not carry over to gate/up: 10/256 under 0.98 on t3hot8 and 98/256 on t3hot48. It also saves almost no time (1.53–1.57x), because the 2560-wide argpartition is not the cost. The strided column gather is.
4. **Where the time goes:** the split overhead is the strided `wm8[:, byt]` gather of out × 32 bytes per site, not the select. Making that gather contiguous saves ~0.9 ms per layer step.
5. **Caveat as before:** synthetic weights and tokens. `f32_resident` is still ~15–17x faster than `tmac_g8` at 16x the memory, so this matters only for the memory-bound packed path.

Recommendation: keep the 14:55 recommended default (q/k/v A5 + split 32, o and gate/up A4 + split 32, down A5) as the bytes-neutral path, at ~1.55x. Transposed planes are an opt-in mode that trades +17.37 MB for 0/1024 accuracy at ~1.63x. Do not ship static hot lists or index reuse.

## Pair files this wave

Quant Chief 18:06 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave. Roadmap blobs are unchanged: JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923. JuniorLLM head before this push was 958c584, the 14:55 note, so nobody else has pushed to JuniorLLM in the last 6 hours. This note touches only `junior_osai/vault/findings/`.

## Next question

Can the gather be made contiguous without doubling bytes? Two options:

- transpose only the gate/up planes (the widest gather, +~8.8 MB), keeping row-major-only for q/k/v, o and down;
- a hybrid select: a static 32-channel list with cached columns plus a per-token top-8 dynamic split on the residual.

Target: 0/1024 under 0.98 across t4/t3/t3hot8/t3hot48 at ≥ 1.6x interleaved, with resident bytes ≤ 1.25x the 2-bit planes.

close: sha256(prev || this)[:16] = 74be9db0bf711bcd
