---
local: true
date: 2026-11-15
wave: "2026-11-15 02:55 (ran ~02:47–03:05 MT; Chief last_run 2026-11-15T09:10:00Z ~02:10 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, a4, bitplane-activation, outlier-split, hybrid-select, static-hot-list, partial-transpose, bytes-vs-speed, tmac-g8, numpy, viable]
---

# A hybrid select at o and gate/up (static calibrated 32-channel list with cached exact columns, plus a per-token dynamic top-8 split) keeps 0/1024 tokens under 0.98 and runs 1.64x interleaved vs `tmac_g8` at 1.12x the 2-bit plane bytes (19.65 MB, 3.54x below int8). Transposing only gate/up planes is slower (1.60x) and costs 1.50x bytes

## Question

The prior note [[2026-11-14_1855_cheaper-all-sites-split-static-hot-list-reuse-transposed-planes-full-bitnetd-layer-b1-vs-tmac-g8]] reached 0/1024 under 0.98 at 1.63x interleaved only by keeping a transposed copy of all mask/sign planes (34.92 MB, 1.99x the 2-bit bytes). Its next question: can the split-column gather be made cheap **without doubling bytes**? Target: 0/1024 under 0.98 across t4/t3/t3hot8/t3hot48 at ≥ 1.6x interleaved vs `tmac_g8`, with resident bytes ≤ 1.25x the 2-bit planes. Options tested: (a) transpose only the gate/up planes (widest gather), optionally also down; (b) a hybrid select: a static calibrated 32-channel list with cached columns plus a per-token dynamic top-8 (or top-16) residual split. Quant Chief `last_tips.json` `last_run` 2026-11-15T09:10:00Z (~02:10 MT) wrote pair_A/pair_B **no gap**, so this note uses the lane's own next question.

## Method

- **Layer, weights, kernels, draws, metric:** unchanged from 14:55/18:55. `base.py` copied byte for byte (sha256[:16] `a38c8245cd13d800`); the 18:55 `bench.py` is imported as `prev18.py` (sha256[:16] `542d52ae6f1eb4c6`) for the split kernels, static lists and token draws. One full bitnetd decoder layer at B=1 (hidden 2560, GQA 20 q / 5 KV heads, FFN 6912), absmean ternary weights as 2-bit mask/sign planes. Accuracy is per-token cosine of the layer delta against the weight-only W1.58 layer, on 2 × 128 tokens per draw (1,024 tokens total).
- **Partial transpose (T_ffn, T_ffn_down):** same per-token argpartition split 32 at every site as 14:55; only gate/up (and down) gather their 32 exact ternary columns from transposed packed planes. Math is identical to the 14:55 reference, so accuracy matches exactly.
- **Hybrid (H):** per site, the 32 channels with the largest mean |x| on the 16-token calibration draw are zeroed and added back through exact ternary columns cached as f32 (gw-scaled, counted as resident bytes). On the remaining residual, a per-token argpartition picks the top-kd (8 or 16) channels, which are also zeroed and added back through exact columns gathered strided from the row-major planes. The rest is A4 (o, gate/up) or A5 (q/k/v, down) bitplane-quantised. c is calibrated per site per draw as before.
- **Timing:** B=1, 3 runs (9 repeats × 5 passes, 3 warm), plus a 7-round interleaved check with rotated order. Run 1 (with eval) took 3m06s. Setup: 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench0255_1115`, next to `base.py` and `prev18.py`):

```
python3 bench.py results_run1.json          # calibration + 4 draws x 2x128 eval + timing
python3 bench.py results_run2.json noeval   # timing only
python3 bench.py results_run3.json noeval
python3 tinter.py                           # interleaved timing
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Contiguous-gather split without doubling bytes (02:55 11/15). Same layer, weights, kernels, token draws
(t4, t3, t3hot8, t3hot48) and metric as 14:55/18:55; base.py and prev18.py (the 18:55 bench.py) copied verbatim.
New options:
  T_ffn       per-token argpartition split 32 everywhere, but only gate/up gather columns from transposed planes
  T_ffn_down  same, plus transposed down planes
  H           hybrid select: static calibrated 32-ch list (exact ternary cols cached f32) + per-token dynamic top-kd
              residual split (row-major strided gather of only kd columns)
numpy-only, offline. Usage: python3 bench.py results_runN.json [noeval]"""
import json, statistics, sys
import numpy as np
import base, prev18 as p
from base import wo_mm, tmac_g8, f32res, timed, cos, rowcos, rms, silu, quantk, a5gemv, PW, GRID5, D, KVH, QH, HD
SEED = base.SEED; GRID4 = p.GRID4; HOT = p.HOT; STATIC = p.STATIC; site_proj = p.site_proj
TSET = set()
def add_transposed(names):
    for n in names:
        P = PW[n]; m = np.unpackbits(P["wm8"], axis=1); s = np.unpackbits(P["ws8"], axis=1)
        P["wmT"] = np.packbits(m.T, axis=1); P["wsT"] = np.packbits(s.T, axis=1); TSET.add(n)
def hyb(x, c, key, kd, kb, names):
    S = STATIC[key]; si = S["idx"]; B, Fi = x.shape
    xr = x.copy(); xr[:, si] = 0.0
    I = np.argpartition(np.abs(xr), Fi - kd, axis=1)[:, Fi - kd:]
    xq = xr.copy()
    for b in range(B): xq[b, I[b]] = 0.0
    pl, hh = quantk(xq, c, kb); xs = x[:, si]; outs = []
    for n in names:
        P = PW[n]; y = a5gemv(pl, hh, P) + xs @ S[n]; g = np.float32(P["gw"])
        for b in range(B): ix = I[b]; y[b] += (x[b, ix] @ p.cols_rowmajor(P, ix)) * g
        outs.append(y)
    return outs
def smm(spec, x, names, ctx):
    if spec[0] == "H": return hyb(x, spec[1], spec[2], spec[3], spec[4], names)
    return p.smm(spec, x, names, ctx)
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
    for s in ("attn", "o", "ffn", "down"): p.make_static(f"{dn}_{s}_32", sites[s], 32, site_proj[s])
    def calib(s, mk, grid):
        sc = {c: statistics.mean(cos(smm(mk(c), sites[s], (q,), {})[0], ycw[q]) for q in site_proj[s]) for c in grid}
        c = max(sc, key=sc.get); return [mk(c), sc[c]]
    cal = {}
    cal["attn_S5"] = calib("attn", lambda c: ("S", c, 32, 5), GRID5 + [0.02, 0.03])
    cal["o_S4"] = calib("o", lambda c: ("S", c, 32, 4), GRID4)
    cal["ffn_S4"] = calib("ffn", lambda c: ("S", c, 32, 4), GRID4)
    cal["down_s32"] = calib("down", lambda c: ("s", c, 32), GRID5)
    for kd in (8, 16):
        cal[f"attn_H5_{kd}"] = calib("attn", lambda c: ("H", c, f"{dn}_attn_32", kd, 5), GRID5 + [0.02, 0.03])
        cal[f"o_H4_{kd}"] = calib("o", lambda c: ("H", c, f"{dn}_o_32", kd, 4), GRID4)
        cal[f"ffn_H4_{kd}"] = calib("ffn", lambda c: ("H", c, f"{dn}_ffn_32", kd, 4), GRID4)
        cal[f"down_H5_{kd}"] = calib("down", lambda c: ("H", c, f"{dn}_down_32", kd, 5), GRID5)
    z = sites["down"]
    sc = {c: cos(base.site_mm((5, c), z, ("down",))[0], z @ PW["down"]["Wt"].T) for c in GRID5}
    cal["down_5"] = [[5, max(sc, key=sc.get)], max(sc.values())]
    return cal
def variants(cal):
    sp = lambda k: tuple(cal[k][0]); X = lambda s: ("X",) + s[1:]
    ref = {"attn": sp("attn_S5"), "o": sp("o_S4"), "ffn": sp("ffn_S4"), "down": sp("down_s32")}
    return {"ref_allsplit_1455": ref,
            "rec_1455_downA5": {**ref, "down": sp("down_5")},
            "transposed_allsplit_1855": {"attn": X(sp("attn_S5")), "o": X(sp("o_S4")), "ffn": X(sp("ffn_S4")), "down": ("sX",) + sp("down_s32")[1:]},
            "T_ffn": {**ref, "ffn": X(sp("ffn_S4"))},
            "T_ffn_down": {**ref, "ffn": X(sp("ffn_S4")), "down": ("sX",) + sp("down_s32")[1:]},
            "hyb_o_ffn_8": {**ref, "o": sp("o_H4_8"), "ffn": sp("ffn_H4_8")},
            "hyb_all_8": {"attn": sp("attn_H5_8"), "o": sp("o_H4_8"), "ffn": sp("ffn_H4_8"), "down": sp("down_H5_8")},
            "hyb_all_16": {"attn": sp("attn_H5_16"), "o": sp("o_H4_16"), "ffn": sp("ffn_H4_16"), "down": sp("down_H5_16")}}
def setup():
    _, by = base.build(); add_transposed(PW.keys()); return by
def main():
    by = setup()
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "cal": {}, "variants": {}, "acc": {}, "time": {}}
    V = {}
    for dn, (df, nh) in HOT.items():
        cal = calibrate(dn, p.tokf(df, nh))
        res["cal"][dn] = {k: [list(v[0]), v[1]] for k, v in cal.items()}; V[dn] = variants(cal)
        print("cal", dn, res["cal"][dn], flush=True)
    res["bytes"]["static_cols_f32_t4"] = {k: sum(v.nbytes for kk, v in S.items() if kk != "idx") for k, S in STATIC.items() if k.startswith("t4_")}
    res["bytes"]["transposed"] = {n: int(PW[n]["wmT"].nbytes + PW[n]["wsT"].nbytes) for n in PW}
    print("bytes", res["bytes"], flush=True)
    if "noeval" not in sys.argv:
        for dn, (df, nh) in HOT.items():
            tok = p.tokf(df, nh)
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
    x1 = p.tokf(4, 0)(np.random.default_rng(SEED + 9), 1)
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, cfg in V["t4"].items(): fns["q_" + n] = (lambda cfg=cfg: layer(x1, None, cfg))
    for n, fn in fns.items(): res["time"][n] = timed(fn); print(n, res["time"][n], flush=True)
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
import base, bench, prev18 as p
bench.setup(); base.REPEATS = 5
V = bench.variants(bench.calibrate("t4", p.tokf(4, 0)))
x1 = p.tokf(4, 0)(np.random.default_rng(base.SEED + 9), 1)
fns = {"tmac_g8": lambda: bench.layer(x1, base.tmac_g8)}
for n in V: fns[n] = (lambda cfg=V[n]: bench.layer(x1, None, cfg))
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
| t4 | attn_S5 0.14, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, attn_H5_8 0.18, o_H4_8 0.21, ffn_H4_8 0.21, down_H5_8 0.3, attn_H5_16 0.16, o_H4_16 0.21, ffn_H4_16 0.21, down_H5_16 0.28, down_5 0.38 |
| t3 | attn_S5 0.16, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, attn_H5_8 0.2, o_H4_8 0.21, ffn_H4_8 0.21, down_H5_8 0.32, attn_H5_16 0.18, o_H4_16 0.21, ffn_H4_16 0.21, down_H5_16 0.28, down_5 0.4 |
| t3hot8 | attn_S5 0.16, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, attn_H5_8 0.2, o_H4_8 0.21, ffn_H4_8 0.21, down_H5_8 0.3, attn_H5_16 0.18, o_H4_16 0.21, ffn_H4_16 0.21, down_H5_16 0.28, down_5 0.36 |
| t3hot48 | attn_S5 0.24, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, attn_H5_8 0.28, o_H4_8 0.21, ffn_H4_8 0.21, down_H5_8 0.3, attn_H5_16 0.22, o_H4_16 0.18, ffn_H4_16 0.21, down_H5_16 0.28, down_5 0.38 |

Accuracy per draw: pooled delta cos (seed +2 / +3); worst token over both seeds; **tokens under 0.98 out of 256**:

| config | t4 | t3 | t3hot8 | t3hot48 |
|---|---|---|---|---|
| 14:55 all-sites split (q/k/v A5 S32, o + gate/up A4 S32, down A5 s32) | 0.9901/0.9900; worst 0.9876; **0/256** | 0.9898/0.9900; worst 0.9858; **0/256** | 0.9905/0.9904; worst 0.9878; **0/256** | 0.9910/0.9909; worst 0.9866; **0/256** |
| 14:55 recommended (same, down plain A5) | 0.9876/0.9875; worst 0.9831; **0/256** | 0.9873/0.9875; worst 0.9824; **0/256** | 0.9878/0.9881; worst 0.9813; **0/256** | 0.9884/0.9887; worst 0.9792; **1/256** |
| 18:55 all-sites split, all planes transposed | 0.9901/0.9900; worst 0.9876; **0/256** | 0.9898/0.9900; worst 0.9858; **0/256** | 0.9905/0.9904; worst 0.9878; **0/256** | 0.9910/0.9909; worst 0.9866; **0/256** |
| all-sites split, only gate/up planes transposed | 0.9901/0.9900; worst 0.9876; **0/256** | 0.9898/0.9900; worst 0.9858; **0/256** | 0.9905/0.9904; worst 0.9878; **0/256** | 0.9910/0.9909; worst 0.9866; **0/256** |
| all-sites split, gate/up + down planes transposed | 0.9901/0.9900; worst 0.9876; **0/256** | 0.9898/0.9900; worst 0.9858; **0/256** | 0.9905/0.9904; worst 0.9878; **0/256** | 0.9910/0.9909; worst 0.9866; **0/256** |
| **hybrid at o + gate/up (static 32 + dynamic top-8); q/k/v S32, down s32** | 0.9880/0.9881; worst 0.9847; **0/256** | 0.9879/0.9879; worst 0.9837; **0/256** | 0.9886/0.9884; worst 0.9854; **0/256** | 0.9892/0.9892; worst 0.9855; **0/256** |
| hybrid static 32 + dynamic top-8 at every site | 0.9854/0.9854; worst 0.9808; **0/256** | 0.9846/0.9843; worst 0.9722; **8/256** | 0.9861/0.9858; worst 0.9761; **1/256** | 0.9867/0.9867; worst 0.9691; **3/256** |
| hybrid static 32 + dynamic top-16 at every site | 0.9873/0.9873; worst 0.9840; **0/256** | 0.9870/0.9872; worst 0.9827; **0/256** | 0.9880/0.9880; worst 0.9847; **0/256** | 0.9899/0.9897; worst 0.9857; **0/256** |

Mean relative L2 error of the layer delta (t4 / t3 / t3hot8 / t3hot48):

| config | rel L2 |
|---|---|
| 14:55 all-sites split (q/k/v A5 S32, o + gate/up A4 S32, down A5 s32) | 0.141 / 0.142 / 0.138 / 0.135 |
| 14:55 recommended (same, down plain A5) | 0.157 / 0.158 / 0.155 / 0.151 |
| 18:55 all-sites split, all planes transposed | 0.141 / 0.142 / 0.138 / 0.135 |
| all-sites split, only gate/up planes transposed | 0.141 / 0.142 / 0.138 / 0.135 |
| all-sites split, gate/up + down planes transposed | 0.141 / 0.142 / 0.138 / 0.135 |
| **hybrid at o + gate/up (static 32 + dynamic top-8); q/k/v S32, down s32** | 0.154 / 0.155 / 0.151 / 0.147 |
| hybrid static 32 + dynamic top-8 at every site | 0.170 / 0.176 / 0.167 / 0.163 |
| hybrid static 32 + dynamic top-16 at every site | 0.159 / 0.160 / 0.155 / 0.142 |

Full layer step at B=1 in ms, median (min–max), for runs 1 / 2 / 3. Ratio = all-`tmac_g8` median / variant median, worst case (`tmac_g8` min / variant max) in parentheses:

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| 14:55 all-sites split (q/k/v A5 S32, o + gate/up A4 S32, down A5 s32) | 13.75 (13.73–14.94) / 13.81 (13.76–13.85) / 13.90 (13.86–14.32) | 1.49x (1.36x) / 1.51x (1.50x) / 1.53x (1.48x) |
| 14:55 recommended (same, down plain A5) | 13.46 (13.37–14.06) / 13.52 (13.50–13.66) / 13.63 (13.54–13.71) | 1.52x (1.44x) / 1.54x (1.52x) / 1.56x (1.54x) |
| 18:55 all-sites split, all planes transposed | 12.84 (12.80–12.96) / 12.95 (12.90–12.98) / 13.10 (13.05–13.43) | 1.59x (1.56x) / 1.61x (1.60x) / 1.62x (1.57x) |
| all-sites split, only gate/up planes transposed | 13.20 (13.11–13.43) / 13.24 (13.21–13.29) / 13.34 (13.31–13.68) | 1.55x (1.51x) / 1.57x (1.56x) / 1.59x (1.55x) |
| all-sites split, gate/up + down planes transposed | 13.04 (13.01–13.09) / 13.16 (13.06–14.10) / 13.29 (13.14–14.17) | 1.57x (1.55x) / 1.58x (1.47x) / 1.60x (1.49x) |
| **hybrid at o + gate/up (static 32 + dynamic top-8); q/k/v S32, down s32** | 12.75 (12.68–12.78) / 12.92 (12.88–13.04) / 12.98 (12.92–13.49) | 1.60x (1.59x) / 1.61x (1.59x) / 1.63x (1.57x) |
| hybrid static 32 + dynamic top-8 at every site | 12.58 (12.56–12.98) / 12.60 (12.56–12.66) / 12.67 (12.62–12.84) | 1.63x (1.56x) / 1.65x (1.64x) / 1.68x (1.65x) |
| hybrid static 32 + dynamic top-16 at every site | 13.08 (13.03–13.15) / 13.10 (13.07–13.16) / 13.16 (13.09–13.18) | 1.56x (1.54x) / 1.59x (1.58x) / 1.61x (1.60x) |
| all-`tmac_g8` | 20.46 (20.28–21.19) / 20.82 (20.77–21.09) / 21.22 (21.16–21.52) | 1.00x |
| `f32_resident` | 1.43 (1.38–2.25) / 1.38 (1.32–2.01) / 1.44 (1.41–1.57) | 14.3x / 15.1x / 14.7x |

Interleaved check: median of 7 per-round medians (min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 21.26 (21.11–21.49) | 1.00x |
| 14:55 all-sites split (q/k/v A5 S32, o + gate/up A4 S32, down A5 s32) | 14.23 (13.82–17.85) | 1.49x |
| 14:55 recommended (same, down plain A5) | 13.52 (13.47–14.01) | 1.57x |
| 18:55 all-sites split, all planes transposed | 12.99 (12.88–13.26) | 1.64x |
| all-sites split, only gate/up planes transposed | 13.29 (13.22–13.53) | 1.60x |
| all-sites split, gate/up + down planes transposed | 13.21 (13.06–13.46) | 1.61x |
| **hybrid at o + gate/up (static 32 + dynamic top-8); q/k/v S32, down s32** | 12.98 (12.87–13.25) | 1.64x |
| hybrid static 32 + dynamic top-8 at every site | 12.74 (12.49–13.06) | 1.67x |
| hybrid static 32 + dynamic top-16 at every site | 13.22 (12.94–13.41) | 1.61x |

Resident weight bytes for the whole layer:

| layout | bytes | vs 2-bit planes | vs int8 |
|---|---|---|---|
| 2-bit mask/sign planes + row sums (= `tmac_g8` bytes; 14:55 paths) | 17.55 MB | 1.00x | 3.96x |
| + static f32 cols at o + gate/up (hybrid o/ffn) | 19.65 MB | 1.12x | 3.54x |
| + static f32 cols at every site (hybrid all) | 20.47 MB | 1.17x | 3.39x |
| + transposed gate/up planes | 26.40 MB | 1.50x | 2.63x |
| + transposed gate/up + down planes | 30.82 MB | 1.76x | 2.25x |
| + transposed planes, all projections (18:55) | 34.92 MB | 1.99x | 1.99x |
| int8 weights | 69.47 MB | 3.96x | 1.00x |
| f32 weights | 277.87 MB | 15.83x | 0.25x |

## Verdict

**Viable** (vs packed `tmac_g8`, the same 2-bit weight path), with the synthetic-data caveat below.

1. **Hybrid at o + gate/up meets all three targets.**
   - Accuracy: **0/1024** tokens under 0.98 across t4/t3/t3hot8/t3hot48 (worst 0.9837), between the 14:55 recommended default and the full all-sites split.
   - Speed: 1.60–1.63x in three runs and **1.64x interleaved** (worst-case ratios 1.57–1.59x). The 14:55 all-sites split runs 1.49–1.53x here and the 14:55 recommended default 1.52–1.56x.
   - Bytes: +2.10 MB of cached columns, **19.65 MB = 1.12x the 2-bit planes**, 3.54x below int8. Same speed as the 18:55 all-transposed mode (1.59–1.62x, 1.64x interleaved) at 15.3 MB less.
2. **Hybrid top-16 at every site** is also 0/1024 (worst 0.9827) at 1.56–1.61x and 1.61x interleaved for 20.47 MB (1.17x). It is a valid fallback but slower than hybrid o/ffn.
3. **Hybrid top-8 at every site is the fastest** (1.63–1.68x, 1.67x interleaved) but loses 8/256 on t3, 1/256 on t3hot8 and 3/256 on t3hot48: q/k/v and down need more than 8 dynamic channels.
4. **Partial transpose is not worth it.** Transposing only gate/up runs 1.55–1.59x (1.60x interleaved) for 26.40 MB (1.50x the planes); adding down gives 1.57–1.60x (1.61x) at 1.76x. Both miss the ≤ 1.25x byte budget, and the hybrid is faster.
5. **Why it works:** the cost was the strided gather of out × 32 bytes per site. The static list moves most of the outlier mass to a contiguous cached f32 GEMV (32 × out), and the strided gather shrinks from 32 to 8 columns.
6. **Caveat as before:** synthetic weights and tokens; the static list is calibrated on 16 tokens from the same distribution as the eval. `f32_resident` is still ~14–15x faster than `tmac_g8` at ~16x the memory, so this matters only on the memory-bound packed path.

Recommendation: make **hybrid at o + gate/up (static 32 + dynamic top-8), q/k/v A5 + shared split 32, down A5 + split 32** the packed B=1 default (~1.6x vs `tmac_g8`, 0/1024, +2.1 MB). Drop the transposed-plane mode.

## Pair files this wave

Quant Chief 02:10 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave. Roadmap blob unchanged: JuniorLLM docs/BETA_TO_OS.md bac6923. JuniorLLM head before this push was 854c73f (cloudcover95, 02:54 MT, `rails/loopback/loopback_only_check.py`, no roadmap item), parent 84941b7, the 18:55 note. This note touches only `junior_osai/vault/findings/`.

## Next question

Is the hybrid robust when the static list is calibrated on a different distribution than the eval (e.g. calibrate on t4, evaluate on t3hot48 with hot channels the list has never seen), and can the cached columns be stored as int8 trits (×gw at runtime) to cut the +2.10 MB to +0.53 MB without losing the 1.6x?

close: sha256(prev || this)[:16] = 8af48381da8e7f11
