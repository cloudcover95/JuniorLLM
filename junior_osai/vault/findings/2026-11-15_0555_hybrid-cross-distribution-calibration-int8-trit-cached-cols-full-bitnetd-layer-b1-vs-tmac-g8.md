---
local: true
date: 2026-11-15
wave: "2026-11-15 05:55 (ran ~05:49–06:10 MT; Chief last_run 2026-11-15T12:10:00Z ~05:10 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, hybrid-select, static-hot-list, calibration-shift, robustness, int8-trit-cols, bytes-vs-speed, tmac-g8, numpy, viable]
---

# Storing the hybrid's cached columns as int8 trits (×gw at runtime) costs nothing: identical accuracy and the same speed, with cached bytes cut from 2.10 MB to 0.52 MB (whole layer 18.07 MB, 1.03x the 2-bit planes). Calibrated on a light-tailed draw (t4), the hybrid is not robust to hot channels (12/256 under 0.98). Calibrated on a heavy-tailed hot draw (t3hot48), hybrid top-16 holds 0/1280 across all five draws, including an unseen hot set, at 1.64x interleaved vs `tmac_g8`

## Question

The prior note [[2026-11-15_0255_hybrid-static-list-plus-dynamic-top8-split-partial-transpose-full-bitnetd-layer-b1-vs-tmac-g8]] made the hybrid select the packed B=1 default: at o and gate/up, a static calibrated 32-channel list with cached exact columns plus a per-token dynamic top-8 split. It reached 0/1024 under 0.98 at 1.64x interleaved vs `tmac_g8` for 19.65 MB (1.12x the 2-bit planes). Its next question had two parts:

1. **Robustness.** Is the hybrid robust when the static list and clip c come from a different distribution than the eval?
2. **Bytes.** Can the cached f32 columns be stored as int8 trits (×gw at runtime) to cut +2.10 MB to +0.52 MB without losing the 1.6x?

Quant Chief `last_tips.json` `last_run` 2026-11-15T12:10:00Z (~05:10 MT) set pair_A and pair_B to **no gap**, so this note uses the lane's own next question.

## Method

- **Unchanged from 14:55/18:55/02:55:** layer, weights, kernels, metric and token generator.
  - `base.py` is copied byte for byte (sha256[:16] `a38c8245cd13d800`).
  - The 18:55 `bench.py` is imported as `prev18.py` (`542d52ae6f1eb4c6`) and the 02:55 `bench.py` as `prev0255.py` (`ab695e26a598fc37`). This note uses the 02:55 `hyb` and `layer`.
  - The layer is one full bitnetd decoder layer at B=1: hidden 2560, GQA with 20 q / 5 KV heads, FFN 6912. Weights are absmean ternary, stored as 2-bit mask/sign planes.
  - Accuracy is the per-token cosine of the layer delta against the weight-only W1.58 layer, on 2 × 128 tokens per draw.
- **Draws (5 × 256 = 1,280 tokens).** t4, t3, t3hot8 and t3hot48, with the hot set from rng 777 at ×15, as before. New: **t3hot48b**, which uses 48 hot channels drawn from rng 778, so its hot set is unseen.
- **Calibration sources.** The 32-channel list (top mean |x| over 16 calibration tokens) and every site's clip c come from one of:
  - **matched:** the same distribution as the eval draw.
  - **t4:** light tail, no hot channels.
  - **t3hot48:** heavy tail, but the wrong hot set for t3hot48b.
- **Static list overlap** with the t4 list (o + gate/up channels in common out of 32 + 32): t4 32+32 / t3 0+1 / t3hot8 0+0 / t3hot48 0+1 / t3hot48b 0+0. Overlap between the t3hot48b and t3hot48 lists: 0+0. The cross-calibrated lists are therefore essentially disjoint from the matched ones.
- **int8-trit columns (`Hi`).** The cached columns are stored as `int8` trits `cols_rowmajor(P, idx)` with shape 32 × out, and the hybrid computes `(xs @ T8) * gw`. The f32 path caches `cols * gw` instead. Everything else is identical to the 02:55 `hyb`.
- **Every variant keeps the rest of the layer:** q/k/v A5 with a shared per-token split 32, o and gate/up A4, and down A5 with split 32.
- **Timing.** B=1, 3 runs (9 repeats × 5 passes, 3 warm), plus a 7-round interleaved check in rotated order. Timed configs use the t4 calibration; cost does not depend on which list is cached. Run 1 (with eval) took 3m42s. Setup: 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench0555_1115`, next to `base.py`, `prev18.py`, `prev0255.py`):

```
python3 bench.py results_run1.json          # calibration on 5 draws + 5 draws x 2x128 eval + timing
python3 bench.py results_run2.json noeval   # timing only
python3 bench.py results_run3.json noeval
python3 tinter.py                           # interleaved timing
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Hybrid robustness + int8-trit cached columns (05:55 11/15). Same layer, weights, kernels and metric as 14:55/18:55/02:55;
base.py, prev18.py (18:55 bench.py) and prev0255.py (02:55 bench.py) copied verbatim.
Question: is the 02:55 hybrid (static 32-ch list with cached exact cols + per-token dynamic top-kd split at o and gate/up)
robust when the static list AND clip c are calibrated on a different distribution than the eval, and can the cached
columns be stored as int8 trits (x gw at runtime) instead of gw-scaled f32 (4x fewer cached bytes) without losing speed?
Eval draws: t4, t3, t3hot8, t3hot48 (hot set rng 777, x15) and t3hot48b (48 hot channels from rng 778: unseen hot set).
Calibration sources: matched (same draw), t4 (no hot channels), t3hot48 (wrong hot set for t3hot48b).
numpy-only, offline. Usage: python3 bench.py results_runN.json [noeval]"""
import json, statistics, sys
import numpy as np
import base, prev18 as p, prev0255 as h
from base import wo_mm, tmac_g8, f32res, timed, cos, rowcos, quantk, a5gemv, PW, GRID5, D
SEED = base.SEED; GRID4 = p.GRID4; STATIC = p.STATIC; site_proj = p.site_proj
def tokf(df, nh, hseed=777):
    hot = np.random.default_rng(hseed).choice(D, nh, replace=False) if nh else None
    def f(r, n):
        x = r.standard_t(df, (n, D)) * 0.05
        if nh: x[:, hot] *= 15.0
        return x.astype(np.float32)
    return f
DRAWS = {"t4": (4, 0, 777), "t3": (3, 0, 777), "t3hot8": (3, 8, 777), "t3hot48": (3, 48, 777), "t3hot48b": (3, 48, 778)}
I8 = {}
def make_i8(key):
    S = STATIC[key]; I8[key] = {n: np.ascontiguousarray(p.cols_rowmajor(PW[n], S["idx"]).astype(np.int8)) for n in S if n != "idx"}
def hyb_i8(x, c, key, kd, kb, names):
    S = STATIC[key]; T8 = I8[key]; si = S["idx"]; B, Fi = x.shape
    xr = x.copy(); xr[:, si] = 0.0
    I = np.argpartition(np.abs(xr), Fi - kd, axis=1)[:, Fi - kd:]
    xq = xr.copy()
    for b in range(B): xq[b, I[b]] = 0.0
    pl, hh = quantk(xq, c, kb); xs = x[:, si]; outs = []
    for n in names:
        P = PW[n]; g = np.float32(P["gw"])
        y = a5gemv(pl, hh, P) + (xs @ T8[n]) * g
        for b in range(B): ix = I[b]; y[b] += (x[b, ix] @ p.cols_rowmajor(P, ix)) * g
        outs.append(y)
    return outs
_smm = h.smm
def smm(spec, x, names, ctx):
    if spec[0] == "Hi": return hyb_i8(x, spec[1], spec[2], spec[3], spec[4], names)
    return _smm(spec, x, names, ctx)
h.smm = smm
layer = h.layer
def calibrate(dn, tok):
    xc = tok(np.random.default_rng(SEED + 1), 16)
    _, ycw, sites = layer(xc, wo_mm)
    for s in ("o", "ffn"):
        p.make_static(f"{dn}_{s}_32", sites[s], 32, site_proj[s]); make_i8(f"{dn}_{s}_32")
    def calib(s, mk, grid):
        sc = {c: statistics.mean(cos(smm(mk(c), sites[s], (q,), {})[0], ycw[q]) for q in site_proj[s]) for c in grid}
        c = max(sc, key=sc.get); return [mk(c), sc[c]]
    cal = {"attn_S5": calib("attn", lambda c: ("S", c, 32, 5), GRID5 + [0.02, 0.03]),
           "o_S4": calib("o", lambda c: ("S", c, 32, 4), GRID4),
           "ffn_S4": calib("ffn", lambda c: ("S", c, 32, 4), GRID4),
           "down_s32": calib("down", lambda c: ("s", c, 32), GRID5)}
    for kd in (8, 16):
        cal[f"o_H4_{kd}"] = calib("o", lambda c: ("H", c, f"{dn}_o_32", kd, 4), GRID4)
        cal[f"ffn_H4_{kd}"] = calib("ffn", lambda c: ("H", c, f"{dn}_ffn_32", kd, 4), GRID4)
    return cal
def hcfg(cal, kd, i8):
    sp = lambda k: tuple(cal[k][0]); f = (lambda s: ("Hi",) + s[1:]) if i8 else (lambda s: s)
    return {"attn": sp("attn_S5"), "o": f(sp(f"o_H4_{kd}")), "ffn": f(sp(f"ffn_H4_{kd}")), "down": sp("down_s32")}
def refcfg(cal):
    sp = lambda k: tuple(cal[k][0]); return {"attn": sp("attn_S5"), "o": sp("o_S4"), "ffn": sp("ffn_S4"), "down": sp("down_s32")}
def variants(CAL, dn):
    m, t4, hz = CAL[dn], CAL["t4"], CAL["t3hot48"]
    return {"ref_allsplit_matched": refcfg(m),
            "H8_f32_matched": hcfg(m, 8, False), "H8_i8_matched": hcfg(m, 8, True),
            "H8_f32_cal_t4": hcfg(t4, 8, False), "H8_i8_cal_t4": hcfg(t4, 8, True),
            "H16_f32_cal_t4": hcfg(t4, 16, False), "H16_i8_cal_t4": hcfg(t4, 16, True),
            "H8_i8_cal_hot48": hcfg(hz, 8, True), "H16_i8_cal_hot48": hcfg(hz, 16, True)}
TIMED = {"ref_allsplit": lambda C: refcfg(C), "H8_f32": lambda C: hcfg(C, 8, False), "H8_i8": lambda C: hcfg(C, 8, True),
         "H16_f32": lambda C: hcfg(C, 16, False), "H16_i8": lambda C: hcfg(C, 16, True)}
def overlap(a, b): return len(set(STATIC[a]["idx"].tolist()) & set(STATIC[b]["idx"].tolist()))
def main():
    _, by = base.build()
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "cal": {}, "acc": {}, "time": {}}
    CAL = {}
    for dn, a in DRAWS.items():
        CAL[dn] = calibrate(dn, tokf(*a)); res["cal"][dn] = {k: [list(v[0]), v[1]] for k, v in CAL[dn].items()}
        print("cal", dn, res["cal"][dn], flush=True)
    res["static_overlap_vs_t4"] = {dn: {s: overlap(f"{dn}_{s}_32", f"t4_{s}_32") for s in ("o", "ffn")} for dn in DRAWS}
    res["static_overlap_hot48b_vs_hot48"] = {s: overlap(f"t3hot48b_{s}_32", f"t3hot48_{s}_32") for s in ("o", "ffn")}
    res["bytes"]["static_cols_f32_t4"] = {k: sum(v.nbytes for kk, v in STATIC[k].items() if kk != "idx") for k in ("t4_o_32", "t4_ffn_32")}
    res["bytes"]["static_cols_i8_t4"] = {k: sum(v.nbytes for v in I8[k].values()) for k in ("t4_o_32", "t4_ffn_32")}
    print("bytes", res["bytes"], res["static_overlap_vs_t4"], res["static_overlap_hot48b_vs_hot48"], flush=True)
    if "noeval" not in sys.argv:
        for dn, a in DRAWS.items():
            tok = tokf(*a); V = variants(CAL, dn)
            for es in (2, 3):
                xe = tok(np.random.default_rng(SEED + es), 128)
                yw = np.concatenate([layer(xe[i:i+32], wo_mm)[0] for i in range(0, 128, 32)])
                for n, cfg in V.items():
                    ya = np.concatenate([layer(xe[i:i+32], None, cfg)[0] for i in range(0, 128, 32)])
                    d1, d0 = ya - xe, yw - xe; rc = rowcos(d1, d0)
                    r = {"delta_pooled": cos(d1, d0), "delta_min": min(rc), "rows_below_098": int(sum(v < 0.98 for v in rc)), "n": 128,
                         "rel_l2": float(np.linalg.norm((d1 - d0).astype(np.float64)) / np.linalg.norm(d0.astype(np.float64)))}
                    res["acc"].setdefault(dn, {}).setdefault(n, {})[f"seed{es}"] = r
                    print(dn, n, es, r, flush=True)
    x1 = tokf(4, 0)(np.random.default_rng(SEED + 9), 1)
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, mk in TIMED.items(): fns["q_" + n] = (lambda cfg=mk(CAL["t4"]): layer(x1, None, cfg))
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
import base, bench
base.build(); base.REPEATS = 5
C = bench.calibrate("t4", bench.tokf(4, 0))
x1 = bench.tokf(4, 0)(np.random.default_rng(base.SEED + 9), 1)
fns = {"tmac_g8": lambda: bench.layer(x1, base.tmac_g8)}
for n, mk in bench.TIMED.items(): fns[n] = (lambda cfg=mk(C): bench.layer(x1, None, cfg))
names = list(fns); rounds = {n: [] for n in names}
for r in range(7):
    for n in names[r % len(names):] + names[:r % len(names)]: rounds[n].append(base.timed(fns[n])["median_ms"])
out = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
for n in names: out[n]["vs_tmac"] = out["tmac_g8"]["median_ms"] / out[n]["median_ms"]
print(json.dumps(out, indent=1)); json.dump(out, open("tinter.json", "w"), indent=1)
```

</details>

## Results

Calibrated c per draw (each spec's clip value):

| draw | c |
|---|---|
| t4 | attn_S5 0.14, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, o_H4_8 0.21, ffn_H4_8 0.21, o_H4_16 0.21, ffn_H4_16 0.21 |
| t3 | attn_S5 0.16, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, o_H4_8 0.21, ffn_H4_8 0.21, o_H4_16 0.21, ffn_H4_16 0.21 |
| t3hot8 | attn_S5 0.16, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, o_H4_8 0.21, ffn_H4_8 0.21, o_H4_16 0.21, ffn_H4_16 0.21 |
| t3hot48 | attn_S5 0.24, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, o_H4_8 0.21, ffn_H4_8 0.21, o_H4_16 0.18, ffn_H4_16 0.21 |
| t3hot48b | attn_S5 0.24, o_S4 0.18, ffn_S4 0.18, down_s32 0.26, o_H4_8 0.21, ffn_H4_8 0.21, o_H4_16 0.18, ffn_H4_16 0.21 |

The only differences between t4 and t3hot48 calibration are the q/k/v clip (attn_S5 0.14 vs 0.24) and o_H4_16 (0.21 vs 0.18).

Accuracy per eval draw: pooled delta cos (seed +2 / +3), worst token over both seeds, and **tokens under 0.98 out of 256**:

| config | t4 | t3 | t3hot8 | t3hot48 | t3hot48b (unseen hot set) | total under 0.98 |
|---|---|---|---|---|---|---|
| 14:55 all-sites split, matched cal | 0.9901/0.9900; worst 0.9876; **0/256** | 0.9898/0.9900; worst 0.9858; **0/256** | 0.9905/0.9904; worst 0.9878; **0/256** | 0.9910/0.9909; worst 0.9866; **0/256** | 0.9910/0.9910; worst 0.9871; **0/256** | 0/1280 |
| 02:55 hybrid top-8, f32 cols, matched cal (02:55 default) | 0.9880/0.9881; worst 0.9847; **0/256** | 0.9879/0.9879; worst 0.9837; **0/256** | 0.9886/0.9884; worst 0.9854; **0/256** | 0.9892/0.9892; worst 0.9855; **0/256** | 0.9893/0.9893; worst 0.9836; **0/256** | 0/1280 |
| hybrid top-8, int8-trit cols, matched cal | 0.9880/0.9881; worst 0.9847; **0/256** | 0.9879/0.9879; worst 0.9837; **0/256** | 0.9886/0.9884; worst 0.9854; **0/256** | 0.9892/0.9892; worst 0.9855; **0/256** | 0.9893/0.9893; worst 0.9836; **0/256** | 0/1280 |
| hybrid top-8, f32 cols, list + c from t4 | 0.9880/0.9881; worst 0.9847; **0/256** | 0.9877/0.9878; worst 0.9835; **0/256** | 0.9879/0.9879; worst 0.9839; **0/256** | 0.9852/0.9847; worst 0.9740; **12/256** | 0.9852/0.9852; worst 0.9750; **12/256** | 24/1280 |
| hybrid top-8, int8-trit cols, list + c from t4 | 0.9880/0.9881; worst 0.9847; **0/256** | 0.9877/0.9878; worst 0.9835; **0/256** | 0.9879/0.9879; worst 0.9839; **0/256** | 0.9852/0.9847; worst 0.9740; **12/256** | 0.9852/0.9852; worst 0.9750; **12/256** | 24/1280 |
| hybrid top-16, f32 cols, list + c from t4 | 0.9886/0.9885; worst 0.9863; **0/256** | 0.9882/0.9883; worst 0.9848; **0/256** | 0.9884/0.9884; worst 0.9847; **0/256** | 0.9858/0.9855; worst 0.9749; **7/256** | 0.9860/0.9861; worst 0.9766; **9/256** | 16/1280 |
| hybrid top-16, int8-trit cols, list + c from t4 | 0.9886/0.9885; worst 0.9863; **0/256** | 0.9882/0.9883; worst 0.9848; **0/256** | 0.9884/0.9884; worst 0.9847; **0/256** | 0.9858/0.9855; worst 0.9749; **7/256** | 0.9860/0.9861; worst 0.9766; **9/256** | 16/1280 |
| hybrid top-8, int8-trit cols, list + c from t3hot48 | 0.9847/0.9845; worst 0.9796; **2/256** | 0.9855/0.9858; worst 0.9808; **0/256** | 0.9869/0.9869; worst 0.9835; **0/256** | 0.9892/0.9892; worst 0.9855; **0/256** | 0.9888/0.9888; worst 0.9842; **0/256** | 2/1280 |
| **hybrid top-16, int8-trit cols, list + c from t3hot48** | 0.9852/0.9850; worst 0.9809; **0/256** | 0.9862/0.9864; worst 0.9801; **0/256** | 0.9877/0.9876; worst 0.9832; **0/256** | 0.9901/0.9899; worst 0.9846; **0/256** | 0.9898/0.9900; worst 0.9860; **0/256** | 0/1280 |

Mean relative L2 error of the layer delta (t4 / t3 / t3hot8 / t3hot48 / t3hot48b):

| config | rel L2 |
|---|---|
| 14:55 all-sites split, matched cal | 0.141 / 0.142 / 0.138 / 0.135 / 0.134 |
| 02:55 hybrid top-8, f32 cols, matched cal (02:55 default) | 0.154 / 0.155 / 0.151 / 0.147 / 0.146 |
| hybrid top-8, int8-trit cols, matched cal | 0.154 / 0.155 / 0.151 / 0.147 / 0.146 |
| hybrid top-8, f32 cols, list + c from t4 | 0.154 / 0.156 / 0.155 / 0.173 / 0.171 |
| hybrid top-8, int8-trit cols, list + c from t4 | 0.154 / 0.156 / 0.155 / 0.173 / 0.171 |
| hybrid top-16, f32 cols, list + c from t4 | 0.151 / 0.153 / 0.152 / 0.169 / 0.167 |
| hybrid top-16, int8-trit cols, list + c from t4 | 0.151 / 0.153 / 0.152 / 0.169 / 0.167 |
| hybrid top-8, int8-trit cols, list + c from t3hot48 | 0.176 / 0.170 / 0.162 / 0.147 / 0.150 |
| **hybrid top-16, int8-trit cols, list + c from t3hot48** | 0.172 / 0.165 / 0.157 / 0.141 / 0.142 |

Full layer step at B=1 in ms, median (min–max), for runs 1 / 2 / 3. Ratio = all-`tmac_g8` median / variant median, with the worst case (`tmac_g8` min / variant max) in parentheses:

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| 14:55 all-sites split | 14.13 (14.03–14.41) / 14.07 (13.92–16.80) / 13.96 (13.80–14.17) | 1.50x (1.47x) / 1.53x (1.26x) / 1.56x (1.47x) |
| hybrid top-8, f32 cols | 13.28 (13.20–13.61) / 12.95 (12.87–14.84) / 12.88 (12.83–12.95) | 1.60x (1.55x) / 1.66x (1.43x) / 1.70x (1.61x) |
| hybrid top-8, int8-trit cols | 13.29 (13.26–13.31) / 13.16 (12.88–16.71) / 12.96 (12.88–14.48) | 1.60x (1.59x) / 1.63x (1.27x) / 1.68x (1.44x) |
| hybrid top-16, f32 cols | 13.55 (13.49–13.92) / 13.30 (13.20–13.64) / 13.18 (13.11–13.21) | 1.57x (1.52x) / 1.62x (1.56x) / 1.66x (1.58x) |
| hybrid top-16, int8-trit cols | 13.57 (13.50–15.38) / 13.39 (13.23–13.49) / 13.21 (13.18–13.32) | 1.56x (1.37x) / 1.61x (1.57x) / 1.65x (1.57x) |
| all-`tmac_g8` | 21.21 (21.13–22.20) / 21.50 (21.25–21.68) / 21.84 (20.89–24.91) | 1.00x |
| `f32_resident` | 3.01 (1.68–4.77) / 1.60 (1.29–3.02) / 1.48 (1.40–2.24) | 7.1x / 13.5x / 14.8x |

Interleaved check: median of 7 per-round medians (min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 21.74 (21.29–22.60) | 1.00x |
| 14:55 all-sites split | 13.75 (13.62–13.86) | 1.58x |
| hybrid top-8, f32 cols | 12.93 (12.83–13.03) | 1.68x |
| hybrid top-8, int8-trit cols | 12.94 (12.87–13.11) | 1.68x |
| hybrid top-16, f32 cols | 13.14 (13.08–13.37) | 1.65x |
| hybrid top-16, int8-trit cols | 13.27 (13.15–13.45) | 1.64x |

Resident weight bytes for the whole layer:

| layout | bytes | vs 2-bit planes | vs int8 |
|---|---|---|---|
| 2-bit mask/sign planes + row sums (= `tmac_g8` bytes) | 17.55 MB | 1.00x | 3.96x |
| + int8-trit static cols at o + gate/up (this note) | 18.07 MB | 1.03x | 3.84x |
| + f32 static cols at o + gate/up (02:55) | 19.65 MB | 1.12x | 3.54x |
| int8 weights | 69.47 MB | 3.96x | 1.00x |
| f32 weights | 277.87 MB | 15.83x | 0.25x |

## Verdict

**Viable** (vs packed `tmac_g8`, the same 2-bit weight path), with the synthetic-data caveat below.

1. **int8-trit cached columns are free.**
   - Accuracy matches the f32 columns to about 1e-10 on every draw. The trits are exact, and gw is applied once after the GEMV.
   - Speed: top-8 runs 1.60–1.68x (f32 1.60–1.70x) and 1.68x interleaved (f32 1.68x). Top-16 runs 1.56–1.65x (f32 1.57–1.66x) and 1.64x interleaved (f32 1.65x).
   - Bytes: the cached columns drop from 2.10 MB to 0.52 MB. The whole layer is 18.07 MB = **1.03x the 2-bit planes and 3.84x below int8**, down from 1.12x.
2. **Calibrating on a light tail is not robust.** With the list and c from t4, hybrid top-8 loses **12/256** tokens on t3hot48 and **12/256** on t3hot48b (worst 0.9740). Top-16 still loses 7/256 and 9/256. On t3 and t3hot8 it stays at 0/256.
3. **Calibrating on a heavy tail transfers, even to an unseen hot set.**
   - Top-8 with t3hot48 calibration: 0/256 on t3hot48b, whose static list shares 0 channels with the calibration list. It loses 2/256 on the light t4 draw (worst 0.9796).
   - **Top-16 with t3hot48 calibration: 0/1280 on all five draws** (worst 0.9801).
   - The two calibrations differ mainly in the q/k/v clip (0.14 vs 0.24), and the static lists are disjoint. This points at the q/k/v clip c as the sensitive piece, more than the static list. That attribution was not separately ablated.
4. **Reference:** the 14:55 all-sites split is 0/1280 under any calibration tested, but runs only 1.50–1.56x (1.58x interleaved).

Recommendation: make the packed B=1 default **hybrid at o + gate/up with int8-trit cached columns and dynamic top-16, calibrated on a heavy-tailed hot draw**. It holds 0/1280, runs 1.56–1.65x (1.64x interleaved) and takes 18.07 MB (1.03x the planes). Top-8 is about 0.04x faster but costs 2/1280. Do not calibrate c on light-tailed activations.

**Caveat:** weights and tokens are synthetic, and hot channels are modelled as fixed ×15 scales. `f32_resident` is still about 7–15x faster than `tmac_g8` at about 16x the memory, so this matters only on the memory-bound packed path.

## Pair files this wave

At Quant Chief 05:10 MT, pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave. The roadmap blob is unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923). JuniorLLM head before this push was a5805be, the 02:55 note. This note touches only `junior_osai/vault/findings/`.

## Next question

Ablate the q/k/v clip: keep the t4 static lists but use the heavy-tail attn c. Does that alone recover 0/256 on t3hot48? Can a per-token adaptive q/k/v clip (c scaled by the token's own top-k |x| mass) remove the calibration-distribution dependence at no measurable cost vs the fixed-c top-16 hybrid?

close: sha256(prev || this)[:16] = 40eef8ed37d79d66
