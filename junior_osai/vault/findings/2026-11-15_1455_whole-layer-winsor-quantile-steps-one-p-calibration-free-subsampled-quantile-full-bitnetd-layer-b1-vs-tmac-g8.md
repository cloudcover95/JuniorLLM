---
local: true
date: 2026-11-15
wave: "2026-11-15 14:55 (ran ~14:44–15:10 MT; Chief last_run 2026-11-15T21:05:00Z ~14:05 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, hybrid-select, winsor, per-token-clip, quantile-step, calibration-free, one-p, subsample, robustness, tmac-g8, numpy, viable]
---

# One Winsor-quantile p = 0.993 for the whole layer (q/k/v, o, gate/up, down), calibrated on light-tailed t4 only, holds **0/1280** on all five draws at 1.60–1.65x (1.66x interleaved) vs `tmac_g8` and 18.07 MB. No per-site c is left. Subsampling the quantile is not worth it.

## Question

The prior note [[2026-11-15_1055_qkv-clip-ablation-per-token-winsor-quantile-clip-calibration-free-hybrid-full-bitnetd-layer-b1-vs-tmac-g8]] showed that a per-token Winsor-quantile step at q/k/v (`h = quantile_p(|x_res|)/31`) removes the calibration-distribution dependence there, and made hybrid top-8 the packed B=1 default (1.64x interleaved vs `tmac_g8`, 0/1280). The o, gate/up and down sites still used a fixed absmean clip c per site. Its next question:

1. **Whole layer.** Does the quantile step also make the o / gate/up (A4 hybrid) and down (A5 split 32) clips calibration-free, so the layer ships with **one p** and no per-site c?
2. **Cheaper quantile.** Can the per-token `np.partition` be replaced with a cheap estimate (here a strided subsample, every 4th or 8th element) without losing 0/1280?

Quant Chief `last_tips.json` `last_run` 2026-11-15T21:05:00Z (~14:05 MT) set pair_A and pair_B to **no gap**, so this note uses the lane's own next question.

## Method

- **Unchanged from 14:55/18:55/02:55/05:55/10:55:** layer, weights, kernels, metric, draws.
  - `base.py` byte for byte (sha256[:16] `a38c8245cd13d800`); `prev18.py` `542d52ae6f1eb4c6`, `prev0255.py` `ab695e26a598fc37`, `prev0555.py` `546dddb014321b86`, and the 10:55 `bench.py` as `prev1055.py` `d1a0f1d644195409`.
  - One full bitnetd decoder layer at B=1 (hidden 2560, GQA 20 q / 5 KV heads, FFN 6912), absmean ternary weights in 2-bit mask/sign planes. Accuracy = per-token cosine of the layer delta vs the weight-only W1.58 layer, 2 × 128 tokens per draw, draws t4, t3, t3hot8, t3hot48 (hot set rng 777, ×15) and t3hot48b (unseen hot set, rng 778).
- **Every config is calibrated on t4 only** (16 tokens, light-tailed, no hot channels), including the static 32-channel lists for the o and gate/up hybrids.
- **Quantile step at all four sites:** after each site's split (q/k/v and down: per-token top-32; o and gate/up: static 32 int8-trit cols + dynamic top-8), `h = quantile_p(|x_res|)/Lk` with Lk = 31 (A5) or 15 (A4).
  - Per-site p comes from a grid 0.95…1.0.
  - "One p" picks the single p that maximizes the mean per-site calibration cosine.
- **Subsample:** the same quantile from `|x_res[:, ::4]|` or `|x_res[:, ::8]|`.
- **Timing.** B=1, 3 runs (9 repeats × 5 passes, 3 warm) plus a 7-round interleaved check (10:55 `tinter.py`, unchanged). There is also a micro-timing of the step estimate alone (200 calls × 15 repeats). Run 1 (with eval) took 3m12s. 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.
- **Harmless overhead:** `main()` also runs the 10:55 calibration on the four non-t4 draws, but none of their outputs are used by any config here.

Commands (in `/workspace/bench1455_1115`, next to `base.py`, `prev18.py`, `prev0255.py`, `prev0555.py`, `prev1055.py`):

```
python3 bench.py results_run1.json          # t4 calibration + 5 draws x 2x128 eval + step micro-timing + layer timing
python3 bench.py results_run2.json noeval   # timing only
python3 bench.py results_run3.json noeval
python3 tinter.py                           # interleaved timing (10:55 tinter.py, unchanged)
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Whole-layer Winsor-quantile steps + cheap quantile estimate (14:55 11/15). Same layer, weights, kernels, draws and
metric as 14:55/18:55/02:55/05:55/10:55; base.py, prev18.py, prev0255.py, prev0555.py and prev1055.py (10:55 bench.py)
copied verbatim. Base config = 10:55 default: q/k/v A5 + per-token top-32 split with Winsor-quantile step, hybrid top-8
int8-trit at o + gate/up A4 with fixed absmean c, down A5 split 32 with fixed absmean c.
Question: (A) does the per-token Winsor-quantile step h = quantile_p(|x_res|)/Lk also make the o / gate/up / down clips
calibration-free (t4-only calibration, 0/1280 on 5 draws), ideally with ONE p for the whole layer? (B) can the
per-token np.partition be replaced by a strided-subsample quantile (every 4th / 8th element) without losing 0/1280 or speed?
numpy-only, offline. Usage: python3 bench.py results_runN.json [noeval]"""
import json, statistics, sys, time
import numpy as np
import base, prev18 as p, prev0255 as h, prev0555 as f, prev1055 as w
from base import wo_mm, tmac_g8, f32res, timed, cos, rowcos, a5gemv, u64, PW, D
SEED = base.SEED; site_proj = p.site_proj; DRAWS = f.DRAWS; tokf = f.tokf; layer = f.layer; STATIC = p.STATIC; I8 = f.I8
def qhat(x, pq, Lk, sub):
    xs = np.abs(x[:, ::sub]) if sub > 1 else np.abs(x)
    m = xs.shape[1]; j = min(m - 1, int(round(pq * (m - 1))))
    return (np.partition(xs, j, axis=1)[:, j:j+1] / Lk + 1e-7).astype(np.float32)
def quant_w(x, pq, k, sub):
    Lk = (1 << k) - 1; hh = qhat(x, pq, Lk, sub)
    q = np.clip(np.rint((x / hh + Lk) / 2), 0, Lk).astype(np.uint8)
    return [u64(np.packbits((q >> b) & 1, axis=-1)) for b in range(k)], hh
def split_w(x, pq, kout, kb, sub, names):  # per-token top-kout split + quantile step on residual (q/k/v and down)
    B, Fi = x.shape
    I = np.argpartition(np.abs(x), Fi - kout, axis=1)[:, Fi - kout:]
    xr = x.copy()
    for b in range(B): xr[b, I[b]] = 0.0
    pl, hh = quant_w(xr, pq, kb, sub); outs = []
    for n in names:
        P = PW[n]; y = a5gemv(pl, hh, P); g = np.float32(P["gw"])
        for b in range(B): ix = I[b]; y[b] += (x[b, ix] @ p.cols_rowmajor(P, ix)) * g
        outs.append(y)
    return outs
def hyb_w(x, pq, key, kd, kb, sub, names):  # 05:55 hyb_i8 with the quantile step
    S = STATIC[key]; T8 = I8[key]; si = S["idx"]; B, Fi = x.shape
    xr = x.copy(); xr[:, si] = 0.0
    I = np.argpartition(np.abs(xr), Fi - kd, axis=1)[:, Fi - kd:]
    xq = xr.copy()
    for b in range(B): xq[b, I[b]] = 0.0
    pl, hh = quant_w(xq, pq, kb, sub); xs = x[:, si]; outs = []
    for n in names:
        P = PW[n]; g = np.float32(P["gw"])
        y = a5gemv(pl, hh, P) + (xs @ T8[n]) * g
        for b in range(B): ix = I[b]; y[b] += (x[b, ix] @ p.cols_rowmajor(P, ix)) * g
        outs.append(y)
    return outs
_smm = h.smm  # = prev1055.smm chain
def smm(spec, x, names, ctx):
    if spec[0] == "Wq": return split_w(x, spec[1], spec[2], spec[3], spec[4], names)
    if spec[0] == "Wh": return hyb_w(x, spec[1], spec[2], spec[3], spec[4], spec[5], names)
    return _smm(spec, x, names, ctx)
h.smm = smm
GRID_P = [0.95, 0.97, 0.98, 0.985, 0.99, 0.993, 0.996, 0.998, 1.0]
SITES = {"attn": lambda pq, dn, sub: ("Wq", pq, 32, 5, sub), "o": lambda pq, dn, sub: ("Wh", pq, f"{dn}_o_32", 8, 4, sub),
         "ffn": lambda pq, dn, sub: ("Wh", pq, f"{dn}_ffn_32", 8, 4, sub), "down": lambda pq, dn, sub: ("Wq", pq, 32, 5, sub)}
def calibrate(dn, tok):
    cal = w.calibrate(dn, tok)  # 10:55 cal (also builds static lists + int8 cols for dn)
    xc = tok(np.random.default_rng(SEED + 1), 16); _, ycw, sites = layer(xc, wo_mm)
    score = lambda s, spec: statistics.mean(cos(smm(spec, sites[s], (q,), {})[0], ycw[q]) for q in site_proj[s])
    for sub in (1, 4, 8):
        tab = {s: {pq: score(s, SITES[s](pq, dn, sub)) for pq in GRID_P} for s in SITES}
        for s in SITES:
            pb = max(tab[s], key=tab[s].get); cal[f"{s}_W{sub}"] = [list(SITES[s](pb, dn, sub)), tab[s][pb]]
        g1 = {pq: statistics.mean(tab[s][pq] for s in SITES) for pq in GRID_P}; pg = max(g1, key=g1.get)
        cal[f"onep_W{sub}"] = pg; cal[f"onep_W{sub}_score"] = g1[pg]
    return cal
def c1055(C):  # 10:55 default: q/k/v win, hybrid top-8 int8 fixed c, down fixed c
    return w.cfg(C, C, 8, "attn_win")
def call(C, sub): return {s: tuple(C[f"{s}_W{sub}"][0]) for s in SITES}
def cone(C, sub, dn="t4"):
    pq = C[f"onep_W{sub}"]; return {s: SITES[s](pq, dn, sub) for s in SITES}
def variants(CAL):
    t4 = CAL["t4"]
    return {"H8_1055_default": c1055(t4), "H8_allwin_persite": call(t4, 1), "H8_allwin_onep": cone(t4, 1),
            "H8_allwin_persite_sub4": call(t4, 4), "H8_allwin_onep_sub4": cone(t4, 4),
            "H8_allwin_persite_sub8": call(t4, 8), "H8_allwin_onep_sub8": cone(t4, 8)}
TIMED = {"H8_1055_default": c1055, "H8_allwin_onep": lambda C: cone(C, 1), "H8_allwin_onep_sub4": lambda C: cone(C, 4),
         "H8_allwin_onep_sub8": lambda C: cone(C, 8)}
def qcost():  # micro: per-token step estimate cost at B=1
    r = np.random.default_rng(SEED + 7); out = {}
    for n in (D, base.F):
        x = (r.standard_t(4, (1, n)) * 0.05).astype(np.float32)
        fs = {"absmean": lambda: (0.2 * np.mean(np.abs(x), axis=1, keepdims=True)),
              "partition_full": lambda: qhat(x, 0.993, 31, 1), "partition_sub4": lambda: qhat(x, 0.993, 31, 4),
              "partition_sub8": lambda: qhat(x, 0.993, 31, 8)}
        for k, fn in fs.items():
            for _ in range(50): fn()
            s = []
            for _ in range(15):
                t0 = time.perf_counter()
                for _ in range(200): fn()
                s.append((time.perf_counter() - t0) / 200 * 1e6)
            out[f"{k}_{n}"] = {"median_us": statistics.median(s), "min_us": min(s), "max_us": max(s)}
    return out
def main():
    _, by = base.build()
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "cal": {}, "acc": {}, "time": {}}
    CAL = {"t4": calibrate("t4", tokf(*DRAWS["t4"]))}
    for dn, a in DRAWS.items():  # static lists for every draw (needed by w.calibrate chain) but configs use t4 only
        if dn != "t4": CAL[dn] = w.calibrate(dn, tokf(*a))
    res["cal"]["t4"] = {k: (v if not isinstance(v, list) else [list(v[0]) if isinstance(v[0], (list, tuple)) else v[0], v[1]]) for k, v in CAL["t4"].items()}
    print("cal t4", json.dumps(res["cal"]["t4"], default=str), flush=True)
    if "noeval" not in sys.argv:
        V = variants(CAL)
        for dn, a in DRAWS.items():
            tok = tokf(*a)
            for es in (2, 3):
                xe = tok(np.random.default_rng(SEED + es), 128)
                yw = np.concatenate([layer(xe[i:i+32], wo_mm)[0] for i in range(0, 128, 32)])
                for n, c in V.items():
                    ya = np.concatenate([layer(xe[i:i+32], None, c)[0] for i in range(0, 128, 32)])
                    d1, d0 = ya - xe, yw - xe; rc = rowcos(d1, d0)
                    r = {"delta_pooled": cos(d1, d0), "delta_min": min(rc), "rows_below_098": int(sum(v < 0.98 for v in rc)), "n": 128,
                         "rel_l2": float(np.linalg.norm((d1 - d0).astype(np.float64)) / np.linalg.norm(d0.astype(np.float64)))}
                    res["acc"].setdefault(dn, {}).setdefault(n, {})[f"seed{es}"] = r
                    print(dn, n, es, r["rows_below_098"], round(r["delta_min"], 4), flush=True)
    res["qcost"] = qcost(); print("qcost", json.dumps(res["qcost"]), flush=True)
    x1 = tokf(4, 0)(np.random.default_rng(SEED + 9), 1)
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, mk in TIMED.items(): fns["q_" + n] = (lambda c=mk(CAL["t4"]): layer(x1, None, c))
    for n, fn in fns.items(): res["time"][n] = timed(fn); print(n, res["time"][n], flush=True)
    t = res["time"]; tm = t["tmac_g8"]
    res["ratios"] = {n: {"vs_tmac": tm["median_ms"] / t[n]["median_ms"], "worst": tm["min_ms"] / t[n]["max_ms"]} for n in t if n.startswith("q_")}
    print(json.dumps(res["ratios"]), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
if __name__ == "__main__": main()
```

</details>

## Results

t4-calibrated p per site (identical in all 3 runs):

| quantile from | q/k/v | o (A4 hybrid) | gate/up (A4 hybrid) | down (A5 split 32) | one p |
|---|---|---|---|---|---|
| full | 0.993 | 0.985 | 0.985 | 0.996 | 0.993 |
| every 4th | 0.996 | 0.985 | 0.985 | 0.996 | 0.993 |
| every 8th | 1.0 | 0.99 | 0.985 | 0.996 | 0.996 |

Accuracy: worst per-token delta cosine over both seeds; tokens under 0.98 out of 256 per draw:

| config | t4 | t3 | t3hot8 | t3hot48 | t3hot48b | total |
|---|---|---|---|---|---|---|
| top-8, 10:55 default (quantile step at q/k/v only; fixed absmean c at o, gate/up, down) | 0.9848; **0/256** | 0.9851; **0/256** | 0.9851; **0/256** | 0.9845; **0/256** | 0.9838; **0/256** | 0/1280 |
| **quantile step at all four sites, per-site p (t4 cal)** | 0.9854; **0/256** | 0.9852; **0/256** | 0.9857; **0/256** | 0.9848; **0/256** | 0.9813; **0/256** | 0/1280 |
| **quantile step at all four sites, one p = 0.993 for the whole layer (t4 cal)** | 0.9842; **0/256** | 0.9845; **0/256** | 0.9851; **0/256** | 0.9840; **0/256** | 0.9842; **0/256** | 0/1280 |
| all four sites, per-site p, quantile from every 4th element | 0.9836; **0/256** | 0.9822; **0/256** | 0.9848; **0/256** | 0.9793; **1/256** | 0.9827; **0/256** | 1/1280 |
| all four sites, one p, every 4th element | 0.9842; **0/256** | 0.9830; **0/256** | 0.9841; **0/256** | 0.9796; **1/256** | 0.9829; **0/256** | 1/1280 |
| all four sites, per-site p, every 8th element | 0.9821; **0/256** | 0.9829; **0/256** | 0.9825; **0/256** | 0.9814; **0/256** | 0.9786; **2/256** | 2/1280 |
| all four sites, one p, every 8th element | 0.9808; **0/256** | 0.9807; **0/256** | 0.9811; **0/256** | 0.9795; **1/256** | 0.9771; **2/256** | 3/1280 |

Pooled delta cosine / rel-L2 (mean of two seeds), draws in the order t4 / t3 / t3hot8 / t3hot48 / t3hot48b:

| config | pooled cos | rel-L2 |
|---|---|---|
| top-8, 10:55 default (quantile step at q/k/v only; fixed absmean c at o, gate/up, down) | 0.9881 / 0.9880 / 0.9884 / 0.9883 / 0.9883 | 0.154 / 0.155 / 0.152 / 0.152 / 0.152 |
| **quantile step at all four sites, per-site p (t4 cal)** | 0.9884 / 0.9883 / 0.9888 / 0.9888 / 0.9888 | 0.152 / 0.153 / 0.150 / 0.149 / 0.149 |
| **quantile step at all four sites, one p = 0.993 for the whole layer (t4 cal)** | 0.9877 / 0.9876 / 0.9883 / 0.9881 / 0.9882 | 0.157 / 0.157 / 0.153 / 0.154 / 0.153 |
| all four sites, per-site p, quantile from every 4th element | 0.9880 / 0.9879 / 0.9886 / 0.9885 / 0.9885 | 0.155 / 0.155 / 0.151 / 0.151 / 0.151 |
| all four sites, one p, every 4th element | 0.9875 / 0.9875 / 0.9880 / 0.9879 / 0.9880 | 0.158 / 0.158 / 0.155 / 0.155 / 0.154 |
| all four sites, per-site p, every 8th element | 0.9874 / 0.9873 / 0.9880 / 0.9882 / 0.9880 | 0.158 / 0.159 / 0.155 / 0.153 / 0.155 |
| all four sites, one p, every 8th element | 0.9871 / 0.9869 / 0.9876 / 0.9876 / 0.9874 | 0.160 / 0.162 / 0.157 / 0.157 / 0.159 |

Step-estimate cost alone at B=1, µs median (runs 1 / 2 / 3):

| estimate | 1 × 2560 | 1 × 6912 |
|---|---|---|
| absmean (fixed-c reference) | 4.92 / 4.94 / 4.95 | 5.85 / 5.91 / 5.88 |
| full `np.partition` | 5.77 / 5.70 / 5.79 | 9.35 / 9.20 / 9.21 |
| every 4th element | 4.50 / 4.51 / 4.57 | 5.78 / 5.73 / 5.77 |
| every 8th element | 4.30 / 4.33 / 4.35 | 4.92 / 4.90 / 4.99 |

Full layer step at B=1 in ms, median (min–max), runs 1 / 2 / 3. Ratio = all-`tmac_g8` median / variant median (worst case = `tmac_g8` min / variant max):

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| top-8, 10:55 default | 13.19 (13.14–13.24) / 12.91 (12.86–13.28) / 12.94 (12.92–15.02) | 1.60x (1.59x) / 1.65x (1.60x) / 1.63x (1.39x) |
| all four sites, one p (full partition) | 13.18 (13.11–13.19) / 12.89 (12.84–12.92) / 12.90 (12.85–13.70) | 1.60x (1.60x) / 1.65x (1.64x) / 1.63x (1.53x) |
| all four sites, one p, every 4th | 13.18 (13.10–13.20) / 12.89 (12.80–13.29) / 12.89 (12.84–13.35) | 1.60x (1.60x) / 1.65x (1.60x) / 1.63x (1.57x) |
| all four sites, one p, every 8th | 13.15 (13.11–13.21) / 12.87 (12.83–13.07) / 12.91 (12.83–13.30) | 1.61x (1.59x) / 1.66x (1.63x) / 1.63x (1.57x) |
| all-`tmac_g8` | 21.15 (21.07–21.26) / 21.31 (21.24–21.53) / 21.04 (20.93–21.15) | 1.00x |
| `f32_resident` | 1.36 (1.31–2.06) / 1.29 (1.26–1.91) / 1.25 (1.23–1.43) | 15.5x / 16.5x / 16.8x |

Interleaved check: median of 7 per-round medians (min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 21.65 (21.13–22.39) | 1.00x |
| top-8, 10:55 default | 13.07 (12.96–13.15) | 1.66x |
| all four sites, one p (full partition) | 13.02 (12.99–13.19) | 1.66x |
| all four sites, one p, every 4th | 13.00 (12.96–13.19) | 1.67x |
| all four sites, one p, every 8th | 13.02 (12.98–13.05) | 1.66x |

Resident bytes are unchanged: 2-bit planes + row sums 17.55 MB, plus int8-trit static cols 0.52 MB = 18.07 MB (1.03x the planes, 3.84x below int8 69.47 MB). The quantile step adds no resident bytes.

## Verdict

**Viable** (vs packed `tmac_g8`, same 2-bit weight path), with the synthetic-data caveat below.

1. **The whole layer is now calibration-free.** With the quantile step at all four sites, a t4-only calibration holds **0/1280** both with per-site p and with **one p = 0.993 everywhere** (worst 0.9840, including the unseen hot set t3hot48b). Per-site t4 p lands at 0.985–0.996, so one global p loses little. The fixed per-site c at o, gate/up and down is no longer needed.
2. **No measurable speed cost.** The one-p layer runs 1.60–1.65x over 3 runs (1.66x interleaved), the same as the 10:55 default (1.60–1.65x, 1.66x interleaved). The four per-token partitions cost ≈27 µs per layer vs ≈21 µs for absmean, about 0.05% of a ~13 ms layer.
3. **Subsampling is not worth it.** Every 4th element loses 1/1280 and every 8th loses 2–3/1280 (worst 0.9771 on t3hot48b). The subsample misses the tail it is trying to estimate. It saves only ~1–4 µs per call, and the layer timings are indistinguishable. Keep the full `np.partition`.

Recommendation: packed B=1 default = **one Winsor-quantile p = 0.993 at every site** (q/k/v A5 + top-32 split, o and gate/up A4 hybrid top-8 int8-trit, down A5 top-32 split), full per-token partition, static lists calibrated on any light-tailed sample. The ship config is now one scalar plus two 32-channel lists.

**Caveat:** weights and tokens are synthetic and hot channels are fixed ×15 scales. `f32_resident` is still ~15–17x faster than `tmac_g8` at ~16x the memory, so this matters only on the memory-bound packed path.

## Pair files this wave

At Quant Chief 14:05 MT, pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave. Roadmap blobs unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666). JuniorLLM head before this push was e5d6751 (the 10:55 note). No Trit Pack, Handshake or LLC commit touched JuniorLLM in the last 6 hours. This note touches only `junior_osai/vault/findings/`.

## Next question

Do the static 32-channel lists still matter once every site uses the quantile step? Compare (a) dropping the static list at o and gate/up (pure dynamic top-8/16 + quantile step) against the hybrid on all five draws. Also check whether one p = 0.993 holds at B=8 batched decode, where per-token partitions scale with B.

close: sha256(prev || this)[:16] = b41cd3391cf1615a
