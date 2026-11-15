---
local: true
date: 2026-11-15
wave: "2026-11-15 10:55 (ran ~10:42–11:05 MT; Chief last_run 2026-11-15T17:10:00Z ~10:10 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, hybrid-select, qkv-clip, ablation, winsor, per-token-clip, calibration-free, robustness, tmac-g8, numpy, viable]
---

# The q/k/v clip alone explains the 05:55 calibration failure. A per-token Winsor-quantile q/k/v step (`h = quantile_0.993(|x_res|)/31`) calibrated only on light-tailed t4 holds **0/1280** on all five draws for both hybrid top-16 and top-8, at no measurable cost (top-8 1.63–1.68x over 3 runs, 1.64x interleaved vs `tmac_g8`)

## Question

The prior note [[2026-11-15_0555_hybrid-cross-distribution-calibration-int8-trit-cached-cols-full-bitnetd-layer-b1-vs-tmac-g8]] made the packed B=1 default the hybrid at o + gate/up with int8-trit cached columns and dynamic top-16, calibrated on a heavy-tailed hot draw. Calibrating on light-tailed t4 instead lost 12/256 tokens on the hot draws, and the note guessed that the q/k/v clip c (0.14 vs 0.24) was the sensitive piece. Its next question:

1. **Ablation.** Keep the t4 static lists and t4 o/ffn/down clips, and swap in only the heavy-tail q/k/v clip. Does that alone recover 0/256 on t3hot48 (and does the reverse swap break the heavy-tail config)?
2. **Adaptive clip.** Can a per-token adaptive q/k/v step remove the calibration-distribution dependence at no measurable cost vs fixed-c top-16?

Quant Chief `last_tips.json` `last_run` 2026-11-15T17:10:00Z (~10:10 MT) set pair_A and pair_B to **no gap**, so this note uses the lane's own next question.

## Method

- **Unchanged from 14:55/18:55/02:55/05:55:** layer, weights, kernels, metric, draws.
  - `base.py` byte for byte (sha256[:16] `a38c8245cd13d800`); `prev18.py` `542d52ae6f1eb4c6`, `prev0255.py` `ab695e26a598fc37`, and the 05:55 `bench.py` as `prev0555.py` `546dddb014321b86`.
  - One full bitnetd decoder layer at B=1 (hidden 2560, GQA 20 q / 5 KV heads, FFN 6912), absmean ternary weights in 2-bit mask/sign planes. Accuracy = per-token cosine of the layer delta vs the weight-only W1.58 layer, 2 × 128 tokens per draw, draws t4, t3, t3hot8, t3hot48 (hot set rng 777, ×15) and t3hot48b (unseen hot set, rng 778).
  - Base config = 05:55 default: q/k/v A5 + shared per-token split 32; o and gate/up hybrid A4 (static 32-ch int8-trit cols + dynamic top-kd); down A5 split 32.
- **Ablations A/B** mix two calibrations: one supplies the static lists and o/ffn/down clips, the other only the q/k/v spec.
- **Adaptive q/k/v step** on the residual after the per-token top-32 split (`x_res`), A5 (Lk = 31):
  - `rms`: `h = c · rms(x_res)` (grid 0.06–0.40).
  - `win`: `h = quantile_p(|x_res|) / 31`, one `np.partition` per token (grid p ∈ 0.90…1.0).
  - Both calibrated per draw on the same 16-token calibration sample as before; the "t4 cal" rows use the t4 value everywhere.
- **Timing.** B=1, 3 runs (9 repeats × 5 passes, 3 warm) plus a 7-round interleaved check; timed configs use t4 calibration. Run 1 (with eval) took 3m56s. 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench1055_1115`, next to `base.py`, `prev18.py`, `prev0255.py`, `prev0555.py`):

```
python3 bench.py results_run1.json          # calibration on 5 draws + 5 draws x 2x128 eval + timing
python3 bench.py results_run2.json noeval   # timing only
python3 bench.py results_run3.json noeval
python3 tinter.py                           # interleaved timing (05:55 tinter.py, unchanged)
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""q/k/v clip ablation + per-token adaptive q/k/v clip for the packed B=1 hybrid (10:55 11/15). Same layer, weights,
kernels, draws and metric as 14:55/18:55/02:55/05:55; base.py, prev18.py, prev0255.py and prev0555.py (05:55 bench.py)
copied verbatim. Base config = 05:55 default: q/k/v A5 + shared per-token split 32, o and gate/up hybrid (static 32-ch
int8-trit cols + dynamic top-kd) A4, down A5 split 32.
Question: (A) with t4 static lists + t4 o/ffn/down clips, does swapping in only the heavy-tail (t3hot48) q/k/v clip c
recover 0/256 on hot draws (and the reverse swap break it)? (B) can a per-token adaptive q/k/v step remove the
calibration-distribution dependence: 'rms' h = c*rms(x_res) or 'win' h = quantile_p(|x_res|)/31, calibrated on t4 only?
numpy-only, offline. Usage: python3 bench.py results_runN.json [noeval]"""
import json, statistics, sys
import numpy as np
import base, prev18 as p, prev0255 as h, prev0555 as f
from base import wo_mm, tmac_g8, f32res, timed, cos, rowcos, a5gemv, u64, PW, GRID5, D
SEED = base.SEED; site_proj = p.site_proj; DRAWS = f.DRAWS; tokf = f.tokf; layer = f.layer
def quant_ad(x, c, k, mode):
    Lk = (1 << k) - 1
    if mode == "rms": hh = (c * np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1, keepdims=True)) + 1e-7).astype(np.float32)
    else:
        m = x.shape[1]; j = min(m - 1, int(round(c * (m - 1))))
        hh = (np.partition(np.abs(x), j, axis=1)[:, j:j+1] / Lk + 1e-7).astype(np.float32)
    q = np.clip(np.rint((x / hh + Lk) / 2), 0, Lk).astype(np.uint8)
    return [u64(np.packbits((q >> b) & 1, axis=-1)) for b in range(k)], hh
def dyn_ad(x, c, kout, kb, mode, names):
    B, Fi = x.shape
    I = np.argpartition(np.abs(x), Fi - kout, axis=1)[:, Fi - kout:]
    xr = x.copy()
    for b in range(B): xr[b, I[b]] = 0.0
    pl, hh = quant_ad(xr, c, kb, mode); outs = []
    for n in names:
        P = PW[n]; y = a5gemv(pl, hh, P); g = np.float32(P["gw"])
        for b in range(B): ix = I[b]; y[b] += (x[b, ix] @ p.cols_rowmajor(P, ix)) * g
        outs.append(y)
    return outs
_smm = h.smm  # = prev0555.smm (handles Hi, H, S, s ...)
def smm(spec, x, names, ctx):
    if spec[0] == "Sa": return dyn_ad(x, spec[1], spec[2], spec[3], spec[4], names)
    return _smm(spec, x, names, ctx)
h.smm = smm
GRID_RMS = [round(0.06 + 0.02 * i, 2) for i in range(18)]
GRID_WIN = [0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.993, 0.996, 0.998, 1.0]
def calibrate(dn, tok):
    cal = f.calibrate(dn, tok)
    xc = tok(np.random.default_rng(SEED + 1), 16); _, ycw, sites = layer(xc, wo_mm)
    def calib(mk, grid):
        sc = {c: statistics.mean(cos(smm(mk(c), sites["attn"], (q,), {})[0], ycw[q]) for q in site_proj["attn"]) for c in grid}
        c = max(sc, key=sc.get); return [mk(c), sc[c]]
    cal["attn_rms"] = calib(lambda c: ("Sa", c, 32, 5, "rms"), GRID_RMS)
    cal["attn_win"] = calib(lambda c: ("Sa", c, 32, 5, "win"), GRID_WIN)
    return cal
def cfg(L, A, kd, attn="attn_S5"):
    """L: calibration supplying static lists + o/ffn/down clips; A: calibration supplying the q/k/v spec."""
    sp = lambda C, k: tuple(C[k][0]); hi = lambda s: ("Hi",) + s[1:]
    return {"attn": sp(A, attn), "o": hi(sp(L, f"o_H4_{kd}")), "ffn": hi(sp(L, f"ffn_H4_{kd}")), "down": sp(L, "down_s32")}
def variants(CAL, dn):
    t4, hz, m = CAL["t4"], CAL["t3hot48"], CAL[dn]
    return {"H16_cal_t4": cfg(t4, t4, 16), "H16_cal_hot48": cfg(hz, hz, 16),
            "H16_t4_attnc_hot48": cfg(t4, hz, 16), "H16_hot48_attnc_t4": cfg(hz, t4, 16),
            "H16_t4_rms": cfg(t4, t4, 16, "attn_rms"), "H16_t4_win": cfg(t4, t4, 16, "attn_win"),
            "H8_t4_rms": cfg(t4, t4, 8, "attn_rms"), "H8_t4_win": cfg(t4, t4, 8, "attn_win"),
            "H16_matched_win": cfg(m, m, 16, "attn_win")}
TIMED = {"H16_fixed": lambda C: cfg(C, C, 16), "H16_rms": lambda C: cfg(C, C, 16, "attn_rms"),
         "H16_win": lambda C: cfg(C, C, 16, "attn_win"), "H8_win": lambda C: cfg(C, C, 8, "attn_win")}
def main():
    _, by = base.build()
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "cal": {}, "acc": {}, "time": {}}
    CAL = {}
    for dn, a in DRAWS.items():
        CAL[dn] = calibrate(dn, tokf(*a)); res["cal"][dn] = {k: [list(v[0]), v[1]] for k, v in CAL[dn].items()}
        print("cal", dn, {k: res["cal"][dn][k] for k in ("attn_S5", "attn_rms", "attn_win")}, flush=True)
    if "noeval" not in sys.argv:
        for dn, a in DRAWS.items():
            tok = tokf(*a); V = variants(CAL, dn)
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
    x1 = tokf(4, 0)(np.random.default_rng(SEED + 9), 1)
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, mk in TIMED.items(): fns["q_" + n] = (lambda c=mk(CAL["t4"]): layer(x1, None, c))
    for n, fn in fns.items(): res["time"][n] = timed(fn); print(n, res["time"][n], flush=True)
    t = res["time"]; tm = t["tmac_g8"]
    res["ratios"] = {n: {"vs_tmac": tm["median_ms"] / t[n]["median_ms"], "worst": tm["min_ms"] / t[n]["max_ms"]} for n in t if n.startswith("q_")}
    print(json.dumps(res["ratios"]), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
if __name__ == "__main__": main()
```

</details>

## Results

Calibrated q/k/v settings per draw (and the hybrid/down clips they ship with):

| draw | q/k/v absmean c | q/k/v rms c | q/k/v winsor p | o H16 c | gate/up H16 c | down c |
|---|---|---|---|---|---|---|
| t4 | 0.14 | 0.1 | 0.993 | 0.21 | 0.21 | 0.26 |
| t3 | 0.16 | 0.12 | 0.996 | 0.21 | 0.21 | 0.26 |
| t3hot8 | 0.16 | 0.12 | 0.996 | 0.21 | 0.21 | 0.26 |
| t3hot48 | 0.24 | 0.16 | 0.996 | 0.18 | 0.21 | 0.26 |
| t3hot48b | 0.24 | 0.16 | 0.996 | 0.18 | 0.21 | 0.26 |

The fixed absmean c moves 0.14 → 0.24 (1.7x) from t4 to the hot draws; the Winsor quantile moves only 0.993 → 0.996.

Accuracy: worst per-token delta cosine over both seeds; tokens under 0.98 out of 256 per draw:

| config | t4 | t3 | t3hot8 | t3hot48 | t3hot48b | total |
|---|---|---|---|---|---|---|
| top-16, everything from t4 (05:55 failing case) | 0.9863; **0/256** | 0.9848; **0/256** | 0.9847; **0/256** | 0.9749; **7/256** | 0.9766; **9/256** | 16/1280 |
| top-16, everything from t3hot48 (05:55 default) | 0.9809; **0/256** | 0.9801; **0/256** | 0.9832; **0/256** | 0.9846; **0/256** | 0.9860; **0/256** | 0/1280 |
| **ablation A:** top-16, t4 lists + t4 o/ffn/down c, q/k/v c from t3hot48 | 0.9808; **0/256** | 0.9823; **0/256** | 0.9837; **0/256** | 0.9865; **0/256** | 0.9841; **0/256** | 0/1280 |
| **ablation B:** top-16, t3hot48 lists + o/ffn/down c, q/k/v c from t4 | 0.9840; **0/256** | 0.9829; **0/256** | 0.9826; **0/256** | 0.9743; **10/256** | 0.9751; **10/256** | 20/1280 |
| top-16, t4 cal, q/k/v step = c·rms(x_res) | 0.9862; **0/256** | 0.9855; **0/256** | 0.9847; **0/256** | 0.9771; **4/256** | 0.9792; **4/256** | 8/1280 |
| **top-16, t4 cal, q/k/v step = quantile_p(abs x_res)/31** | 0.9861; **0/256** | 0.9860; **0/256** | 0.9857; **0/256** | 0.9855; **0/256** | 0.9846; **0/256** | 0/1280 |
| top-8, t4 cal, rms step | 0.9848; **0/256** | 0.9839; **0/256** | 0.9844; **0/256** | 0.9756; **6/256** | 0.9780; **6/256** | 12/1280 |
| **top-8, t4 cal, winsor-quantile step** | 0.9848; **0/256** | 0.9851; **0/256** | 0.9851; **0/256** | 0.9845; **0/256** | 0.9838; **0/256** | 0/1280 |
| top-16, matched cal, winsor-quantile step | 0.9861; **0/256** | 0.9859; **0/256** | 0.9863; **0/256** | 0.9854; **0/256** | 0.9844; **0/256** | 0/1280 |

Pooled delta cosine / rel-L2 (mean of two seeds), draws in the order t4 / t3 / t3hot8 / t3hot48 / t3hot48b:

| config | pooled cos | rel-L2 |
|---|---|---|
| top-16, everything from t4 (05:55 failing case) | 0.9886 / 0.9883 / 0.9884 / 0.9857 / 0.9860 | 0.151 / 0.153 / 0.152 / 0.169 / 0.167 |
| top-16, everything from t3hot48 (05:55 default) | 0.9851 / 0.9863 / 0.9876 / 0.9900 / 0.9899 | 0.172 / 0.165 / 0.157 / 0.141 / 0.142 |
| **ablation A:** top-16, t4 lists + t4 o/ffn/down c, q/k/v c from t3hot48 | 0.9852 / 0.9861 / 0.9875 / 0.9894 / 0.9896 | 0.173 / 0.167 / 0.159 / 0.145 / 0.144 |
| **ablation B:** top-16, t3hot48 lists + o/ffn/down c, q/k/v c from t4 | 0.9884 / 0.9882 / 0.9886 / 0.9860 / 0.9863 | 0.152 / 0.153 / 0.150 / 0.167 / 0.165 |
| top-16, t4 cal, q/k/v step = c·rms(x_res) | 0.9886 / 0.9881 / 0.9884 / 0.9863 / 0.9866 | 0.151 / 0.154 / 0.152 / 0.165 / 0.163 |
| **top-16, t4 cal, q/k/v step = quantile_p(abs x_res)/31** | 0.9886 / 0.9885 / 0.9889 / 0.9890 / 0.9891 | 0.151 / 0.152 / 0.149 / 0.148 / 0.147 |
| top-8, t4 cal, rms step | 0.9881 / 0.9875 / 0.9879 / 0.9857 / 0.9859 | 0.154 / 0.157 / 0.155 / 0.169 / 0.167 |
| **top-8, t4 cal, winsor-quantile step** | 0.9881 / 0.9880 / 0.9884 / 0.9883 / 0.9883 | 0.154 / 0.155 / 0.152 / 0.152 / 0.152 |
| top-16, matched cal, winsor-quantile step | 0.9886 / 0.9886 / 0.9890 / 0.9899 / 0.9901 | 0.151 / 0.151 / 0.148 / 0.141 / 0.141 |

Full layer step at B=1 in ms, median (min–max), runs 1 / 2 / 3. Ratio = all-`tmac_g8` median / variant median (worst case = `tmac_g8` min / variant max):

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| top-16, fixed absmean q/k/v c (05:55 default) | 13.10 (13.07–13.34) / 13.22 (13.14–13.34) / 13.63 (13.50–14.58) | 1.62x (1.59x) / 1.57x (1.55x) / 1.58x (1.47x) |
| top-16, rms q/k/v step | 13.11 (13.04–13.16) / 13.07 (13.02–13.21) / 13.43 (13.38–14.24) | 1.62x (1.61x) / 1.59x (1.57x) / 1.61x (1.50x) |
| top-16, winsor-quantile q/k/v step | 13.13 (13.08–13.23) / 13.12 (13.10–13.14) / 13.53 (13.44–16.10) | 1.62x (1.60x) / 1.58x (1.58x) / 1.59x (1.33x) |
| top-8, winsor-quantile q/k/v step | 12.76 (12.72–12.89) / 12.77 (12.72–12.83) / 12.87 (12.74–18.39) | 1.67x (1.64x) / 1.63x (1.61x) / 1.68x (1.16x) |
| all-`tmac_g8` | 21.27 (21.14–21.47) / 20.78 (20.71–20.97) / 21.58 (21.40–33.77) | 1.00x |
| `f32_resident` | 1.35 (1.29–4.56) / 1.34 (1.30–2.05) / 2.32 (1.43–5.80) | 15.8x / 15.5x / 9.3x |

Interleaved check: median of 7 per-round medians (min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 21.21 (21.06–21.32) | 1.00x |
| top-16, fixed absmean q/k/v c (05:55 default) | 13.26 (13.07–13.89) | 1.60x |
| top-16, rms q/k/v step | 13.19 (13.07–13.68) | 1.61x |
| top-16, winsor-quantile q/k/v step | 13.32 (13.15–14.20) | 1.59x |
| top-8, winsor-quantile q/k/v step | 12.90 (12.83–13.51) | 1.64x |

Resident bytes are unchanged from 05:55: 2-bit planes + row sums 17.55 MB, plus int8-trit static cols 0.52 MB = 18.07 MB (1.03x the planes, 3.84x below int8 69.47 MB). The adaptive step adds no resident bytes.

## Verdict

**Viable** (vs packed `tmac_g8`, same 2-bit weight path), with the synthetic-data caveat below.

1. **The q/k/v clip is the whole story.** Ablation A (t4 lists + t4 o/ffn/down c, only the q/k/v c from t3hot48) goes from 16/1280 to **0/1280**. Ablation B (heavy-tail lists and clips, t4 q/k/v c) breaks the 05:55 default: 0 → **20/1280**. The static channel lists do not matter for robustness.
2. **An rms-scaled step is not enough:** t4-calibrated top-16 still loses 8/1280 (top-8 12/1280); rms(x_res) does not grow enough with the hot channels that survive the top-32 split.
3. **A per-token Winsor-quantile step removes the dependence.** Calibrated on t4 only (p = 0.993), top-16 is **0/1280** (worst 0.9846) and top-8 is **0/1280** (worst 0.9838), including the unseen hot set. That is also tighter than the 05:55 heavy-tail default (worst 0.9801 on t4), because a fixed heavy-tail c uses too coarse a step on light tokens while the quantile step tracks each token.
4. **No measurable cost.** Top-16 winsor 1.58–1.62x vs fixed-c 1.57–1.62x over 3 runs (1.59x vs 1.60x interleaved). Because top-8 is now also 0/1280, it becomes the default: **1.63–1.68x, 1.64x interleaved**, 18.07 MB.

Recommendation: packed B=1 default = **q/k/v A5 with a per-token Winsor-quantile step (p ≈ 0.993–0.996) after the shared top-32 split, hybrid top-8 int8-trit at o + gate/up, down A5 split 32**. Calibration no longer needs to see hot channels.

**Caveat:** weights and tokens are synthetic and hot channels are fixed ×15 scales. `f32_resident` is still ~9–16x faster than `tmac_g8` at ~16x the memory, so this matters only on the memory-bound packed path.

## Pair files this wave

At Quant Chief 10:10 MT, pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave. Roadmap blobs unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666). JuniorLLM head before this push was 8c1bad8 (the 05:55 note). This note touches only `junior_osai/vault/findings/`.

## Next question

Does the Winsor-quantile step also make the o / gate/up A4 clips and down A5 clip calibration-free, so the whole layer ships with one quantile p and no per-site c? And can the per-token `np.partition` be replaced by a cheap histogram/top-k estimate of the quantile without losing 0/1280?

close: sha256(prev || this)[:16] = 7df1cce267041b28
