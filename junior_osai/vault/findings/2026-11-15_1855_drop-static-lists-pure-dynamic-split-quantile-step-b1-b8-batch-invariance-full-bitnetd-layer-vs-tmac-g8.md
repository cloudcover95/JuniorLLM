---
local: true
date: 2026-11-15
wave: "2026-11-15 18:55 (ran ~18:45–19:10 MT; Chief last_run 2026-11-16T01:12:00Z ~18:12 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, dynamic-split, static-list-ablation, winsor, quantile-step, one-p, batch-invariance, b8, tmac-g8, int8-expand, numpy, viable]
---

# The static 32-channel lists can go. With the one-p Winsor-quantile step everywhere, pure dynamic top-8 at o and gate/up holds **0/1280** (worst 0.9839) at 1.61–1.62x B=1 (1.64x interleaved) vs `tmac_g8`. That is slightly faster than the hybrid, and the layer drops to 17.55 MB (1.00x planes, 3.96x below int8). At B=8 one p is exactly batch-invariant, but the packed path is 5.5x slower than int8-expand, so B≥8 stays on int8/BLAS.

## Question

The prior note [[2026-11-15_1455_whole-layer-winsor-quantile-steps-one-p-calibration-free-subsampled-quantile-full-bitnetd-layer-b1-vs-tmac-g8]] made one Winsor-quantile p = 0.993 the whole-layer default (0/1280, t4-only calibration). The ship config was one scalar plus two static 32-channel lists (o and gate/up, int8-trit cols, 0.52 MB). Its next question:

1. **Static lists.** Do the static lists still matter once every site uses the quantile step? Compare pure dynamic per-token top-kd split (kd = 8/16/32/40, no static list) against the hybrid on all five draws.
2. **B=8.** Does one p = 0.993 hold at B=8 batched decode, where the per-token partitions scale with B? And what does the packed path cost there?

Quant Chief `last_tips.json` `last_run` 2026-11-16T01:12:00Z (~18:12 MT) set pair_A and pair_B to **no gap** ("18:05 11/15: blobs unchanged; both no gap"), so this note uses the lane's own next question.

## Method

- **Unchanged from 14:55/18:55/02:55/05:55/10:55/14:55:** layer, weights, kernels, metric, draws.
  - `base.py` byte for byte (sha256[:16] `a38c8245cd13d800`), `prev18.py` `542d52ae6f1eb4c6`, `prev0255.py` `ab695e26a598fc37`, `prev0555.py` `546dddb014321b86`, `prev1055.py` `d1a0f1d644195409`, and the 14:55 `bench.py` as `prev1455.py` `15103b7cfcff3b28`.
  - One full bitnetd decoder layer (hidden 2560, GQA 20 q / 5 KV heads, FFN 6912), absmean ternary weights in 2-bit mask/sign planes.
  - Accuracy = per-token cosine of the layer delta vs the weight-only W1.58 layer, 2 × 128 tokens per draw, draws t4, t3, t3hot8, t3hot48 and t3hot48b (unseen hot set).
- **Every config is calibrated on t4 only.** q/k/v and down are the 14:55 config for every variant (A5 + per-token top-32 split, quantile step, one p = 0.993). Only o and gate/up change:
  - the hybrid (static 32 int8-trit cols + dynamic top-8, A4), or
  - `("Wq", 0.993, kd, 4, 1)`, a per-token top-kd split with the remaining channels at A4 with the quantile step, and no static list.
- **Batch invariance:** max |y(B=8) − concat y(B=1)| on 8 t3hot48 tokens.
- **Timing:** B=1 and B=8, 3 runs (9 repeats × 5 passes, 3 warm), plus a 7-round interleaved B=1 check.
  - `tmac_g8` is a B=1 kernel, so at B=8 it runs as a per-token loop (its LUT is per token anyway).
  - `int8_expand` (int8 → f32 scratch + BLAS) and `f32_resident` are the dense references.
  - Run 1 (with eval) took 2m38s. 8 cores, numpy 2.5.4, offline, nothing bound, no downloads.

Commands (in `/workspace/bench1855_1115`, next to the copied files above):

```
python3 bench.py results_run1.json          # t4 calibration + 5 draws x 2x128 eval + batch invariance + B=1/B=8 timing
python3 bench.py results_run2.json noeval   # timing only
python3 bench.py results_run3.json noeval
python3 tinter.py                           # interleaved B=1 timing
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Do the static 32-channel lists still matter once every site uses the Winsor-quantile step? + B=8 batched decode (18:55 11/15).
Same layer, weights, kernels, draws and metric as 14:55/18:55/02:55/05:55/10:55/14:55; base.py, prev18.py, prev0255.py,
prev0555.py, prev1055.py and prev1455.py (14:55 bench.py) copied verbatim. Reference = 14:55 default: one p at all four sites,
o + gate/up A4 hybrid (static 32 int8-trit cols + dynamic top-8). Candidates: o + gate/up pure dynamic top-kd (kd 8/16/32/40)
split + quantile step, no static list, no int8 cols (17.55 MB, 1.00x planes). All calibrated on t4 only. Timing at B=1 and B=8
(tmac_g8 baseline = per-token LUT loop, as it is B=1 only). numpy-only, offline. Usage: python3 bench.py results_runN.json [noeval]"""
import json, statistics, sys, time
import numpy as np
import base, prev18 as p, prev1455 as q
from base import wo_mm, tmac_g8, f32res, i8scr, timed, cos, rowcos, D
SEED = base.SEED; site_proj = p.site_proj; DRAWS = q.DRAWS; tokf = q.tokf; layer = q.layer; smm = q.smm; GRID_P = q.GRID_P
KDS = (8, 16, 32, 40)
def calibrate():
    cal = q.calibrate("t4", tokf(*DRAWS["t4"]))  # 14:55 cal: static lists + int8 cols for t4, one p for hybrid layer
    xc = tokf(*DRAWS["t4"])(np.random.default_rng(SEED + 1), 16); _, ycw, sites = layer(xc, wo_mm)
    score = lambda s, spec: statistics.mean(cos(smm(spec, sites[s], (n,), {})[0], ycw[n]) for n in site_proj[s])
    for kd in KDS:
        tab = {s: {pq: score(s, ("Wq", pq, kd, 4, 1)) for pq in GRID_P} for s in ("o", "ffn")}
        cal[f"dyn{kd}"] = {s: max(tab[s], key=tab[s].get) for s in ("o", "ffn")}
        cal[f"dyn{kd}_score"] = {s: tab[s][cal[f"dyn{kd}"][s]] for s in ("o", "ffn")}
        cal[f"dyn{kd}_score_at_onep"] = {s: tab[s][cal["onep_W1"]] for s in ("o", "ffn")}
    return cal
def hyb(C):
    return q.cone(C, 1)
def dyn(C, kd):
    pq = C["onep_W1"]; c = q.cone(C, 1)
    c["o"] = ("Wq", pq, kd, 4, 1); c["ffn"] = ("Wq", pq, kd, 4, 1); return c
def variants(C):
    v = {"H8_1455_default": hyb(C)}
    for kd in KDS: v[f"D{kd}_nostatic"] = dyn(C, kd)
    return v
def per_tok(mm):
    return lambda P, x: np.concatenate([mm(P, x[i:i+1]) for i in range(x.shape[0])])
def main():
    _, by = base.build()
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "acc": {}, "time": {}}
    C = calibrate()
    res["cal"] = {k: (v if not isinstance(v, list) else [list(v[0]) if isinstance(v[0], (list, tuple)) else v[0], v[1]]) for k, v in C.items()}
    print("cal", json.dumps({k: res["cal"][k] for k in res["cal"] if k.startswith(("dyn", "onep"))}, default=str), flush=True)
    V = variants(C)
    if "noeval" not in sys.argv:
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
        # batch invariance: B=8 chunks give the same outputs as B=1 rows (all steps are per-token)
        x8 = tokf(*DRAWS["t3hot48"])(np.random.default_rng(SEED + 4), 8); bi = {}
        for n, c in V.items():
            y8 = layer(x8, None, c)[0]; y1 = np.concatenate([layer(x8[i:i+1], None, c)[0] for i in range(8)])
            bi[n] = float(np.max(np.abs(y8 - y1)))
        res["batch_invariance_maxabs"] = bi; print("batch_inv", bi, flush=True)
    for B in (1, 8):
        xb = tokf(4, 0)(np.random.default_rng(SEED + 9), B); T = {}
        fns = {"tmac_g8": lambda: layer(xb, per_tok(tmac_g8)), "f32_resident": lambda: layer(xb, f32res),
               "int8_expand": lambda: layer(xb, i8scr)}
        for n, c in V.items(): fns["q_" + n] = (lambda c=c: layer(xb, None, c))
        for n, fn in fns.items(): T[n] = timed(fn); print(B, n, T[n], flush=True)
        tm = T["tmac_g8"]
        R = {n: {"vs_tmac": tm["median_ms"] / T[n]["median_ms"], "worst": tm["min_ms"] / T[n]["max_ms"]} for n in T if n != "tmac_g8"}
        res["time"][f"B{B}"] = T; res.setdefault("ratios", {})[f"B{B}"] = R; print(B, json.dumps(R), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
if __name__ == "__main__": main()
```

</details>

<details><summary>tinter.py</summary>

```python
"""Interleaved B=1 timing: 7 rounds, each round times every config (REPEATS=5 x PASSES=5) in rotated order; reports median of round medians."""
import json, statistics
import numpy as np
import base, bench
base.build(); base.REPEATS = 5
C = bench.calibrate(); V = bench.variants(C)
x1 = bench.tokf(4, 0)(np.random.default_rng(base.SEED + 9), 1)
fns = {"tmac_g8": lambda: bench.layer(x1, base.tmac_g8)}
for n in ("H8_1455_default", "D8_nostatic", "D16_nostatic"): fns[n] = (lambda cfg=V[n]: bench.layer(x1, None, cfg))
names = list(fns); rounds = {n: [] for n in names}
for r in range(7):
    for n in names[r % len(names):] + names[:r % len(names)]: rounds[n].append(base.timed(fns[n])["median_ms"])
out = {n: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for n, v in rounds.items()}
for n in names: out[n]["vs_tmac"] = out["tmac_g8"]["median_ms"] / out[n]["median_ms"]
print(json.dumps(out, indent=1)); json.dump(out, open("tinter.json", "w"), indent=1)
```

</details>

## Results

t4 calibration for the pure-dynamic o / gate/up sites (16 tokens, mean cosine per site):

| split | best p (o / gate-up) | cos at best p | cos at one p = 0.993 |
|---|---|---|---|
| top-8 | 0.985 / 0.985 | 0.99524 / 0.99520 | 0.99500 / 0.99489 |
| top-16 | 0.98 / 0.98 | 0.99573 / 0.99567 | 0.99546 / 0.99532 |
| top-32 | 0.985 / 0.98 | 0.99621 / 0.99625 | 0.99593 / 0.99588 |
| top-40 | 0.98 / 0.98 | 0.99641 / 0.99643 | 0.99612 / 0.99609 |

The per-site best p sits at 0.98–0.985, and one p = 0.993 costs ≤ 0.0004 cosine, so the one-p rule is kept.

Accuracy: worst per-token delta cosine over both seeds; tokens under 0.98 out of 256 per draw:

| config (q/k/v, down = 14:55, one p) | t4 | t3 | t3hot8 | t3hot48 | t3hot48b | total |
|---|---|---|---|---|---|---|
| hybrid top-8, 14:55 default (static 32 int8-trit cols + dynamic top-8 at o, gate/up) | 0.9842; **0/256** | 0.9845; **0/256** | 0.9851; **0/256** | 0.9840; **0/256** | 0.9842; **0/256** | 0/1280 |
| **no static list: pure dynamic top-8 at o, gate/up** | 0.9840; **0/256** | 0.9854; **0/256** | 0.9857; **0/256** | 0.9842; **0/256** | 0.9839; **0/256** | 0/1280 |
| no static list: dynamic top-16 | 0.9844; **0/256** | 0.9860; **0/256** | 0.9854; **0/256** | 0.9847; **0/256** | 0.9846; **0/256** | 0/1280 |
| no static list: dynamic top-32 | 0.9873; **0/256** | 0.9863; **0/256** | 0.9870; **0/256** | 0.9865; **0/256** | 0.9839; **0/256** | 0/1280 |
| no static list: dynamic top-40 | 0.9879; **0/256** | 0.9865; **0/256** | 0.9878; **0/256** | 0.9867; **0/256** | 0.9841; **0/256** | 0/1280 |

Pooled delta cosine / rel-L2 (mean of two seeds), draws in the order t4 / t3 / t3hot8 / t3hot48 / t3hot48b:

| config | pooled cos | rel-L2 |
|---|---|---|
| hybrid top-8, 14:55 default (static 32 int8-trit cols + dynamic top-8 at o, gate/up) | 0.9877 / 0.9876 / 0.9883 / 0.9881 / 0.9882 | 0.157 / 0.157 / 0.153 / 0.154 / 0.153 |
| **no static list: pure dynamic top-8 at o, gate/up** | 0.9879 / 0.9878 / 0.9885 / 0.9883 / 0.9884 | 0.156 / 0.156 / 0.152 / 0.153 / 0.152 |
| no static list: dynamic top-16 | 0.9886 / 0.9886 / 0.9890 / 0.9892 / 0.9893 | 0.151 / 0.151 / 0.148 / 0.147 / 0.146 |
| no static list: dynamic top-32 | 0.9895 / 0.9894 / 0.9899 / 0.9900 / 0.9902 | 0.145 / 0.146 / 0.142 / 0.141 / 0.140 |
| no static list: dynamic top-40 | 0.9899 / 0.9897 / 0.9902 / 0.9904 / 0.9905 | 0.142 / 0.143 / 0.140 / 0.139 / 0.138 |

Batch invariance (max |B=8 − B=1| over the layer output): H8_1455_default 0.0e+00, D8_nostatic 0.0e+00, D16_nostatic 0.0e+00, D32_nostatic 0.0e+00, D40_nostatic 0.0e+00. Every step (top-k split, quantile, A4/A5 code, popcount GEMV) is per token, so **one p at B=8 gives exactly the B=1 accuracy above**.

Full layer at **B=1**, ms median (min–max), runs 1 / 2 / 3; ratio = `tmac_g8` median / variant median:

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 20.58 (20.46–20.73) / 20.69 (20.64–20.84) / 20.88 (20.71–21.70) | 1.00x |
| hybrid top-8 (14:55 default) | 13.03 (12.98–13.06) / 14.27 (13.02–17.63) / 13.10 (13.07–14.55) | 1.58x / 1.45x / 1.59x |
| **no static, dynamic top-8** | 12.81 (12.76–12.90) / 12.85 (12.81–12.91) / 12.89 (12.87–12.95) | 1.61x / 1.61x / 1.62x |
| no static, dynamic top-16 | 13.17 (13.15–13.23) / 13.25 (13.19–13.28) / 13.23 (13.22–13.38) | 1.56x / 1.56x / 1.58x |
| no static, dynamic top-32 | 14.05 (13.82–14.14) / 13.98 (13.91–14.06) / 14.01 (13.98–14.15) | 1.47x / 1.48x / 1.49x |
| no static, dynamic top-40 | 14.32 (14.29–14.35) / 14.40 (14.39–14.43) / 14.40 (14.37–14.45) | 1.44x / 1.44x / 1.45x |
| int8 expand + BLAS (69.47 MB) | 19.04 (17.61–20.60) / 23.92 (22.76–24.74) / 18.27 (17.18–28.04) | 1.08x / 0.86x / 1.14x |
| `f32_resident` (277.9 MB) | 1.66 (1.34–2.23) / 1.43 (1.28–2.06) / 1.47 (1.40–2.17) | 12.42x / 14.45x / 14.21x |

Interleaved B=1 check: median of 7 per-round medians (min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 20.91 (20.83–21.00) | 1.00x |
| hybrid top-8 (14:55 default) | 13.15 (13.01–13.46) | 1.59x |
| **no static, dynamic top-8** | 12.79 (12.68–13.03) | 1.64x |
| no static, dynamic top-16 | 13.17 (13.05–13.57) | 1.59x |

Full layer at **B=8** (8 tokens per call), ms median (min–max), runs 1 / 2 / 3:

| config | ms | vs per-token `tmac_g8` |
|---|---|---|
| all-`tmac_g8` (per-token loop) | 168.11 (165.46–170.15) / 167.62 (166.05–170.27) / 166.53 (164.97–168.67) | 1.00x |
| hybrid top-8 (14:55 default) | 103.83 (102.90–116.36) / 107.53 (106.12–109.69) / 105.23 (104.99–106.31) | 1.62x / 1.56x / 1.58x |
| **no static, dynamic top-8** | 101.80 (101.02–106.26) / 104.69 (103.85–106.63) / 104.12 (103.32–107.62) | 1.65x / 1.60x / 1.60x |
| no static, dynamic top-16 | 106.16 (105.10–107.04) / 107.35 (106.73–109.97) / 107.79 (106.49–109.24) | 1.58x / 1.56x / 1.54x |
| no static, dynamic top-32 | 111.03 (110.75–113.23) / 114.93 (112.45–117.17) / 112.69 (111.98–114.44) | 1.51x / 1.46x / 1.48x |
| no static, dynamic top-40 | 114.04 (113.29–116.02) / 116.95 (114.82–120.23) / 115.61 (114.86–117.31) | 1.47x / 1.43x / 1.44x |
| int8 expand + BLAS (69.47 MB) | 20.07 (19.24–21.40) / 19.07 (18.67–22.37) / 18.66 (17.16–20.16) | 8.38x / 8.79x / 8.92x |
| `f32_resident` (277.9 MB) | 7.43 (4.42–8.85) / 4.52 (4.11–5.56) / 4.33 (4.10–5.59) | 22.64x / 37.09x / 38.42x |

Resident bytes:
- 2-bit planes + row sums: 17.55 MB.
- Hybrid: + 0.52 MB int8-trit static cols = 18.07 MB (1.03x planes).
- **No static list: 17.55 MB (1.00x planes, 3.96x below int8 69.47 MB, 15.8x below f32).**

## Verdict

**Viable** at B=1 (vs packed `tmac_g8`, same 2-bit weight path), with the synthetic-data caveat below. **Not worth it at B=8** vs dense int8.

1. **The static lists are no longer needed.** With the quantile step at every site, pure dynamic top-8 at o and gate/up holds 0/1280 on all five draws (worst 0.9839), like the hybrid. Top-16/32/40 also hold 0/1280 and slightly raise pooled cosine. The hybrid's static channels were compensating for the fixed absmean clip, which the quantile step already removed.
2. **Dynamic top-8 is simpler and marginally faster.** It runs 1.61–1.62x over 3 runs (1.64x interleaved), against 1.45–1.59x (1.59x interleaved) for the hybrid. That is about 2% faster in ms, because it skips the 32-col int8 matvec. It also drops 0.52 MB (layer back to 1.00x the planes). Larger kd costs speed: top-32 and top-40 fall below 1.5x most likely because the per-token row-major column gather grows with kd.
3. **B=8: one p holds, and the outputs are bit-identical to B=1.** The packed path keeps its ~1.6x over a per-token `tmac_g8` loop, but dense int8-expand + BLAS amortizes the weight expansion across the batch and is **5.5x faster** than packed top-8 at B=8 (f32 resident 23x). The packed popcount GEMV scales linearly with B. So B≥8 should use int8/BLAS (or a batched packed kernel), and ternary's B=8 win is memory only (3.96x vs int8).

Recommendation: packed B=1 default = **one Winsor-quantile p = 0.993 at every site, per-token top-32 split at q/k/v and down, per-token top-8 split at o and gate/up, no static lists**. The ship config is now a single scalar plus kd constants, with nothing to calibrate per deployment. Dispatch B≥8 to int8-expand/BLAS.

**Caveat:** weights and tokens are synthetic and hot channels are fixed ×15 scales. `f32_resident` remains ~12–14x faster than `tmac_g8` at B=1 here at ~16x the memory, so this matters only on the memory-bound packed path.

## Pair files this wave

At Quant Chief 18:05 MT (`last_run` 01:12Z), pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave.
- Roadmap blobs are unchanged (JuniorLLM docs/BETA_TO_OS.md bac6923, JuniorHome docs/ROADMAP.md 4efb666).
- JuniorLLM head before this push was 0a8301b (the 14:55 note), so no Trit Pack, Handshake or LLC commit touched JuniorLLM in the last 6 hours. Chief's LLC skip path (JuniorHome 5b54913) is in another repo.
- This note touches only `junior_osai/vault/findings/`.

## Next question

Can a batched packed kernel close the B=8 gap? Options: expand the 2-bit planes once per tile to int8 and reuse them across the 8 tokens (a cache-blocked expand+GEMM that never holds a full int8 matrix), or a T-MAC LUT shared across the batch. Measure B=2/4/8/16 against int8-expand + BLAS at the same 17.55 MB resident and find the crossover B.

close: sha256(prev || this)[:16] = 228708d067f59220
