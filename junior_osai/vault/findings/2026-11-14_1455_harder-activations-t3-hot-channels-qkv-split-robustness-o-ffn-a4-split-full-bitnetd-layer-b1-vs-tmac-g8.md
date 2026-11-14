---
local: true
date: 2026-11-14
wave: "2026-11-14 14:55 (ran ~14:45–14:55 MT; Chief last_run 2026-11-14T21:13:00Z ~14:13 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: inconclusive
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, a4, bitplane-activation, outlier-split, shared-split, robustness, heavy-tail, massive-activations, calibration-drift, tmac-g8, numpy, inconclusive]
---

# The 10:55 packed B=1 default is not robust to harder activations: on t(3) tokens with fixed hot channels it lets 7–24/256 tokens fall under 0.98 (97/256 on t3hot48 if c stays calibrated on t4). Splitting o and gate/up too (A4 + shared 32-ch split) fixes it, with 0/1024 tokens under 0.98 across all four draws when down is also split, but that mix runs only 1.50–1.53x (1.54x interleaved), right on the 1.5x bar

## Question

The prior note [[2026-11-14_1055_shared-outlier-split-qkv-a5-down-a5-cheaper-split-full-bitnetd-layer-b1-vs-tmac-g8]] made q/k/v A5 + one shared 32-channel outlier split (o, gate/up, down at plain A5) the packed B=1 default. It reached 0/256 tokens under 0.98 at 1.57x vs `tmac_g8`, but only on t(4)·0.05 synthetic tokens. Its next question: does it hold on harder, more LLM-like statistics, and can o / gate-up drop to A4 + split for more speed? No real BitNet-b1.58-2B-4T activation trace or per-channel outlier profile exists under /workspace (none downloaded, by rule), so this uses harder synthetic draws. Quant Chief `last_tips.json` `last_run` 2026-11-14T21:13:00Z (~14:13 MT) wrote pair_A/pair_B **no gap**, so this note uses the lane's own next question.

## Method

- **Layer, weights, kernels, metric:** unchanged from 10:55. `base.py` is that note's `bench.py`, copied byte for byte (sha256[:16] `a38c8245cd13d800`). It is one full bitnetd decoder layer at B=1: hidden 2560, GQA with 20 q heads and 5 KV heads, FFN 6912, absmean ternary weights held as 2-bit mask/sign planes (the same bytes as `tmac_g8`). Accuracy is the per-token cosine of the layer delta against the weight-only W1.58 layer.
- **Token draws** (residual stream): `t4` = t(4)·0.05 (the 10:55 draw); `t3` = t(3)·0.05; `t3hot8` and `t3hot48` = t(3)·0.05 with 8 or 48 fixed channels (the same for every token) scaled ×15, a stand-in for persistent "massive activation" channels. The 48-channel case has more hot channels than the 32-channel split.
- **Calibration:** each draw's c values are chosen per site on its own 16-token draw ("own-cal"). The 10:55 default calibrated on t4 is also run on every draw (`t4-cal`), which shows calibration drift.
- **New sites:** o and gate/up at A4 (4 activation bitplanes), plain or with one shared 32-channel split. The split channels go back in exactly through ternary columns unpacked from the same planes, so there are no extra resident bytes.
- **Eval:** 2 × 128 tokens per draw (seeds +2/+3), 1,024 tokens in all.
- **Timing:** B=1, 3 runs (9 repeats × 5 passes, 3 warm), plus a 7-round interleaved check. Timing uses argpartition with a fixed k, so it does not depend on the draw.
- **Setup:** 8 cores, numpy 2.5.4, offline, nothing bound, no downloads. Run 1 (with eval) took 2m33s.

Commands (in `/workspace/bench1455_1114`, next to `base.py`):

```
python3 bench.py results_run1.json          # calibration + 4 draws x 2x128 eval + timing
python3 bench.py results_run2.json noeval   # timing only
python3 bench.py results_run3.json noeval
python3 tinter.py                           # interleaved timing
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""Robustness of the 10:55 11/14 packed B=1 default (q/k/v A5 + one shared 32-channel outlier split; o, gate/up,
down plain A5) under harder activation statistics, and whether o and gate/up can drop to A4 (+ shared split 32).
Same full bitnetd decoder layer, weights, kernels and accuracy metric as base.py (copied verbatim from the 10:55 note).
Token draws (residual stream, 2560 ch): t4 = t(4)*0.05 (10:55); t3 = t(3)*0.05; t3hot8 / t3hot48 = t(3)*0.05 with
8 / 48 fixed channels (same for every token) scaled x15 (persistent 'massive activation' channels).
Each draw is calibrated on its own 16-token set ("own-cal"); the 10:55 config calibrated on t4 is also run on every
draw ("t4-cal") to show calibration drift. Eval 2 x 128 tokens per draw. numpy-only, offline.
Usage: python3 bench.py results_runN.json [noeval]"""
import json, statistics, sys
import numpy as np
import base
from base import layer, site_mm, wo_mm, tmac_g8, f32res, timed, cos, rowcos, GRID5, D
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
def calibrate(tok):
    xc = tok(np.random.default_rng(SEED + 1), 16)
    _, ycw, sites = layer(xc, wo_mm)
    def calib(s, mk, grid):
        sc = {c: statistics.mean(cos(site_mm(mk(c), sites[s], (p,))[0], ycw[p]) for p in site_proj[s]) for c in grid}
        c = max(sc, key=sc.get); return [mk(c), sc[c]]
    cal = {f"{s}_5": calib(s, lambda c: (5, c), GRID5) for s in site_proj}
    cal["attn_S5"] = calib("attn", lambda c: ("S", c, 32, 5), GRID5 + [0.02, 0.03])
    cal["down_s32"] = calib("down", lambda c: ("s", c, 32), GRID5)
    for s in ("o", "ffn"):
        cal[f"{s}_4"] = calib(s, lambda c: (4, c), GRID4)
        cal[f"{s}_S4"] = calib(s, lambda c: ("S", c, 32, 4), GRID4)
    return cal
def variants(cal):
    sp = lambda k: tuple(cal[k][0]); base_ = {s: sp(f"{s}_5") for s in site_proj}
    rec = {**base_, "attn": sp("attn_S5")}
    return {"allA5": base_, "rec_1055": rec, "rec_downS32": {**rec, "down": sp("down_s32")},
            "rec_oA4_ffnA4": {**rec, "o": sp("o_4"), "ffn": sp("ffn_4")},
            "rec_oS4_ffnS4": {**rec, "o": sp("o_S4"), "ffn": sp("ffn_S4")},
            "rec_oS4_ffnS4_downS32": {**rec, "o": sp("o_S4"), "ffn": sp("ffn_S4"), "down": sp("down_s32")}}
def main():
    base.build()
    res = {"seed": SEED, "numpy": np.__version__, "cal": {}, "variants": {}, "acc": {}, "time": {}}
    V = {}
    for dn, (df, nh) in HOT.items():
        cal = calibrate(tokf(df, nh)); res["cal"][dn] = cal; V[dn] = variants(cal)
        res["variants"][dn] = {k: {s: list(v) for s, v in c.items()} for k, c in V[dn].items()}
        print("cal", dn, cal, flush=True)
    if "noeval" not in sys.argv:
        for dn, (df, nh) in HOT.items():
            tok = tokf(df, nh); runs = dict(V[dn])
            if dn != "t4": runs["rec_1055_t4cal"] = V["t4"]["rec_1055"]
            for es in (2, 3):
                xe = tok(np.random.default_rng(SEED + es), 128)
                yw = np.concatenate([layer(xe[i:i+32], wo_mm)[0] for i in range(0, 128, 32)])
                for n, cfg in runs.items():
                    ya = np.concatenate([layer(xe[i:i+32], None, cfg)[0] for i in range(0, 128, 32)])
                    d1, d0 = ya - xe, yw - xe; rc = rowcos(d1, d0)
                    r = {"delta_pooled": cos(d1, d0), "delta_min": min(rc), "p05": float(np.percentile(rc, 5)),
                         "rows_below_098": int(sum(v < 0.98 for v in rc)), "n": 128,
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
import json, statistics, sys
import numpy as np
import base, bench
base.build(); base.REPEATS = 5
V = bench.variants(bench.calibrate(bench.tokf(4, 0)))
x1 = bench.tokf(4, 0)(np.random.default_rng(base.SEED + 9), 1)
fns = {"tmac_g8": lambda: base.layer(x1, base.tmac_g8)}
for n in ("allA5", "rec_1055", "rec_oS4_ffnS4", "rec_oS4_ffnS4_downS32", "rec_oA4_ffnA4"):
    fns[n] = (lambda cfg=V[n]: base.layer(x1, None, cfg))
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
| t4 | attn_5 0.28, o_5 0.12, ffn_5 0.12, down_5 0.38, attn_S5 0.14, down_s32 0.26, o_4 0.21, o_S4 0.18, ffn_4 0.21, ffn_S4 0.18 |
| t3 | attn_5 0.38, o_5 0.12, ffn_5 0.12, down_5 0.4, attn_S5 0.16, down_s32 0.26, o_4 0.21, o_S4 0.18, ffn_4 0.21, ffn_S4 0.18 |
| t3hot8 | attn_5 0.4, o_5 0.12, ffn_5 0.14, down_5 0.36, attn_S5 0.16, down_s32 0.26, o_4 0.21, o_S4 0.18, ffn_4 0.21, ffn_S4 0.18 |
| t3hot48 | attn_5 0.4, o_5 0.12, ffn_5 0.26, down_5 0.38, attn_S5 0.24, down_s32 0.26, o_4 0.21, o_S4 0.18, ffn_4 0.33, ffn_S4 0.18 |

Accuracy per draw: pooled delta cos (seed +2 / +3); worst token over both seeds; **tokens under 0.98 out of 256**:

| config | t4 | t3 | t3hot8 | t3hot48 |
|---|---|---|---|---|
| all A5 (no split) | 0.9769/0.9783; worst 0.8025; **91/256** | 0.9608/0.9626; worst 0.4279; **221/256** | 0.9013/0.9029; worst 0.3423; **255/256** | 0.8060/0.8047; worst 0.2826; **256/256** |
| 10:55 default: q/k/v A5 + shared split 32; o, gate/up, down A5 | 0.9914/0.9913; worst 0.9868; **0/256** | 0.9908/0.9911; worst 0.9514; **2/256** | 0.9903/0.9897; worst 0.8999; **7/256** | 0.9846/0.9863; worst 0.8678; **24/256** |
| 10:55 default, c calibrated on t4 (drift) | — | 0.9907/0.9908; worst 0.9502; **2/256** | 0.9898/0.9892; worst 0.8982; **9/256** | 0.9779/0.9788; worst 0.8623; **97/256** |
| 10:55 default + down A5 split 32 | 0.9939/0.9938; worst 0.9917; **0/256** | 0.9934/0.9935; worst 0.9557; **2/256** | 0.9929/0.9921; worst 0.9030; **5/256** | 0.9871/0.9885; worst 0.8681; **18/256** |
| 10:55 default with o, gate/up at plain A4 | 0.9842/0.9842; worst 0.9780; **5/256** | 0.9836/0.9838; worst 0.9454; **8/256** | 0.9829/0.9823; worst 0.8931; **27/256** | 0.9749/0.9761; worst 0.8601; **170/256** |
| **10:55 default with o, gate/up at A4 + shared split 32** | 0.9876/0.9875; worst 0.9831; **0/256** | 0.9873/0.9875; worst 0.9824; **0/256** | 0.9878/0.9881; worst 0.9813; **0/256** | 0.9884/0.9887; worst 0.9792; **1/256** |
| **o, gate/up A4 + shared split 32, down A5 + split 32** (all sites split) | 0.9901/0.9900; worst 0.9876; **0/256** | 0.9898/0.9900; worst 0.9858; **0/256** | 0.9905/0.9904; worst 0.9878; **0/256** | 0.9910/0.9909; worst 0.9866; **0/256** |

Mean relative L2 error of the layer delta (t4 / t3 / t3hot8 / t3hot48):

| config | rel L2 |
|---|---|
| all A5 (no split) | 0.211 / 0.274 / 0.432 / 0.597 |
| 10:55 default: q/k/v A5 + shared split 32; o, gate/up, down A5 | 0.131 / 0.134 / 0.141 / 0.170 |
| 10:55 default, c calibrated on t4 (drift) | — / 0.136 / 0.145 / 0.207 |
| 10:55 default + down A5 split 32 | 0.111 / 0.115 / 0.122 / 0.156 |
| 10:55 default with o, gate/up at plain A4 | 0.177 / 0.180 / 0.186 / 0.220 |
| **10:55 default with o, gate/up at A4 + shared split 32** | 0.157 / 0.158 / 0.155 / 0.151 |
| **o, gate/up A4 + shared split 32, down A5 + split 32** (all sites split) | 0.141 / 0.142 / 0.138 / 0.135 |

Full layer step at B=1 in ms, median (min–max), for runs 1 / 2 / 3. Ratio = all-`tmac_g8` median / variant median, with worst case (`tmac_g8` min / variant max) in parentheses:

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| all A5 (no split) | 13.31 (12.84–15.45) / 12.76 (12.69–12.84) / 12.91 (12.76–13.65) | 1.60x (1.37x) / 1.61x (1.60x) / 1.62x (1.53x) |
| 10:55 default: q/k/v A5 + shared split 32; o, gate/up, down A5 | 13.44 (13.17–15.68) / 12.99 (12.92–13.03) / 13.06 (13.04–13.42) | 1.59x (1.35x) / 1.58x (1.58x) / 1.61x (1.55x) |
| 10:55 default + down A5 split 32 | 13.81 (13.70–15.28) / 13.39 (13.21–14.97) / 13.41 (13.37–13.48) | 1.55x (1.38x) / 1.54x (1.37x) / 1.56x (1.55x) |
| 10:55 default with o, gate/up at plain A4 | 11.75 (11.70–16.00) / 11.51 (11.42–11.84) / 11.65 (11.55–11.78) | 1.82x (1.32x) / 1.79x (1.73x) / 1.80x (1.77x) |
| **10:55 default with o, gate/up at A4 + shared split 32** | 13.57 (13.42–13.77) / 13.46 (13.38–13.73) / 13.47 (13.43–13.54) | 1.57x (1.54x) / 1.53x (1.49x) / 1.56x (1.54x) |
| **o, gate/up A4 + shared split 32, down A5 + split 32** (all sites split) | 13.97 (13.82–14.18) / 13.74 (13.69–14.00) / 13.79 (13.75–13.92) | 1.53x (1.49x) / 1.50x (1.47x) / 1.52x (1.50x) |
| all-`tmac_g8` | 21.34 (21.16–26.15) / 20.58 (20.53–21.86) / 20.97 (20.85–21.33) | 1.00x |
| `f32_resident` | 1.51 (1.37–1.71) / 1.58 (1.41–2.64) / 1.72 (1.45–3.08) | 14.1x / 13.0x / 12.2x |

Interleaved check: median of 7 per-round medians (min–max of those medians):

| config | ms | vs `tmac_g8` |
|---|---|---|
| all-`tmac_g8` | 21.23 (21.16–21.57) | 1.00x |
| all A5 (no split) | 12.95 (12.90–12.99) | 1.64x |
| 10:55 default: q/k/v A5 + shared split 32; o, gate/up, down A5 | 13.48 (13.43–13.53) | 1.58x |
| **10:55 default with o, gate/up at A4 + shared split 32** | 13.50 (13.48–13.59) | 1.57x |
| **o, gate/up A4 + shared split 32, down A5 + split 32** (all sites split) | 13.80 (13.76–13.84) | 1.54x |
| 10:55 default with o, gate/up at plain A4 | 11.93 (11.90–11.98) | 1.78x |

Bytes are unchanged from 10:55: every quantised variant and `tmac_g8` read the same 2-bit packed weights, 4.00x below int8 and 16x below f32.

## Verdict

**Inconclusive.**

1. **The 10:55 default does not hold up.**
   - It stays clean on t4 (0/256) but loses tokens on t3 (2/256, worst 0.9514), t3hot8 (7/256, worst 0.8999) and t3hot48 (24/256, worst 0.8678).
   - With c left calibrated on t4, t3hot48 gets worse: 97/256 under 0.98.
   - The hot channels hurt at o and gate/up, not only at q/k/v. Adding a down split barely helps (18/256 on t3hot48).
2. **Plain A4 at o and gate/up is fast but not accurate.** It runs 1.79–1.82x (1.78x interleaved), but even on t4 5/256 tokens fall under 0.98, and on t3hot48 170/256.
3. **A4 + shared 32-channel split at o and gate/up keeps the speed of the 10:55 default and is far more robust.**
   - Speed: 1.53–1.57x in three runs and 1.57x interleaved, the same as the 10:55 default (1.58x).
   - Accuracy: 0/256 on t4, t3 and t3hot8, and 1/256 on t3hot48 (worst 0.9792).
   - It beats the 10:55 default on every hard draw at equal cost. This is the better default.
4. **Splitting every site** (o, gate/up A4 + split 32, down A5 + split 32) is the only mix with **0/1024** tokens under 0.98 (worst 0.9858).
   - Speed: 1.50–1.53x in three runs (one run is under 1.5x) and 1.54x interleaved.
   - That is on the bar, not clearly above it, so it is not called viable.
5. **The caveat stands.** These are still synthetic Gaussian weights and synthetic tokens. `f32_resident` is still ~12–14x faster than `tmac_g8` at 16x the memory, so this is a win for the memory-bound packed path only.

Recommendation: replace the 10:55 default with **q/k/v A5 + shared split 32, o and gate/up A4 + shared split 32, down A5**. It runs at the same speed (~1.57x vs `tmac_g8`) and is much more robust to heavy tails and hot channels. Add the down split only if real traces show more than 32 persistent hot channels. Calibrate c on representative tokens, because t4-calibrated c fails badly on hot channels.

## Pair files this wave

Quant Chief 14:13 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) were both **no gap**, so there are no pair files this wave. Roadmap blobs are unchanged: JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923. JuniorLLM head before this push was adce445, the 10:55 note; nobody else pushed in the last 6 hours. This note touches only `junior_osai/vault/findings/`.

## Next question

Can the all-sites split clear 1.5x with margin? Two cheaper options to test:

- one argpartition per token for the residual stream, reused for the q/k/v and gate/up inputs (both are rms of related residuals);
- a fixed calibrated hot-channel list (static index set, no per-token select) at o and gate/up, with argpartition only at down.

Target: 0/1024 under 0.98 across t4/t3/t3hot8/t3hot48 at ≥ 1.6x interleaved.

close: sha256(prev || this)[:16] = e89061af3ba4e6ec
