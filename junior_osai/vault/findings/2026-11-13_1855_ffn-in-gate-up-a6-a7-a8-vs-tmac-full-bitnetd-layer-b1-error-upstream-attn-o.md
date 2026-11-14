---
local: true
date: 2026-11-13
wave: "2026-11-13 18:55 (ran ~18:57–19:01 MT; Chief last_run 2026-11-14T01:06:30Z ~18:06 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (B=1 decode) / JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767"
verdict: not-viable
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, full-layer, gqa, a5, a6, a7, a8, bitplane-activation, ffn-in, gate-up, tmac-g8, numpy, not-viable]
---

# Raising the ffn-in (gate/up) input to A6/A7/A8 does not fix the A5 full-layer accuracy gap either: even exact f32 `tmac_g8` on gate/up **and** down leaves 7/32 tokens under 0.98 (pooled 0.980), so the error enters upstream at the attn/o A5 sites; A6 gate/up costs speed (1.48–1.53x) for +0.0008 pooled cos, so not viable

## Question

The prior note [[2026-11-13_1455_down-proj-input-a6-a7-a8-winsor-vs-tmac-down-full-bitnetd-layer-b1]] showed that more bits on the down input (A6/A7/A8/Winsor) or an exact `tmac_g8` down could not lift the A5 full-layer path to cos ≥ 0.98 on every token, and attributed the remaining loss to the gate/up (ffn-in) A5 site. Its next question: does raising only the ffn-in site to A6 or A7 (attn/o/down at A5) bring every token's layer delta to ≥ 0.98, and does the full layer then stay ≥ 1.5x over `tmac_g8` at B=1? Quant Chief `last_tips.json` `last_run` 2026-11-14T01:06:30Z (~18:06 MT) wrote pair_A/pair_B **no gap**, so the lane's own next question is used.

## Method

Same bitnetd GQA layer as the last two notes (hidden 2560, 20 q heads × 128, 5 KV heads, FFN 6912, ctx 1, RMSNorm, SiLU·up, both residuals; absmean-ternarised N(0, 0.02) weights; mask/sign packed GEMV, 2 popcounts per plane, T=1024), new seed 202611131855. Attn (q/k/v) and o stay A5 with per-site calibrated c. The **gate/up** input is A5, A6, A7 or A8 per-row absmean (one shared quant for gate and up, c calibrated per bit width on a 16-token draw), with down at A5 or A6; two ablations send gate/up (and then also down) through exact f32-input `tmac_g8`, which leaves only the attn/o sites quantised. Accuracy on a separate 32-token t(4)·0.05 draw vs the full weight-only W1.58 layer. Timing: full layer step at B=1, 9 repeats × 5 steps after 3 warmups, three full runs; numpy 2.5.4, single process, 8-core virtualised Xeon (load avg ~0.3 at start).

Command (offline, no network, nothing bound):

```
cd /workspace/bench1855_1113 && for i in 1 2 3; do python3 bench.py results_run$i.json > run$i.txt 2>&1; done
```

<details><summary>bench.py</summary>

```python
#!/usr/bin/env python3
"""ffn-in (gate/up) site bit width for the A5 full-layer path (18:55 11/13). Full bitnetd decoder-layer step at B=1 (GQA: hidden 2560, 20 q heads x 128, 5 KV heads -> k/v 640,
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
SEED = 202611131855
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
def site_mm(cfg, x, names, cdef):
    """cfg None -> A5 with calibrated cdef; ("tmac",) -> exact f32-input tmac_g8; (k, c) -> Ak absmean."""
    if cfg is None: pl, hh = quant5(x, cdef); return [a5gemv(pl, hh, PW[n]) for n in names]
    if cfg[0] == "tmac": return [np.concatenate([tmac_g8(PW[n], x[i]) for i in range(x.shape[0])]) for n in names]
    pl, hh = quantk(x, cfg[1], cfg[0]); return [a5gemv(pl, hh, PW[n]) for n in names]
def layer(x, mm, qmode=None, cs=None, fcfg=None, dcfg=None):
    """mm(P, x) -> y for the reference/baseline paths. qmode: A5 at attn/o; ffn-in and down per fcfg/dcfg."""
    h1 = rms(x)
    if qmode:
        q, k, v = site_mm(None, h1, ("q", "k", "v"), cs["attn"])
    else:
        q, k, v = (mm(PW[n], h1) for n in ("q", "k", "v"))
    att = np.repeat(v.reshape(-1, KVH, HD), QH // KVH, axis=1).reshape(-1, D)   # ctx=1 -> softmax=1
    o = site_mm(None, att, ("o",), cs["o"])[0] if qmode else mm(PW["o"], att)
    x2 = x + o; h2 = rms(x2)
    if qmode: g, u = site_mm(fcfg, h2, ("gate", "up"), cs["ffn"])
    else: g, u = mm(PW["gate"], h2), mm(PW["up"], h2)
    z = (silu(g) * u).astype(np.float32)
    dn = site_mm(dcfg, z, ("down",), cs["down"])[0] if qmode else mm(PW["down"], z)
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
FGRID = {6: [round(0.04 + 0.01 * i, 2) for i in range(15)], 7: [round(0.02 + 0.005 * i, 3) for i in range(15)],
         8: [round(0.008 + 0.002 * i, 3) for i in range(15)]}
def main():
    rng, by = build()
    tok = lambda r, n: (r.standard_t(4, (n, D)) * 0.05).astype(np.float32)
    xc = tok(np.random.default_rng(SEED + 1), 16)
    _, ycw, sites = layer(xc, wo_mm)
    site_proj = {"attn": ("q", "k", "v"), "o": ("o",), "ffn": ("gate", "up"), "down": ("down",)}
    cs = {}
    for s, projs in site_proj.items():
        cs[s] = max(CGRID, key=lambda c: statistics.mean(cos(a5gemv(*quant5(sites[s], c), PW[p]), ycw[p]) for p in projs))
    h2 = sites["ffn"]
    fc = {k: max(FGRID[k], key=lambda c: statistics.mean(cos(a5gemv(*quantk(h2, c, k), PW[p]), ycw[p]) for p in ("gate", "up")))
          for k in (6, 7, 8)}
    z = sites["down"]
    dc6 = max(FGRID[6], key=lambda c: cos(a5gemv(*quantk(z, c, 6), PW["down"]), ycw["down"]))
    variants = {"ffnA5": (None, None), "ffnA6": ((6, fc[6]), None), "ffnA7": ((7, fc[7]), None), "ffnA8": ((8, fc[8]), None),
                "ffnA7_downA6": ((7, fc[7]), (6, dc6)), "ffnA8_downA6": ((8, fc[8]), (6, dc6)),
                "ffnTmac": (("tmac",), None), "ffnTmac_downTmac": (("tmac",), ("tmac",))}
    print("c", cs, "fc", fc, "dc6", dc6, flush=True)
    xe = tok(np.random.default_rng(SEED + 2), 32)
    yw, pw, _ = layer(xe, wo_mm)
    ev = {"c": cs, "fc": fc, "dc6": dc6, "variants": {k: [list(a) if a else None for a in v] for k, v in variants.items()}, "acc": {}}
    for n, (f, d) in variants.items():
        ya, pa, _ = layer(xe, None, qmode=True, cs=cs, fcfg=f, dcfg=d)
        rc = rowcos(ya - xe, yw - xe)
        ev["acc"][n] = {"gate_cos": cos(pa["gate"], pw["gate"]), "up_cos": cos(pa["up"], pw["up"]),
                        "down_cos": cos(pa["down"], pw["down"]), "delta_pooled": cos(ya - xe, yw - xe),
                        "delta_min": min(rc), "rows_below_098": sum(r < 0.98 for r in rc),
                        "delta_relmae": float(np.mean(np.abs(ya - yw)) / np.mean(np.abs(yw - xe)))}
        print(n, ev["acc"][n], flush=True)
    x1 = tok(rng, 1)
    res = {"seed": SEED, "numpy": np.__version__, "bytes": by, "eval32": ev, "time": {}}
    fns = {"tmac_g8": lambda: layer(x1, tmac_g8), "f32_resident": lambda: layer(x1, f32res)}
    for n, (f, d) in variants.items():
        fns["a5_" + n] = (lambda f=f, d=d: layer(x1, None, qmode=True, cs=cs, fcfg=f, dcfg=d))
    for n, fn in fns.items():
        res["time"][n] = timed(fn); print(n, res["time"][n], flush=True)
    t = res["time"]; tm = t["tmac_g8"]
    res["ratios"] = {n: {"vs_tmac": tm["median_ms"] / t[n]["median_ms"], "worst": tm["min_ms"] / t[n]["max_ms"]}
                     for n in t if n.startswith("a5_")}
    print(json.dumps(res["ratios"]), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
if __name__ == "__main__": main()
```
</details>

## Results

Resident weight bytes per layer are the same for every variant: packed mask/sign 17,367,040 B (+ Σw 182,272 B), `tmac_g8` PT 17,367,040 B, int8 69,468,160 B (4.00x), f32 277,872,640 B (16.00x). Extra bitplanes cost only activation scratch (≤ 8 × 320 B per token at gate/up).

Calibrated c: attn 0.28, o 0.14, ffn (A5) 0.14, down (A5) 0.38; gate/up A6 0.07, A7 0.035, A8 0.02; down A6 0.18. Identical in all three runs, so accuracy is identical across runs. Note that o and ffn A5 both land on the lowest value of the c grid (0.14).

Accuracy on 32 eval tokens (cos vs weight-only layer):

| config | gate cos | up cos | down cos | layer delta pooled | worst token | tokens < 0.98 |
|---|---|---|---|---|---|---|
| gate/up A5, down A5 | 0.9825 | 0.9825 | 0.9529 | 0.9756 | 0.8839 | 12/32 |
| gate/up A6, down A5 | 0.9839 | 0.9840 | 0.9558 | 0.9764 | 0.8846 | 11/32 |
| gate/up A7, down A5 | 0.9843 | 0.9844 | 0.9565 | 0.9766 | 0.8856 | 11/32 |
| gate/up A8, down A5 | 0.9844 | 0.9845 | 0.9568 | 0.9767 | 0.8854 | 11/32 |
| gate/up A7, down A6 | 0.9843 | 0.9844 | 0.9627 | 0.9784 | 0.8889 | 8/32 |
| gate/up A8, down A6 | 0.9844 | 0.9845 | 0.9629 | 0.9784 | 0.8886 | 8/32 |
| gate/up tmac_g8 (f32), down A5 | 0.9845 | 0.9845 | 0.9569 | 0.9768 | 0.8855 | 11/32 |
| gate/up + down tmac_g8 (only attn/o at A5) | 0.9845 | 0.9845 | 0.9682 | 0.9797 | 0.8889 | 7/32 |

Full layer step at B=1, ms median (min–max), runs 1 / 2 / 3; ratio = all-`tmac_g8` median / variant median ("worst" = `tmac_g8` min / variant max):

| config | ms | vs all-`tmac_g8` |
|---|---|---|
| gate/up A5, down A5 | 12.97 (12.89–14.58) / 12.78 (12.75–12.85) / 13.16 (12.98–13.61) | 1.67x (1.43x) / 1.62x (1.61x) / 1.65x (1.55x) |
| gate/up A6, down A5 | 14.17 (13.96–14.88) / 14.06 (13.96–14.48) / 14.32 (14.23–14.76) | 1.53x (1.40x) / 1.48x (1.43x) / 1.52x (1.43x) |
| gate/up A7, down A5 | 15.17 (15.11–15.33) / 15.32 (15.26–15.55) / 15.54 (15.48–15.62) | 1.43x (1.36x) / 1.36x (1.33x) / 1.40x (1.35x) |
| gate/up A8, down A5 | 16.74 (16.65–17.01) / 17.01 (16.83–20.22) / 17.02 (16.96–17.52) | 1.30x (1.22x) / 1.22x (1.02x) / 1.28x (1.20x) |
| gate/up A7, down A6 | 15.79 (15.67–16.34) / 16.00 (15.93–17.46) / 16.14 (16.06–16.23) | 1.37x (1.27x) / 1.30x (1.18x) / 1.35x (1.30x) |
| gate/up A8, down A6 | 17.77 (17.29–19.96) / 17.56 (17.49–19.56) / 17.73 (17.56–19.02) | 1.22x (1.04x) / 1.18x (1.06x) / 1.23x (1.11x) |
| gate/up tmac_g8 (f32), down A5 | 15.95 (15.88–17.29) / 16.24 (16.13–16.60) / 16.53 (16.32–17.04) | 1.36x (1.20x) / 1.28x (1.24x) / 1.32x (1.24x) |
| gate/up + down tmac_g8 (only attn/o at A5) | 19.67 (18.64–20.25) / 18.70 (18.53–19.26) / 19.34 (18.70–19.47) | 1.10x (1.03x) / 1.11x (1.07x) / 1.12x (1.08x) |
| all-`tmac_g8` baseline | 21.69 (20.79–22.70) / 20.76 (20.66–21.70) / 21.76 (21.08–23.92) | 1.00x |
| `f32_resident` | 1.53 (1.48–2.31) / 1.53 (1.53–1.75) / 1.71 (1.54–2.49) | 14.2x / 13.5x / 12.7x |

## Verdict

**not-viable.**

1. The gate/up site is not the main source either. With gate/up made exact (`tmac_g8` on f32 input), gate/up cos is only 0.9845/0.9845, almost the same as A5 (0.9825). That means the gate/up *input* is already off: the error comes in through the residual from the A5 attn/o sites. With gate/up **and** down both exact, the layer still has 7/32 tokens under 0.98 (worst 0.889). This corrects the prior note, which blamed gate/up.
2. More gate/up bits barely help: pooled delta A5 0.9756 → A6 0.9764 → A7 0.9766 → A8 0.9767. The best mixed config (A7 or A8 gate/up + A6 down) reaches 0.9784 with 8/32 tokens under 0.98.
3. Speed: all-A5 stays 1.62–1.67x over `tmac_g8`, but gate/up A6 is 1.48–1.53x, A7 1.36–1.43x, A8 1.22–1.30x, A7+down A6 1.30–1.37x. Gate/up carries 2 × 6912 × 2560 of the layer's weights, so each extra plane there costs ~1 ms. A6 gate/up only just reaches the 1.5x bar in 2 of 3 runs, buys +0.0008 pooled cos, and still leaves 11/32 tokens under 0.98. Every config that does better is under 1.5x.
4. As before, `f32_resident` is ~13–14x faster than `tmac_g8` at 16x the memory; any win is only versus the packed fallback.

Keep `tmac_g8` as the packed fallback and keep A5 everywhere off the default `juniorctl ask` path. The FFN sites are ruled out; only attn/o are left.

## Pair files this wave

Quant Chief 18:06 MT: pair_A (Trit Pack & I2_S + Winsor & Quant Error) and pair_B (Side-Commit & Rate Lock + Handshake & Ports) both **no gap**, so this wave produced no pair files. Roadmap blobs are unchanged (JuniorHome docs/ROADMAP.md 4efb666, JuniorLLM docs/BETA_TO_OS.md bac6923). JuniorLLM head at clone was ce31d3f (the previous findings note); this note touches only `junior_osai/vault/findings/`.

## Next question

Is the gap set at the attn/o sites? The o-site (and ffn) c hit the floor of the 0.14–0.40 calibration grid. Does extending the grid down to 0.04, or raising only q/k/v and o to A6 (o is only 2560 × 2560, so it is cheap), bring every token to ≥ 0.98 while the full layer stays ≥ 1.5x over `tmac_g8` at B=1?

close: 36c6bdc8f0599185 (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-13_1455)
