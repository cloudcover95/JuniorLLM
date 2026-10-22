---
local: true
date: 2026-10-22
wave: "2026-10-22 05:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH (service-level memory fit of the I2_S path under bitnetd.service MemoryMax=2G)"
chief_assignment: "pair_A Live beta step 5 non-quant-error side; pair_B JuniorOS overlay remaining gaps (Quant Chief 05:15 MT, last_tips.json)"
verdict: viable
tags: [ternary, osai, findings, junioros-overlay, i2s, bitnetd, memorymax, rss, fit, numpy, viable]
---

# Under bitnetd MemoryMax=2G, only packed I2_S fits a whole 2B model (30/30 layers in 558 MB) vs 15 for fp16 and 7 for f32

Previous note: [[2026-10-22_0255_batched-blocked-lut-gemm-vs-f32-resident]]

## Question

The 02:55 note asked: since numpy decode cannot beat f32 on time, does the memory side pay off at the service level? Under the hardened unit's `MemoryMax=2G` (`rails/linux/bitnetd.service` line 32), how many 2B-model-sized layers fit resident as packed I2_S plus a 512 KiB decode buffer vs fp16 or f32 resident, and what is the per-token decode-time cost of that fit?

## Method

- Layer shape is BitNet b1.58 2B4T: q 2560x2560, k and v 640x2560, o 2560x2560, gate and up 6912x2560, down 2560x6912; 30 layers; 69.47M ternary weights per layer.
- Fit: one fresh python3 child per format allocates layers and touches every page, reading real RSS from `/proc/self/statm`, and stops before the next layer would push RSS past 2 GiB. I2_S holds packed uint8 bytes (4 trits/byte) plus one f32 gamma per matrix and the 512 KiB f32 decode buffer; fp16 also gets the 512 KiB buffer.
- Per-token time (k = 1): all 7 matvecs of one layer, Student-t(4) weights at scale 0.02, seed 555, quantized with the landed lean path (abs-mean gamma, threshold gamma/2, SWAR pack). Paths: f32-resident `W @ x`; fp16-resident blocked `copyto` into the f32 buffer then matvec; I2_S blocked `np.take` LUT decode into the buffer then matvec. 11 timed repeats after warm-up, median and min-max. Activations are int8-valued, so outputs must be bit-identical.
- Host: box, 8 cores, numpy 2.5.4, 16 GB RAM. Offline, numpy and stdlib only, nothing bound, no downloads.

Command: `python3 bench.py` (script below; writes results.json).

## Results

Identity: I2_S and fp16 outputs bit-identical to f32 on all 7 matrices (`identical True`, y sha16 17dc9598aed66a88).

### Fit under MemoryMax=2G (measured RSS, child baseline about 32.8 MB)

| format | bytes per layer | layers fit | RSS at stop | full 30-layer model fits |
|---|---|---|---|---|
| f32 resident | 277.87 MB | 7 | 1980.8 MB | no |
| fp16 resident + 512 KiB buffer | 138.94 MB | 15 | 2120.4 MB | no |
| I2_S packed + gamma + 512 KiB buffer | 17.37 MB | 30 (all) | 558.1 MB | yes, about 1589 MB headroom |

### Per-token decode time (k = 1), ms, median (min-max)

| path | one layer | 30 layers (x30) | vs f32 per layer |
|---|---|---|---|
| f32 resident | 1.41 (1.30-3.73) | 42.4, but only 7 layers fit | 1.00x |
| fp16 blocked | 302.56 (296.74-323.06) | 9077, but only 15 layers fit | 214x slower |
| I2_S blocked LUT | 32.44 (31.08-35.56) | 973 | 23x slower |

## Verdict

**Viable on memory, at a clear speed cost.** Under the real `MemoryMax=2G` cap, packed I2_S is the only format that holds the whole 2B model resident: 30 of 30 layers in 558 MB, 4.3x the layers of f32 (7) and 2x fp16 (15), with 16x and 8x smaller per-layer bytes and exact outputs. That clears the 1.5x memory bar with zero error. The price is speed on the numpy fallback: about 0.97 s per token for the full model, 23x slower per layer than f32 (which cannot run the whole model under the cap at all). fp16 is the worst of both here, because numpy's f16-to-f32 copy costs about 9x the I2_S LUT decode. So for the overlay, the cap itself decides the format: I2_S is what makes bitnetd fit, and the native kernel (not numpy) is what has to close the latency gap.

## This wave's pair files

- Pair A (Live beta step 5 non-QE): JuniorLLM [abcb4ff](https://github.com/cloudcover95/JuniorLLM/commit/abcb4ff) `scripts/backfill_interpreter_path_i2s_key.py` (interpreter resolution and min-version trits for backfill_check). No Winsor push seen by 05:55.
- Pair B (JuniorOS overlay gaps): Handshake, JuniorLLM [af5bc5c](https://github.com/cloudcover95/JuniorLLM/commit/af5bc5c) `ports/overlay_alpine_openrc_lane_handshake.py` (Alpine/OpenRC init lane). No Side-Commit push seen by 05:55.
- This note touches neither path.

## Next question

The Alpine/OpenRC handshake found no OpenRC carrier for the MemoryMax cap. With the whole model at 558 MB, what is the smallest cap that still holds 30 I2_S layers plus a KV cache for a 2048-token context (5 KV heads x 128, f32 vs fp16 vs int8 KV), and does an int8 or ternary KV cache give at least 1.5x more context per cap at acceptable attention-output error?

## Script

```python
#!/usr/bin/env python3
"""JuniorOS overlay, service-level memory fit: under a bitnetd MemoryMax=2G budget, how many
BitNet-2B-shaped transformer layers fit resident as packed I2_S (+ per-matrix f32 gamma + one
512 KiB decode buffer) vs fp16- or f32-resident, and what is the per-token (k=1) decode time of
each fit? Layer = q 2560x2560, k/v 640x2560, o 2560x2560, gate/up 6912x2560, down 2560x6912
(2B4T: 30 layers, hidden 2560, FFN 6912, 5 KV heads x 128). Fit is measured by real RSS in a
fresh child per format (pages touched), stopping before RSS would cross 2 GiB. Activations are
int8-valued so all paths must be bit-identical. numpy + stdlib only, offline, no sockets."""
import json, os, sys, time, hashlib, subprocess
import numpy as np

LAYER = [("q", 2560, 2560), ("k", 640, 2560), ("v", 640, 2560), ("o", 2560, 2560),
         ("gate", 6912, 2560), ("up", 6912, 2560), ("down", 2560, 6912)]
NLAYERS = 30
BUDGET = 2 << 30                      # MemoryMax=2G
BLOCK = 1 << 19                       # 512 KiB decode buffer (18:55 note)
REP = 11
SEED = 555
M = np.uint32((1 << 30) | (1 << 20) | (1 << 10) | 1); LO = np.uint32(0x01010101)
TRIT = np.array([0, 1, -1, 0], np.int8)
LUTF = np.array([[TRIT[(b >> s) & 3] for s in (6, 4, 2, 0)] for b in range(256)], np.float32)
PAGE = os.sysconf("SC_PAGE_SIZE")

def rss():
    with open("/proc/self/statm") as f: return int(f.read().split()[1]) * PAGE

def quant_pack(w):
    f = w.reshape(-1); g = np.float32(np.abs(f).mean(dtype=np.float32) + np.float32(1e-7)); T = np.float32(g * np.float32(0.5))
    q = (f > T).view(np.int8) - (f < -T).view(np.int8)
    x = q.view(np.uint32); c = (x & LO) + ((x >> 7) & LO); c *= M; c >>= 24
    return c.astype(np.uint8).reshape(w.shape[0], w.shape[1] // 4), q.reshape(w.shape), g

def layer_bytes(fmt):
    n = sum(r * c for _, r, c in LAYER)
    return {"f32": n * 4, "fp16": n * 2, "i2s": n // 4 + 4 * len(LAYER)}[fmt]

def fit_child(fmt):
    """Runs in a fresh interpreter: allocate+touch layers until the next one would cross BUDGET."""
    base = rss(); held = []; rng = np.random.default_rng(1)
    extra = np.empty(BLOCK // 4, np.float32) if fmt in ("i2s", "fp16") else None
    if extra is not None: extra.fill(0)
    per = layer_bytes(fmt); n = 0
    while n < NLAYERS and rss() + per <= BUDGET:
        lay = []
        for _, r, c in LAYER:
            if fmt == "f32": a = np.empty((r, c), np.float32); a.fill(1.0)
            elif fmt == "fp16": a = np.empty((r, c), np.float16); a.fill(1.0)
            else: a = rng.integers(0, 256, (r, c // 4), dtype=np.uint8); lay.append(np.float32(0.02))
            lay.append(a)
        held.append(lay); n += 1
    print(json.dumps({"fmt": fmt, "layers_fit": n, "rss_end": rss(), "rss_base": base, "per_layer": per,
                      "full_model_fits": n >= NLAYERS}))

def blocked(P, x, buf, y, lut):
    rows, nb = P.shape; R = max(1, min(rows, BLOCK // (nb * 16)))
    for s in range(0, rows, R):
        m = min(R, rows - s); b = buf[:m * nb * 4].reshape(m, nb, 4)
        np.take(lut, P[s:s + m], axis=0, out=b)
        np.matmul(b.reshape(m, -1), x, out=y[s:s + m])
    return y

def blocked16(W16, x, buf, y):
    rows, cols = W16.shape; R = max(1, min(rows, BLOCK // (cols * 4)))
    for s in range(0, rows, R):
        m = min(R, rows - s); b = buf[:m * cols].reshape(m, cols)
        np.copyto(b, W16[s:s + m], casting="unsafe")
        np.matmul(b, x, out=y[s:s + m])
    return y

def tstat(f, n=REP):
    f(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts); return {"med": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": n}

def main():
    out = {"numpy": np.__version__, "rep": REP, "budget": BUDGET, "block": BLOCK, "nlayers": NLAYERS, "fit": {}, "time": {}}
    for fmt in ("f32", "fp16", "i2s"):
        r = subprocess.run([sys.executable, __file__, "--fit", fmt], capture_output=True, text=True, check=True)
        out["fit"][fmt] = json.loads(r.stdout.strip().splitlines()[-1]); print(out["fit"][fmt], flush=True)
    rng = np.random.default_rng(SEED); mats = []
    for name, r, c in LAYER:
        P, q, g = quant_pack((rng.standard_t(4, size=(r, c)) * 0.02).astype(np.float32))
        mats.append((name, P, q.astype(np.float32), q.astype(np.float16)))
    x = {c: rng.integers(-127, 128, (c,)).astype(np.float32) for c in (2560, 6912)}
    buf = np.empty(BLOCK // 4, np.float32); ident = True; ys = {}
    for name, P, Wf, W16 in mats:
        ys[name] = (np.empty(P.shape[0], np.float32), np.empty(P.shape[0], np.float32))
        xi = x[Wf.shape[1]]; ref = Wf @ xi
        ident &= bool(np.array_equal(blocked(P, xi, buf, ys[name][0], LUTF), ref))
        ident &= bool(np.array_equal(blocked16(W16, xi, buf, ys[name][1]), ref))
    out["identical"] = ident; print("identical", ident, flush=True)
    paths = {
        "f32": lambda: [Wf @ x[Wf.shape[1]] for _, P, Wf, W16 in mats],
        "fp16_blocked": lambda: [blocked16(W16, x[Wf.shape[1]], buf, ys[n][1]) for n, P, Wf, W16 in mats],
        "i2s_blocked": lambda: [blocked(P, x[Wf.shape[1]], buf, ys[n][0], LUTF) for n, P, Wf, W16 in mats],
    }
    for k, f in paths.items():
        out["time"][k] = tstat(f); print(k, out["time"][k], flush=True)
    out["y_sha16"] = hashlib.sha256(b"".join((Wf @ x[Wf.shape[1]]).tobytes() for _, P, Wf, W16 in mats)).hexdigest()[:16]
    json.dump(out, open("results.json", "w"), indent=1)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fit": fit_child(sys.argv[2])
    else: main()
```

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-22_0255_batched-blocked-lut-gemm-vs-f32-resident.md.

close: c1dbf81ac4c7d0ec
