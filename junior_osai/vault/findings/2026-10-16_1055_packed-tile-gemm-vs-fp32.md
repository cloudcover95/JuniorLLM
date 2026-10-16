---
local: true
date: 2026-10-16T10:55:00-06:00
wave: "2026-10-16 10:05"
roadmap_item: "i2sd userspace 127.0.0.1:8767 no docker.sock"
roadmap_ref: "JuniorHome docs/ROADMAP.md #3 (Quant Chief pair_A assignment, wave 10:05)"
verdict: not-viable
tags: [osai, vault, findings, ternary, bitnet, jtr1, i2s-2b4t, i2sd, packed-resident, tile-decode, gemm, rss, benchmark]
---

# Packed-resident JTR1 / 2B4T tile-decode GEMM vs fp32-resident GEMM

- Prev: [[2026-10-16_0555_jtr1-load-vs-npy]]
- Answers: that note's next question (2B4T key-pack vs JTR1, and whether weights left packed for in-process i2sd use stay at least 1.5x ahead of fp32)
- Roadmap: i2sd userspace 127.0.0.1:8767 no docker.sock (JuniorHome docs/ROADMAP.md #3), the item Quant Chief gave pair_A this wave

## Question
The 05:55 note showed decode-on-load loses because numpy expands JTR1 to full fp32. The shape i2sd actually wants is weights kept packed in RAM and decoded one tile at a time inside the matmul. Does that packed-resident path beat a plain fp32-resident `W @ X` on resident memory without giving up throughput, for batch sizes from a single token (B=1) up to a prefill chunk (B=1024)? And does Trit Pack's I2_S 2B4T key-pack format decode any differently from JTR1?

## Method
- python3, numpy 2.5.4, 8 CPUs. No install, no model, no GGUF, no network, nothing bound.
- Reused the local seed-0 data in `/workspace/tbench/data` (4 layers of 4096x4096 exact ternary, gamma 0.02): `jtr1_00..03.bin` (JTR1 frames) and `fp32_00..03.npy` (the same weights as fp32).
- Forward pass through 4 chained layers, `X` is 4096xB float32 (seed 1055), B in 1, 16, 64, 256, 1024.
- fp32 path: weights resident as fp32, `X = W @ X` per layer.
- Packed paths: weights resident as the raw 2-bit payload (4 MiB per layer). For each 512-row tile, decode into one preallocated buffer, then `np.matmul(buf, X, out=Y[tile])`, then scale by gamma once per layer. Two decoders: `lut_f32` (256x4 float32 table via `np.take`) and `lut_i8` (256x4 int8 table, then cast into the float32 buffer).
- Each path runs in its own child process so `VmHWM` (peak RSS) is isolated; it is read after load and again after all batches. 1 warm-up plus 7 timed repeats per B, median with min-max reported. The whole run was repeated with the path order reversed.
- Error: relative L2 of each packed output against the fp32 output at the same B.
- 2B4T check: `pack_2b4t` from Trit Pack's `scripts/i2s_2b4t_key_pack.py` @ e8af871 on row 0 (4096 trits) compared byte-for-byte with the JTR1 payload.
- Bar applied: viable only if resident memory is at least 1.5x smaller and latency stays within 1.25x of fp32 at some B up to 1024, at rel err of 1e-5 or less.

Command: `cd /workspace/tbench && python3 bench_packed_tile_gemm_20261016.py fp32,lut_f32,lut_i8 run1 && python3 bench_packed_tile_gemm_20261016.py lut_i8,lut_f32,fp32 run2`

```python
#!/usr/bin/env python3
"""Packed-resident JTR1/2B4T tile-decode GEMM vs fp32-resident GEMM (wave 2026-10-16 10:05).

Reuses /workspace/tbench/data (seed 0, 4096x4096 exact ternary, gamma 0.02) written by
bench_jtr1_load_20261016.py. python3 + numpy only; no network, nothing bound.
"""
from __future__ import annotations
import gc, json, os, struct, sys, time, subprocess
from pathlib import Path
import numpy as np

DATA = Path("/workspace/tbench/data"); OUT = Path("/workspace/tbench/r1055")
M = K = 4096; L = 4; HEADER = 16; TILE = 512; BATCHES = (1, 16, 64, 256, 1024)
N_WARM = 1; N_REP = 7; SEED = 1055
DEC = {0: 0, 1: 1, 2: -1, 3: 0}
LUT_F32 = np.array([[DEC[(b >> s) & 3] for s in (6, 4, 2, 0)] for b in range(256)], np.float32)
LUT_I8 = LUT_F32.astype(np.int8)

def vmhwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"): return int(line.split()[1]) / 1024.0

def load_packed():
    out = []
    for i in range(L):
        fr = (DATA / f"jtr1_{i:02d}.bin").read_bytes()
        gamma, n = struct.unpack(">dI", fr[4:HEADER]); assert n == M * K
        out.append((np.float32(gamma), np.frombuffer(fr[HEADER:HEADER + n // 4], np.uint8).reshape(M, K // 4)))
    return out

def load_fp32():
    return [np.load(DATA / f"fp32_{i:02d}.npy") for i in range(L)]

def fwd_fp32(Ws, X):
    for W in Ws: X = W @ X
    return X

def make_fwd_packed(decoder):
    buf = np.empty((TILE, K), np.float32); buf8 = np.empty((TILE, K), np.int8)
    def fwd(Ps, X):
        for gamma, P in Ps:
            Y = np.empty((M, X.shape[1]), np.float32)
            for r in range(0, M, TILE):
                p = P[r:r + TILE]
                if decoder == "lut_f32":
                    np.take(LUT_F32, p, axis=0, out=buf.reshape(TILE, K // 4, 4))
                else:  # lut_i8 then cast
                    np.take(LUT_I8, p, axis=0, out=buf8.reshape(TILE, K // 4, 4))
                    buf[...] = buf8
                np.matmul(buf, X, out=Y[r:r + TILE])
            Y *= gamma; X = Y
        return X
    return fwd

def child(mode, outp):
    rng = np.random.default_rng(SEED)
    Xs = {B: rng.standard_normal((K, B)).astype(np.float32) for B in BATCHES}
    if mode == "fp32": Ws = load_fp32(); fwd = fwd_fp32
    else: Ws = load_packed(); fwd = make_fwd_packed(mode)
    rss_resident = vmhwm_mb(); res = {}
    for B in BATCHES:
        X = Xs[B]
        for _ in range(N_WARM): fwd(Ws, X)
        ts = []
        for _ in range(N_REP):
            t0 = time.perf_counter(); y = fwd(Ws, X); ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        res[B] = {"median_ms": float(np.median(ts)), "min_ms": float(ts.min()), "max_ms": float(ts.max()),
                  "checksum": float(np.abs(y).sum())}
        np.save(OUT / f"y_{mode}_B{B}.npy", y)
    wbytes = sum(W.nbytes for W in Ws) if mode == "fp32" else sum(P.nbytes for _, P in Ws)
    Path(outp).write_text(json.dumps({"mode": mode, "weight_bytes": wbytes, "vmhwm_after_load_mb": rss_resident,
                                      "vmhwm_peak_mb": vmhwm_mb(), "batches": res}, indent=2))

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child": return child(sys.argv[2], sys.argv[3])
    OUT.mkdir(parents=True, exist_ok=True)
    # 2B4T (Trit Pack e8af871) vs JTR1 payload byte identity on a 4096-trit slice
    sys.path.insert(0, "/workspace/JuniorLLM-trit-20261016/scripts")
    import i2s_2b4t_key_pack as kp
    T0 = np.load(DATA / "ref_trits0.npy", mmap_mode="r")
    fr0 = (DATA / "jtr1_00.bin").read_bytes()
    row = [int(t) for t in T0[0]]
    same = kp.pack_2b4t(row) == fr0[HEADER:HEADER + K // 4]
    order = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fp32", "lut_f32", "lut_i8"]
    tag = sys.argv[2] if len(sys.argv) > 2 else "run1"
    rows = {}
    for mode in order:
        outp = OUT / f"{tag}_{mode}.json"
        p = subprocess.run([sys.executable, __file__, "--child", mode, str(outp)], capture_output=True, text=True)
        if p.returncode: raise SystemExit(p.stderr)
        rows[mode] = json.loads(outp.read_text())
    errs = {}
    for mode in rows:
        if mode == "fp32": continue
        errs[mode] = {}
        for B in BATCHES:
            a = np.load(OUT / f"y_fp32_B{B}.npy"); b = np.load(OUT / f"y_{mode}_B{B}.npy")
            errs[mode][B] = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    base = rows["fp32"]
    print(f"2B4T payload == JTR1 payload (row 0, {K} trits): {same}")
    print("| path | resident weight bytes | VmHWM after load MB | VmHWM peak MB | " + " | ".join(f"B={B} ms (min-max)" for B in BATCHES) + " |")
    print("|---|---|---|---|" + "---|" * len(BATCHES))
    for mode, r in rows.items():
        cells = [f"{r['batches'][str(B)]['median_ms']:.1f} ({r['batches'][str(B)]['min_ms']:.1f}-{r['batches'][str(B)]['max_ms']:.1f})" for B in BATCHES]
        print(f"| {mode} | {r['weight_bytes']:,} | {r['vmhwm_after_load_mb']:.1f} | {r['vmhwm_peak_mb']:.1f} | " + " | ".join(cells) + " |")
    print("\n| path | " + " | ".join(f"B={B} speed vs fp32" for B in BATCHES) + " | rel err (max over B) |")
    print("|---|" + "---|" * (len(BATCHES) + 1))
    for mode, r in rows.items():
        if mode == "fp32": continue
        sp = [base['batches'][str(B)]['median_ms'] / r['batches'][str(B)]['median_ms'] for B in BATCHES]
        print(f"| {mode} | " + " | ".join(f"{s:.2f}x" for s in sp) + f" | {max(errs[mode].values()):.1e} |")
    (OUT / f"{tag}_summary.json").write_text(json.dumps({"same_2b4t_jtr1": same, "rows": rows, "errs": errs}, indent=2))

if __name__ == "__main__":
    raise SystemExit(main())
```

## Results
Run 1 (order fp32, lut_f32, lut_i8):

2B4T payload == JTR1 payload (row 0, 4096 trits): True
| path | resident weight bytes | VmHWM after load MB | VmHWM peak MB | B=1 ms (min-max) | B=16 ms (min-max) | B=64 ms (min-max) | B=256 ms (min-max) | B=1024 ms (min-max) |
|---|---|---|---|---|---|---|---|---|
| fp32 | 268,435,456 | 311.8 | 383.4 | 2.5 (1.8-3.1) | 6.8 (6.4-12.9) | 14.9 (13.1-19.1) | 44.5 (41.6-48.4) | 137.7 (130.9-154.8) |
| lut_f32 | 16,777,216 | 87.4 | 150.2 | 66.5 (61.5-69.8) | 71.1 (65.0-81.9) | 90.1 (89.0-93.8) | 128.1 (119.9-153.8) | 246.2 (221.9-283.6) |
| lut_i8 | 16,777,216 | 87.4 | 144.8 | 57.1 (52.8-61.3) | 59.6 (56.9-70.5) | 73.4 (67.3-84.4) | 112.0 (108.5-127.3) | 246.2 (244.3-296.7) |

| path | B=1 speed vs fp32 | B=16 speed vs fp32 | B=64 speed vs fp32 | B=256 speed vs fp32 | B=1024 speed vs fp32 | rel err (max over B) |
|---|---|---|---|---|---|---|
| lut_f32 | 0.04x | 0.10x | 0.17x | 0.35x | 0.56x | 9.4e-07 |
| lut_i8 | 0.04x | 0.11x | 0.20x | 0.40x | 0.56x | 9.4e-07 |

Run 2 (order reversed):

2B4T payload == JTR1 payload (row 0, 4096 trits): True
| path | resident weight bytes | VmHWM after load MB | VmHWM peak MB | B=1 ms (min-max) | B=16 ms (min-max) | B=64 ms (min-max) | B=256 ms (min-max) | B=1024 ms (min-max) |
|---|---|---|---|---|---|---|---|---|
| lut_i8 | 16,777,216 | 87.4 | 144.9 | 56.6 (54.3-61.0) | 62.0 (56.9-65.6) | 74.8 (68.1-80.9) | 112.6 (101.9-117.4) | 212.9 (208.0-241.2) |
| lut_f32 | 16,777,216 | 87.3 | 150.1 | 72.2 (63.7-75.3) | 73.7 (69.7-78.0) | 87.3 (81.0-92.3) | 122.9 (119.6-128.5) | 242.3 (231.0-265.1) |
| fp32 | 268,435,456 | 311.8 | 383.5 | 4.9 (3.3-6.6) | 6.0 (5.6-8.0) | 13.5 (11.5-16.4) | 40.7 (38.8-49.4) | 146.9 (144.3-154.7) |

| path | B=1 speed vs fp32 | B=16 speed vs fp32 | B=64 speed vs fp32 | B=256 speed vs fp32 | B=1024 speed vs fp32 | rel err (max over B) |
|---|---|---|---|---|---|---|
| lut_i8 | 0.09x | 0.10x | 0.18x | 0.36x | 0.69x | 9.4e-07 |
| lut_f32 | 0.07x | 0.08x | 0.15x | 0.33x | 0.61x | 9.4e-07 |

Memory summary: resident weight bytes are 16.0x smaller (16.8 MB vs 268.4 MB). Process peak RSS is 3.6x smaller after load (87.4 MB vs 311.8 MB) and 2.6x smaller at peak (145-150 MB vs 383 MB); the gap shrinks because the interpreter, numpy, activations and the 8 MiB tile buffer are shared fixed costs.

Speed summary: the packed paths are 11x to 27x slower at B=1, 9x to 12x slower at B=16, 5x to 6.5x slower at B=64, 2.5x to 3x slower at B=256 and 1.45x to 1.8x slower at B=1024. The decode costs about 55-70 ms for 4 layers (about 14-18 ms per 4096x4096 layer) no matter the batch size, and the 512-row tiled matmul is also less efficient than one full-size BLAS call.

Error: rel err 9.4e-07 at every B (float summation order only; decode is exact). 2B4T payload equals the JTR1 payload byte-for-byte (`True`), so the two formats decode at the same cost; there is no separate 2B4T result to chase.

## Verdict
**Not viable as an efficiency gain.** Keeping weights packed does deliver the memory win (16x on the weights, 2.6x to 3.6x on process RSS), but every batch size from 1 to 1024 is at least 1.45x slower than fp32-resident BLAS, which misses the "within 1.25x" half of the bar. It is a memory-for-latency trade, not a free gain. It only makes sense where the fp32 weights would not fit in RAM at all, for example the 12 W SBC class with a model larger than RAM. On numpy, the expand-to-fp32 step costs the same whether it happens at load time or per tile; the win needs a kernel that never expands to fp32 (bitnetd or a native I2_S kernel), not numpy.

## Ops check
`python3 scripts/terraform_lean_prod.py code` and `... llm` in local `/workspace/JuniorLLM` (offline): both exited 0 with `"ok": true`, `"bind": "127.0.0.1"`, `"bpy": false` (blender off), `"model_pull": false`, `"launch": false`.

## This wave's pair files (2026-10-16 10:05)
Quant Chief last_assignment: pair_A i2sd userspace 127.0.0.1:8767 no docker.sock; pair_B GGUF operator sets JUNIOR_GGUF, bots do not download.
- Trit Pack: [BitNet-mlx scripts/asahi_mlx_i2s_page_probe.py @ 8db848b](https://github.com/cloudcover95/BitNet-mlx/blob/8db848b/scripts/asahi_mlx_i2s_page_probe.py) (commit names ROADMAP #4 Asahi MLX, not the pair_A i2sd item)
- Winsor: [JuniorQuant src/i2sd_residual_bound.py @ 53da841](https://github.com/cloudcover95/JuniorQuant/blob/53da841/src/i2sd_residual_bound.py) (i2sd, matches pair_A)
- Side-Commit: [web3node gguf_receipt_ledger.py @ 3d41171](https://github.com/cloudcover95/web3node/blob/3d41171/gguf_receipt_ledger.py) (ROADMAP #2 GGUF, matches pair_B)
- Handshake & Ports: [JuniorHome ports/gguf_bitnetd_handshake.py @ 13ae8cc](https://github.com/cloudcover95/JuniorHome/blob/13ae8cc/ports/gguf_bitnetd_handshake.py) (ROADMAP #2 GGUF, matches pair_B)

## Next question
Can numpy skip the fp32 expand altogether for B=1? Quantize x to int8, split it into 8 bitplanes, keep the ternary weights as +1 and -1 bitmasks (`np.packbits`), and compute W@x as signed popcounts with `np.bitwise_count`. Does that beat fp32 BLAS matvec (about 0.6-1.2 ms per 4096x4096 layer here) by 1.5x, and what does int8 activation quantization cost in error?

Close rule: sha256(previous note full text || this note text before the close line)[:16], previous note = 2026-10-16_0555_jtr1-load-vs-npy.md.

close: fabaef8eaf2be669
