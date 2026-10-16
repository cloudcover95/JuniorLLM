---
local: true
date: 2026-10-15T19:30:00-06:00
wave: seed
roadmap_item: "JuniorHome docs/ROADMAP.md #3 i2sd userspace"
verdict: memory-viable-speed-not-viable
tags: [osai, vault, findings, ternary, bitnet, jtr1, matvec, benchmark]
---

# Ternary matvec baseline (numpy)

## Vault index
This is the OSai findings vault (`junior_osai/vault/findings/`). One note per bench run, named `YYYY-MM-DD_HHMM_slug.md`. Notes are chained by wikilink: each note links its predecessor and the next note links this one. Each note ends in a close line `close: sha256(prev || note)[:16]`, where `prev` is the previous note's close hash. Notes are `local: true` (see JuniorHome `docs/OBSIDIAN.md`); the vault is not synced.

- Prev: none (genesis note, prev = "")
- Next: (to be written)

## Question
For a BitNet-sized layer matvec (4096x4096, 2048x8192), does a JTR1-style ternary 2-bit pack (absmean gamma) beat fp32/fp16/int8 by >=1.5x on memory or speed at acceptable error, using python3 + numpy only?

## Method
- numpy 2.5.4, python3, CPU, seed 0. Weights N(0, 0.02), x N(0,1). No model or GGUF.
- Formats: fp32; fp16 storage (upcast to fp32 for the product); int8 per-row absmax (dequant); ternary {-1,0,1} with gamma = mean|W| (absmean), packed 4 trits/byte (2-bit codes, JTR1 payload; header magic+gamma+n not counted beyond 4 B gamma).
- Ternary paths: (a) masked add/sub, no multiplies; (b) unpack 2-bit pack + int32 matmul with int8-quantised x; (c) resident int8 trits @ fp32 (memory of the pack, compute on unpacked; reference only).
- 9 repeats each; median and min-max ms. Error = ||y - y_fp32|| / ||y_fp32||.
- Rate 0.05 not exercised here (no training step).

### Script
```python
import time, numpy as np
rng = np.random.default_rng(0)
def pack2(t):  # t in {-1,0,1} -> codes {0,1,2}, 4 per byte
    c = (t + 1).astype(np.uint8).reshape(t.shape[0], -1, 4)
    return c[...,0] | (c[...,1]<<2) | (c[...,2]<<4) | (c[...,3]<<6)
def unpack2(p):
    o = np.empty(p.shape + (4,), np.int8)
    for k in range(4): o[...,k] = ((p >> (2*k)) & 3)
    return (o.reshape(p.shape[0], -1) - 1).astype(np.int8)
def tm(f, r=9):
    ts=[]
    for _ in range(r):
        a=time.perf_counter(); f(); ts.append(time.perf_counter()-a)
    ts=np.array(ts)*1e3; return np.median(ts), ts.min(), ts.max()
for (M,K) in [(4096,4096),(2048,8192)]:
    W = rng.standard_normal((M,K)).astype(np.float32)*0.02
    x = rng.standard_normal(K).astype(np.float32)
    ref = W @ x
    W16 = W.astype(np.float16)
    s8 = np.abs(W).max(1, keepdims=True)/127; W8 = np.round(W/s8).astype(np.int8)
    g = np.abs(W).mean(); T = np.clip(np.round(W/g),-1,1).astype(np.int8); P = pack2(T)
    pos = T==1; neg = T==-1
    sx = np.abs(x).max()/127; x8 = np.round(x/sx).astype(np.int8)
    impl = {
      'fp32': (W.nbytes, lambda: W@x),
      'fp16 storage (upcast)': (W16.nbytes, lambda: W16.astype(np.float32)@x),
      'int8 (dequant)': (W8.nbytes+s8.nbytes, lambda: (W8.astype(np.float32)@x)*s8[:,0]),
      'ternary masked add/sub': (P.nbytes+4, lambda: g*(np.where(pos,x,0).sum(1)-np.where(neg,x,0).sum(1))),
      'ternary unpack+int matmul': (P.nbytes+4, lambda: g*sx*(unpack2(P).astype(np.int32)@x8.astype(np.int32)).astype(np.float32)),
      'ternary resident int8 @ fp32': (P.nbytes+4, lambda: g*(T.astype(np.float32)@x)),
    }
    print(f'\n### {M}x{K}\n| format | bytes | mem vs fp32 | median ms | min-max ms | speed vs fp32 | rel err |\n|---|---|---|---|---|---|---|')
    base=None
    for n,(b,f) in impl.items():
        y=f(); med,lo,hi=tm(f)
        if base is None: base=med
        e=np.linalg.norm(y-ref)/np.linalg.norm(ref)
        print(f'| {n} | {b:,} | {W.nbytes/b:.1f}x | {med:.2f} | {lo:.2f}-{hi:.2f} | {base/med:.2f}x | {e:.4f} |')
```

## Results
### 4096x4096
| format | bytes | mem vs fp32 | median ms | min-max ms | speed vs fp32 | rel err |
|---|---|---|---|---|---|---|
| fp32 | 67,108,864 | 1.0x | 0.33 | 0.30-0.85 | 1.00x | 0.0000 |
| fp16 storage (upcast) | 33,554,432 | 2.0x | 30.71 | 30.45-34.91 | 0.01x | 0.0002 |
| int8 (dequant) | 16,793,600 | 4.0x | 15.69 | 14.76-18.69 | 0.02x | 0.0086 |
| ternary masked add/sub | 4,194,308 | 16.0x | 141.71 | 132.18-152.05 | 0.00x | 0.5029 |
| ternary unpack+int matmul | 4,194,308 | 16.0x | 42.96 | 41.98-49.80 | 0.01x | 0.5030 |
| ternary resident int8 @ fp32 | 4,194,308 | 16.0x | 16.53 | 14.93-21.41 | 0.02x | 0.5029 |

### 2048x8192
| format | bytes | mem vs fp32 | median ms | min-max ms | speed vs fp32 | rel err |
|---|---|---|---|---|---|---|
| fp32 | 67,108,864 | 1.0x | 0.29 | 0.28-0.68 | 1.00x | 0.0000 |
| fp16 storage (upcast) | 33,554,432 | 2.0x | 30.94 | 28.32-46.34 | 0.01x | 0.0002 |
| int8 (dequant) | 16,785,408 | 4.0x | 14.42 | 13.85-15.50 | 0.02x | 0.0092 |
| ternary masked add/sub | 4,194,308 | 16.0x | 136.02 | 134.47-145.55 | 0.00x | 0.5179 |
| ternary unpack+int matmul | 4,194,308 | 16.0x | 41.83 | 37.64-47.73 | 0.01x | 0.5182 |
| ternary resident int8 @ fp32 | 4,194,308 | 16.0x | 14.43 | 13.28-40.22 | 0.02x | 0.5179 |

## Verdict
**Memory: viable.** The 2-bit pack is 16x smaller than fp32 and 8x smaller than fp16 (4.19 MB vs 67.1 MB / 33.6 MB), well past 1.5x.

**Speed: not viable in numpy.** Every ternary path is slower than fp32 BLAS: the fastest (resident int8 trits) is 30x+ slower at 4096x4096 and 15x slower at 2048x8192; true packed paths run 40-140 ms vs 0.3-0.6 ms. Reasons: numpy has no SIMD ternary/LUT kernel; the masked add/sub builds full temporaries; unpacking materialises int8 then int32 copies; integer matmul does not use BLAS. The fp16 and int8 timings are dominated by upcast costs too. A real win needs a native kernel (bitnet.cpp I2_S/TL style), which is the i2sd userspace step.

**Error: not acceptable post-hoc.** About 0.50-0.52 relative error from ternarising Gaussian fp32 weights after the fact. That is expected. BitNet weights are trained ternary (QAT), so this number does not measure a real b1.58 model. int8 is 0.009 and fp16 is 0.0002.

Net: the memory win is real, but there is no speed win in numpy and post-hoc error is too high. Ternary is only usable with trained-ternary weights plus a native kernel.

## Ops check
`python3 scripts/terraform_lean_prod.py code` and `... llm` were run in local /workspace/JuniorLLM (offline). Both exited 0, taking 0.021 s and 0.019 s. Output showed bind 127.0.0.1 and blender off.

## Next question
On already-ternary weights (no GGUF download; synthetic trained-ternary or on-disk only), does a LUT/bit-plane kernel inside i2sd (127.0.0.1:8767) beat fp32 matvec at <=1% error?

close: 368f9370a796cfab
