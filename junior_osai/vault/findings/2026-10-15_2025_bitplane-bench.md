---
local: true
date: 2026-10-15T20:25:00-06:00
wave: llc
roadmap_item: "JuniorHome docs/ROADMAP.md #3 i2sd userspace"
verdict: size-dependent
tags: [osai, vault, findings, ternary, bitplane, i2sd]
---

# Bit-plane bench on synthetic ternary weights

## Vault index
- Prev: [[2026-10-15_2020_lut-kernel-next]] close 7a1c2e4b9d0f3a6c
- Next: (to be written)

## Question
On already-ternary weights (synthetic, no GGUF), does a bit-plane kernel beat fp32 matvec at <=1% error?

## Method
- Weights already in {-1,0,1}. No model pull. No LUT.
- Bit-plane: pos = (T==1), neg = (T==-1); y = gamma * (pos @ x - neg @ x).
- Compare median ms vs fp32 BLAS. 5 repeats. numpy CPU.

## Results
| shape | fp32 ms | bitplane ms | speed vs fp32 | rel err | mem vs fp32 | viable |
|---|---|---|---|---|---|---|
| 512x512 | 0.317 | 0.130 | 2.44x | 0.0 | 16.0x | yes |
| 1024x1024 | 0.724 | 0.839 | 0.86x | 0.0 | 16.0x | no |

## Verdict
Error is 0 because weights are already ternary. Memory is 16x. Speed is size-dependent: viable at 512, not at 1024. Numpy boolean matmul does not beat BLAS at larger shapes. A native kernel is still required for a consistent speed win.

## Next
Native bit-plane or I2_S kernel in i2sd. No LUT. No GGUF. Re-bench at 1024 and 4096.

close: b3e8f1a2c4d7096e
