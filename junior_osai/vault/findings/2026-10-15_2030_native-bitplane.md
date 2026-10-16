---
local: true
date: 2026-10-15T20:30:00-06:00
wave: llc
roadmap_item: "JuniorHome docs/ROADMAP.md #3 i2sd userspace"
verdict: correct-not-faster
tags: [osai, vault, findings, ternary, bitplane, native]
---

# Native bit-plane bench

## Vault index
- Prev: [[2026-10-15_2025_bitplane-bench]] close b3e8f1a2c4d7096e
- Next: (to be written)

## Question
Does a C bit-plane kernel beat fp32 BLAS at <=1% error?

## Method
- rails/linux/bitplane.c: scalar pos-minus-neg loop.
- Compiled -O2 -fPIC -shared.
- Synthetic ternary weights. No GGUF. No LUT.

## Results
| shape | fp32 ms | native ms | speed | err | viable |
|---|---|---|---|---|---|
| 512x512 | 0.122 | 1.782 | 0.07x | 0.0 | no |
| 1024x1024 | 0.744 | 7.219 | 0.10x | 0.0 | no |

## Verdict
Correct. Error is 0. Not faster. A scalar C loop does not beat numpy BLAS. Speed needs SIMD on the inner accumulation or a blocked kernel.

## Next
NEON/SSE2 on the pos/neg accumulation. No LUT. No GGUF.

close: c4d9e2a1b7f3085d
