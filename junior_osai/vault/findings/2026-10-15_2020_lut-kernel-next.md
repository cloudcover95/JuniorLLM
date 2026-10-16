---
local: true
date: 2026-10-15T20:20:00-06:00
wave: llc
roadmap_item: "JuniorHome docs/ROADMAP.md #3 i2sd userspace"
verdict: next-lut-kernel
tags: [osai, vault, findings, ternary, lut, i2sd]
---

# LUT kernel on true ternary weights (next)

## Vault index
- Prev: [[2026-10-15_1930_ternary-matvec-baseline]] close 368f9370a796cfab
- Next: (to be written)

## Question
On already-ternary weights (synthetic, no GGUF), does a lookup-table kernel beat fp32 matvec at <=1% error?

## Context
Baseline (1707ca5): 2-bit pack is 16x smaller than fp32. Numpy ternary matvec is 15-465x slower. Error ~0.5 from post-hoc ternarisation of float weights. Speed needs a native kernel in i2sd.

## Method (planned)
- Synthetic weights already in {-1,0,1}. No model pull.
- LUT: 256-entry table for 4 packed trits. Unpack once, or stream packed bytes.
- Compare median ms vs fp32 BLAS on same shape (4096x4096).
- Viable if >=1.5x faster or smaller and relative error <=0.01.

## Status
Not run. Seed only. Next wave measures it.

close: 7a1c2e4b9d0f3a6c
