# Bit-plane kernel

On already-ternary weights. No LUT. No GGUF. No model pull. JuniorLLM.

## Why not LUT
LUT expands the pack. Trit stays packed. Bit-plane keeps the two planes (pos, neg) and does mask @ x.

## Files
- junior_osai/kernels/bitplane.py
- scripts/bitplane_bench.py

## Viable
>=1.5x faster than fp32 and relative error <=0.01. Otherwise document that it does not.

## Status
Live. Additive. Local only.
