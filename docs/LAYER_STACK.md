# Layer stack

Picks the kernel by size. No bench. No LUT. No GGUF. JuniorLLM.

## Layers
- T0: n < 512*512. Python trit matvec.
- T1: n < 1024*1024. Bit-plane numpy.
- T2: n >= 1024*1024. Native C if .so present, else T1 fallback.

## Why
Bench showed numpy bit-plane wins at 512 and loses at 1024. Scalar C does not beat BLAS. The stack avoids a live bench by using the size cut.

## Files
- junior_osai/kernels/stack.py
- rails/linux/bitplane.c

## Status
Live. Additive. Local only.
