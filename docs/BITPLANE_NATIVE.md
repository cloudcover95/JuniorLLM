# Native bit-plane kernel

C kernel. Local .so. No LUT. No GGUF. JuniorLLM.

## Build
cd rails/linux && make
Produces libjunior_bitplane.so.

## Call
from rails.linux.bitplane_ffi import bitplane
y = bitplane(T, x, gamma=0.02)
Returns None if the .so is missing.

## Bench
Scalar C is correct (err 0) but slower than numpy BLAS at 512 and 1024. See junior_osai/vault/findings/2026-10-15_2030_native-bitplane.md.

## Next
SIMD on the inner accumulation.

## Status
Live. Additive. Pair with junior_osai/kernels/bitplane.py.
