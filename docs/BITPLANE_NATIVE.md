# Native bit-plane kernel

C kernel. Local .so. No LUT. No GGUF. JuniorLLM.

## Build
cd rails/linux && make
Produces libjunior_bitplane.so.

## Call
from rails.linux.bitplane_ffi import bitplane
y = bitplane(T, x, gamma=0.02)
Returns None if the .so is missing.

## Status
Live. Additive. Pair with junior_osai/kernels/bitplane.py.
