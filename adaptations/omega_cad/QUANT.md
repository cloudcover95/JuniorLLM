# Ternary methods next to CAD interp

Implementation for shop software: `JuniorEngrTools/software/quant/ternary.py`.

| Method | Map | Use here |
|--------|-----|----------|
| sign | sgn(x) | 1-bit, no sparsity |
| absmean | clip(round(W/Δ),-1,1), Δ=mean\|W\| | BitNet 1.58 weights |
| absmax act | int8 per token | activations |
| I2_S pack | 2-bit codes + scale | store trits, not bitnet.cpp |

Interp still will not extrude on absmean-only guesses.
