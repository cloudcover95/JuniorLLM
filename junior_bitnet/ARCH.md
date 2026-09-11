# JuniorBitNet stack (what we actually ship)

Weights in {-1,0,1}. Scale Δ = mean(|W|). Activations are a separate int8 map.

```
scan/DXF → sidecar
    → compile_sheet (actions; no IQ if height missing)
    → /fix signed by a person
    → Draft + absmean trit-agree
    → Omega project OBJ/STL
```

Benefits we can measure: 30 bytes / 120 trits, explicit-height interp ~15× cheaper than IQ, no 1.5TB weights.
Benefits we do not claim: ImageNet parity, Jolt-speed SNARKs, licensed CU.
