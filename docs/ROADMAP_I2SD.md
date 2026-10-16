# i2sd next

From OSai vault finding 1707ca5 (close 368f9370a796cfab).

## Known
- Memory: 2-bit pack is 16x smaller than fp32. Viable.
- Speed: numpy ternary matvec is 15-465x slower. Not viable without a native kernel.
- Error: ~0.5 on post-hoc ternarised float weights. Not a trained-ternary model.

## Next
LUT kernel on synthetic true ternary weights. No GGUF. No model pull. Target: beat fp32 at <=1% error, or document that it does not.

## Where
- junior_osai/vault/findings/2026-10-15_1930_ternary-matvec-baseline.md
- junior_osai/vault/findings/2026-10-15_2020_lut-kernel-next.md

## Status
Live. Additive. Local only.
