# Quant + interp bench (this tree)

Measured on the clone runner after the explicit-HEIGHT short circuit:

| Path | µs |
|------|-----|
| sign | 8.6 |
| I2_S pack | 8.8 |
| absmean | 44 |
| interp explicit HEIGHT | 11 |
| interpret (Draft+IQ) | 76 |
| interp misc + IQ | 160 |

Sparsity on the 120-number fixture: 0.5. Packed: 30 bytes.

Finding: cost is rigid IQ, not AbsMean. Skip IQ when HEIGHT is already on the sheet.
Shop pack lives in JuniorEngrTools `software/quant`.
Omega `cad/legacy/quant_bridge.py` calls this module when present.
