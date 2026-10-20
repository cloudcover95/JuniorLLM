# Layered ternary compute

Local only. No model pull.

## Layers
ports/layer_compute.py scores a note with trit density/stability and absmean energy.
Combined score selects T0 / T1 / T2.

## Hook
ports/web3_hook.py is the entry web3node math can replace.
Current implementation calls layer_score.

## Goldend
ports/goldend_layers.py records the note and writes ~/.juniorhome/os/goldend_layers.json.

Additive. MIT.
