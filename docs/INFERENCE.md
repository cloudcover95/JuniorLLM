# Inference data engine

Local only. No model pull.

## Engine
ports/inference_engine.py clips a note, scores it with trit, and appends to ~/.juniorhome/os/inference_ledger.jsonl.

## Bench
ports/inference_bench.py times the record path and writes ~/.juniorhome/os/inference_bench.json.

## Goldend
ports/goldend_inference.py records a note then runs the hill-climb path.
Writes ~/.juniorhome/os/goldend_inference.json.

## Integration
- web3node Winsor or absmean can replace winsor_clip
- RLVR cycle can call record before scaling the envelope
- Home rails surface the latest trit score and ledger length

Additive. MIT.
