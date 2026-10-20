# Local hill-climb

Uses trit score as the verifiable reward. One change at a time. Held-out note catches collapse.
No model pull. No external trainer.

## Loop
ports/hillclimb.py mutates a note, scores it, keeps the change only if the score rises and the held-out note does not drop.
Writes ~/.juniorhome/os/hillclimb.jsonl.

## Goldend
ports/goldend_hillclimb.py runs compute-smart path selection then the climb.
Writes ~/.juniorhome/os/goldend_hillclimb.json.

## Integration
- web3node math (Winsor, absmean) can replace trim_mutate with a domain mutate
- JuniorLLM RLVR cycle can call climb before scaling the envelope
- Home rails surface the best score and round count

Additive. MIT.
