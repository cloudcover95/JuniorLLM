# Pack ledger and goldend status

Local only. No model pull.

## Ledger
ports/pack_ledger.py stores the packed trits and score.
Writes ~/.juniorhome/os/trit_packs.jsonl.

## Status
ports/goldend_status.py combines the layer score and latest pack.
Writes ~/.juniorhome/os/goldend_status.json.

Home rails can surface the layer and pack length without re-running the score.

Additive. MIT.
