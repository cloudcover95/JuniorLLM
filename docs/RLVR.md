# RLVR for LLM and OSai agents

Local verifiable reward cycle. No external trainer. No model pull.

## Cycle
1. Agent runs one or more checks that return pass/fail.
2. Mean score and note length select an envelope (T0 / T1 / T2).
3. Envelope records max tokens, max ms, and hardware.
4. Optional handoff to the next agent or layer.
5. Cycle appends to ~/.juniorhome/os/rlvr.jsonl.

## Trit pack scoring
ports/trit_score.py packs a note to {-1,0,1} and scores density and stability.
ports/rlvr_trit.py adds that score as a verifiable reward.
A low trit score tightens the envelope and sets handoff to repack.
Log: ~/.juniorhome/os/rlvr_trit.jsonl.

## Hardware
ports/hardware_probe.py writes ~/.juniorhome/os/hardware_probe.json.
Values: cpu, metal, cuda, asahi.

## Use
```python
from ports.rlvr_trit import cycle_with_trit
from ports.hardware_probe import probe

hw = probe()["hardware"]
result = cycle_with_trit("status note", hardware=hw, next_agent="osai")
```

Additive. Local only.
