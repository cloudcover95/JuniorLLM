# RLVR for LLM and OSai agents

Local verifiable reward cycle. No external trainer. No model pull.

## Cycle
1. Agent runs one or more checks that return pass/fail.
2. Mean score and note length select an envelope (T0 / T1 / T2).
3. Envelope records max tokens, max ms, and hardware.
4. Optional handoff to the next agent or layer.
5. Cycle appends to ~/.juniorhome/os/rlvr.jsonl.

## Hardware
ports/hardware_probe.py writes ~/.juniorhome/os/hardware_probe.json.
Values: cpu, metal, cuda, asahi.

## Use
```python
from ports.rlvr import agent_cycle
from ports.hardware_probe import probe

hw = probe()["hardware"]
result = agent_cycle(
    checks=[("syntax", lambda: True), ("bounds", lambda: True)],
    note_len=128,
    hardware=hw,
    next_agent="osai",
)
```

Additive. Local only.
