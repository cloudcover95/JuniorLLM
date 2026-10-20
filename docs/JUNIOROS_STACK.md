# JuniorOS production stack — agent + RLVR

Local only. No model pull.

## Agent
ports/junioros_agent.py runs one RLVR cycle with trit scoring.
Writes ~/.juniorhome/os/junioros_agent.json.

## Rails
Home reads the agent output via scripts/home_rlvr.py.
End user sees the envelope with scripts/rlvr_cli.py --show.

## Envelope
- T0: short note, high trit score
- T1: medium
- T2: long or low score
- Low trit score forces T0 and handoff repack

## Hardware
ports/hardware_probe.py sets cpu, metal, cuda, or asahi.

Additive. MIT.
