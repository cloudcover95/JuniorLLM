# Compute-smart path

Local only. No model pull.

## Bench
ports/bench_rlvr.py times trit pack and envelope selection.
Writes ~/.juniorhome/os/bench_rlvr.json.

## Path
ports/compute_smart.py chooses T0/T1/T2 from trit score and note length.
High score and short note stay T0.
Writes ~/.juniorhome/os/compute_smart.json.

## Goldend
ports/goldend_agent.py runs the path then the JuniorOS agent cycle.
Writes ~/.juniorhome/os/goldend_agent.json.

Additive. MIT.
