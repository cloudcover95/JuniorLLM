# Extended efficiency

Local only. No model pull.

## Bench
ports/bench_extended.py compares trit pack to a naive baseline and times the compute-smart choose path.
Writes ~/.juniorhome/os/bench_extended.json.

## Goldend
ports/goldend_efficiency.py runs the bench and the goldend agent, then writes a single efficiency report.
Writes ~/.juniorhome/os/goldend_efficiency.json.

Home rails can surface the layer and trit score without re-running the agent.

Additive. MIT.
