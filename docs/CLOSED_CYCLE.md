# Closed cycle

Local only. No model pull.

## Cycle
ports/closed_cycle.py runs the Home full stack rail then the OSai handoff.
Writes ~/.juniorhome/os/closed_cycle.json.

## Bench
ports/cycle_bench.py times the cycle and writes ~/.juniorhome/os/cycle_bench.json.

Home rails can surface keep/handoff and milliseconds per cycle.

Additive. MIT.
