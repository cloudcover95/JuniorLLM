# Provenance and pure events

Local only. No model pull.

## Provenance
ports/provenance.py labels a value as fact or modelled.
Trit scores, envelope layers, and hillclimb results are modelled.
Hardware probe results are fact.

## Events
ports/events.py emits timestamped events to ~/.juniorhome/os/events.jsonl.
assert_event lets a headless script check the latest matching payload.

## Learning
- Zero-asset local-first is already the Home pattern.
- Pure event emit plus headless assert matches the bench_pipe style.
- Fact vs modelled stops an agent from treating a heuristic score as ground truth.
- One status tile driving CLI, rail, and agent is the multi-view pattern.

Additive. MIT.
