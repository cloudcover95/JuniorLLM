# OSai handoff

Local only. No model pull.

## Handoff
ports/osai_handoff.py reads ~/.juniorhome/os/home_full_stack.json.
Keeps if asserted and absmean energy >= 0.3. Otherwise handoff is repack.
Writes ~/.juniorhome/os/osai_handoff.json.

## Goldend
ports/goldend_handoff.py runs the handoff and writes ~/.juniorhome/os/goldend_handoff.json.

Additive. MIT.
