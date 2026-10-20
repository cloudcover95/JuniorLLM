# Efficient trit math

Local only. No model pull.

## Math
ports/trit_math.py caches pack, absmean, and score.
once() returns all three. Agents call this instead of recomputing.

## Path
ports/goldend_path.py uses once() and the terraform decide.
Affirms only if math passed and terraform keeps.
Writes ~/.juniorhome/os/goldend_path.json.

## OSai
ports/osai_affirm.py runs the path and updates the goldend agent.
Writes ~/.juniorhome/os/osai_affirm.json.

Additive. MIT.
