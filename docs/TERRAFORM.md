# OSai terraform and ternary efficiency

Local only. No model pull.

## Terraform
ports/osai_terraform.py reads the Home inference tile and the layer score.
If the note is already T0 with high combined score, it keeps. Otherwise handoff is repack.
Writes ~/.juniorhome/os/osai_terraform.json.

## Efficiency
ports/ternary_efficiency.py compares trit pack cost to a naive slice.
Writes ~/.juniorhome/os/ternary_efficiency.json.

Additive. MIT.
