# Flagstaff votes

`check(note)` is AND of six booleans.

| vote | true when |
|---|---|
| terraform_ok | cleaned text nonempty and terraform.ok |
| covenant | not (private and not consent) |
| area | guessed/passed area in AREAS |
| junior_port | pick_eos name starts with Junior |
| finite_y | fusion_y is finite |
| no_bind | `0.0.0.0` not in raw+cleaned |

Terraform drops `ignore previous`, `system prompt`, `0.0.0.0`, `wget `, `curl http` from the *language* copy. `no_bind` still sees the raw note so a bind attempt fails the gate.
fusion_y is a cached 16-char AbsMean-style fusion, not a model logit.
