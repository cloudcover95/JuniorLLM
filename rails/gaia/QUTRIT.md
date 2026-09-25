# Gaia training: qutrit / CMOS-ternary

This is a study rail. Home does not own a fab, a PDK, or a qutrit.

| Track | What it is | Gaia |
|-------|------------|------|
| Balanced trit | \(\{-1,0,1\}\) wire | **on** (`trit_gates`) |
| Unbalanced ternary | \(\{0,1,2\}\) STI/PTI/NTI papers | map only |
| Qutrit | 3-level quantum | off |
| CMOS ternary | multi-Vt silicon | off |

Training = append a passing vote to FieldCore jsonl. Not an imatrix. Not a tapeout.
`ports.qutrit_rail.to_balanced` exists so a paper that uses 0,1,2 can be read onto the wire.
