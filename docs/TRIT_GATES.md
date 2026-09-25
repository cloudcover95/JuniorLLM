# Ternary gates vs quantum

Home wire is balanced ternary \(\{-1,0,1\}\).
Classical gates that match the wire:

| Gate | Rule |
|------|------|
| NEG | \(-a\) |
| MIN | weak AND |
| MAX | weak OR |
| MUL | product, no carry |
| CONS | a if a=b else 0 |

Qutrit circuits (reversible comparators, STI/PTI/NTI on \(\{0,1,2\}\)) are a different alphabet and need a quantum or CMOS ternary fab. Not T0.
Flagstaff vote is already CONS-like across six checks.
