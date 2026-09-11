# JuniorLLM

Local-first custom ports + ternary engines for JuniorCloud LLC.
Not a 1.5TB download. Not a generic chat wrapper.

## Ports

`ports/registry.py` + `ports/layer_mgr.py`

| Port | Job |
|------|-----|
| JuniorBitNetFieldCore | Crowd/field trit scorer (default, sparse/coexist) |
| JuniorBitNetDraft | CAD sidecar; compile gate before IQ |
| JuniorFable | Safety classifier |
| JuniorAstra | Durable work runtime |
| JuniorAstraReason | Q4 + rigid IQ loops (dense phase) |
| JuniorGemma4-4B / Qwen-local / Kimi-edge | High-quant caps |

Layer manager: Teqp **phase** picks the port. `cad`/`fable`/`astra` keywords still win.

## JuniorTeqp

REFPROP/CoolProp *shape* on trit states (`junior_bitnet/`):

- `teqp.py` — A_id + A_res, reduced T/ρ
- `refprop.py` — named fluids NIGHT FIELD ABSMEAN SPARSE DENSE ZK
- `palace.py` — pull is a copy; SIS commit does not move
- `coolstore.py` — cold JSON table, no `z`
- `vault.py` / `ledger.py` — Obsidian note + public observations

## One script

```bash
PYTHONPATH=. python scripts/home_sync.py ./vault
PYTHONPATH=. python scripts/layer_prod.py
PYTHONPATH=. python scripts/prove_bitnet.py
```

`home_sync` writes `JuniorTeqp/property_table.md`, `observations.jsonl`, `coolstore.json`.

## Prove

`junior_bitnet.prove()` — alphabet, compile-blocks-IQ, BitLinear, night, zk pack, palace isolation, catalog sealed.
Does **not** prove Microsoft BitNet-2B or NIST REFPROP.

## Sisters

- BitNet-mlx — Apple Silicon kernels / vision quant
- JuniorMemSys-Suite — optional SIS backend
- JuniorEngrTools — desk + Obsidian port
- JuniorHome — pointers only (`docs/JUNIOR_TEQP.md`)
- JuniorOmega — CAD solids after compile is ready
