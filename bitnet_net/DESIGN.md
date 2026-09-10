# Local cryptocurrency net (BitNet-quant)

Not Bitcoin. Not a public L1. A **Junior node** mints and transfers integer units.

## Choices

| Topic | Choice |
|-------|--------|
| Consensus | single-operator hash chain (optional federated import) |
| Work | no PoW; block = txs + prev + optional `bitnet_pq` tick |
| Bind | 127.0.0.1 only |
| Asset | integer shares (`JUNIT`) |
| Privacy | receipts, not hidden amounts |
| PQ | `Params.secure` stays False |

Gossip reuses ForumMesh `junior-gossip-v1` shape for blocks.
