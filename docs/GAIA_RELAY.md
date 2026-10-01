# Gaia relay

Protocol goldend-osai-omega/1.
Hops: JuniorOSai → Gaia → mesh.
Inference is a ticket. model_pull false. Bind 127.0.0.1.
Envelope t4 by default.

```bash
python3 scripts/gaia_relay_prod.py t4
```

Appends ~/.juniorhome/gaia_mesh/relay.jsonl
