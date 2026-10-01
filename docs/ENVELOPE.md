# Compute envelopes

`ports/envelope.py` is the shared cap. Not a wattmeter.

| name | watts class | notes | chars | goldens | download_gb | files |
|------|-------------|-------|-------|---------|-------------|-------|
| t4 | 12 | 8 | 256 | 4 | 0 | 4 |
| host | 45 | 16 | 1024 | 16 | 0 | 8 |
| operator | 45 | 34 | 2048 | 32 | 1.5 | 8 |

```bash
python3 scripts/envelope_prod.py t4 JuniorOS
python3 scripts/os_tick_prod.py
```

OS tick defaults to t4. Flagstaff character caps stay in ports/flagstaff.py.
