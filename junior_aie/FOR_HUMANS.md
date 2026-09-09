# If you climb or coach — how to use this

You do not need to import fifteen modules.

```bash
cd JuniorLLM
PYTHONPATH=. python rails/linux/juniorctl.py health
PYTHONPATH=. python rails/linux/juniorctl.py ask "is flagstaff usually dry in late summer?"
```

What you get back:
- a Junior port name (which local brain answered)
- a short packed brief from public seeds + your words
- `cached: true` the second time you ask the same thing

What you will **not** get:
- a Mountain Project dump
- a pin on private land
- a 1.5TB download

Gym staff: keep camp plans `gym_internal` in JuniorClimbs. This ask path only sees the public seed list plus whatever you typed.

Van / Pi: no FastAPI required for `juniorctl ask`.
