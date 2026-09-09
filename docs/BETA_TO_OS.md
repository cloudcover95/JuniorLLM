# End objective

Live production-grade **beta** of the JuniorCloud local suite, then a **JuniorOS overlay** (not a custom kernel) whose PID-adjacent service is BitNet.

**Work list the overnight job must not drop:** `docs/COMPILED_BACKLOG.md`
(old AGENTS checklist + TEST_ROADMAP + this file + container security).

Owner: cloudcover95. TZ: America/Denver.
Worker: Grok Automations `JuniorCloud overnight build`.

## Live beta

A fresh clone:

1. `PYTHONPATH=. python rails/linux/juniorctl.py health`
2. `juniorctl ask` / `night` / `security`
3. `python tests/test_junior_aie.py` and `tests/test_linux_security.py`
4. JuniorClimbs `scripts/stonefield_stdlib_probe.py` (no FastAPI required)
5. `python scripts/backfill_check.py` exits 0
6. No deletes. No >8GB pull. bitnetd on `127.0.0.1:8765` only

## JuniorOS overlay

Debian/Alpine + `os-release.junior` + hardened `bitnetd.service` + `juniorctl` on PATH.
I2_S 2B4T only if GGUF already on disk. No OSWorld score. No kernel blob.
