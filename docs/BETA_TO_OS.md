# End objective

Live production-grade **beta** of the JuniorCloud local suite, then a **JuniorOS overlay** (not a custom kernel) whose PID-adjacent service is BitNet.

Owner: cloudcover95. TZ: America/Denver.
Worker: Grok Automations `JuniorCloud overnight build` + this repo. Not a random Grok Bot device-login.

## Definition of live beta (suite)

A van / Pi / M4 clone can:

1. `PYTHONPATH=. python rails/linux/juniorctl.py health`
2. `juniorctl ask "flagstaff late summer?"` — public seeds + covenant, no MP scrape
3. `python tests/test_junior_aie.py` + astra + astra_reason + grok_bot_controls pass
4. JuniorClimbs `GET /stonefield/health` + `/stonefield/terms` if FastAPI is installed; engines pass without FastAPI
5. Ports listed: FieldCore, BitNet-2B4T, Gemma4, Fable, Astra, AstraReason, Kimi-edge, Qwen-local, NightTernary
6. No file deleted. No >8GB pull. bitnetd advertised only on `127.0.0.1:8765`

## Definition of JuniorOS overlay (later)

Debian/Alpine userspace + `os-release.junior` + `bitnetd.service` + `juniorctl` on PATH. Real I2_S 2B4T when the GGUF is **already on disk**. No OSWorld score. No kernel blob.

## Ordered backlog (one slice per night)

1. Keep tests green; fill missing `__init__.py` only if import breaks.
2. `juniorctl night` — wrap `bitnet_night.cycle` and write `grok_bot/LAST_RECEIPT.md`.
3. Document on-device Gemma Q4 / BitNet-2B4T load path in `ports/ON_DEVICE.md` (no download).
4. Wire `junior_aie.ask` default corpus on every `juniorctl ask` (done) + add one gym_internal note path without leaking it to public ask.
5. `rails/linux/install-overlay.sh` — copies units + os-release, bind loopback.
6. JuniorClimbs beta probe stays stdlib-first.
7. Stop. Receipt. Next line of this list.
