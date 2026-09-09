# Gotchas (update when the bot fails)

- `JuniorAstra` = runtime (Work/checkpoint). `JuniorAstraReason` = Qwen/Gemma stand-in + rigid IQ. Do not merge them.
- GPT-6 Astra is closed. Do not add an API wrapper "just in case".
- `pick("astra long-horizon")` can hit the runtime; use `astra-class` / `reason` for the stand-in.
- Qwen-local cap is 8GB now, not 20GB.
- FastAPI is not on the van by default. Engine tests must stay stdlib.
- SQLite file DB can I/O-fail in sandboxes; health probes use `:memory:`.
- StoneField covenant: unknown/private tenure needs owner_consent to publish.
- Grok Automations is a *fresh* session each night — STATE is GitHub, not chat memory.
- All Grok Bots share one microVM/browser profile. Do not log into third-party climbing sites.
- NightTernary `low_confidence` means stop mutating, not "try a bigger model".
