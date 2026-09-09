# Grok Bot controls for JuniorCloud / cloudcover95

Use **Grok Automations** (grok.com/automations or the Grok iOS/Android app) and **Grok Bot** (computer-use, paid) as the *cloud overnight worker*.
Local inference stays on Junior custom LLMs. The bot is not the model of record.

## One task

Complete building all scripting, folders, and files of the JuniorCloud ecosystem toward a **complete local software stack**, with a roadmap to a **custom BitNet Linux OS**.

Everything else is a slice of that task.

## Where the bot may write

Primary: `cloudcover95/JuniorLLM`  
Then (additive only): JuniorHome, JuniorClimbs, AGI_SDK, JuniorPiPython, BitNet-mlx, JuniorCoach, JuniorStock, JuniorAGI_SDK.

**Never delete** existing files or repos. Additive commits only. Branch `main` unless told otherwise.

## Hard rails (same as JuniorGuardrail)

1. No private-land boulder publish without owner consent.
2. No `eval` / `exec` / raw `socket` in skills.
3. No GPT-6 Astra API wrapper (closed). JuniorAstra is the open runtime port.
4. No 1.5 TB Kimi pull. Cap edge ports.
5. Bind local daemons to `127.0.0.1` only.
6. Memory topics: health/beliefs off unless `include_sensitive=true`.
7. Fable SafetyClassifier first. Refuse/fallback honored.
8. If Enhanced TDA or NightTernary says `low_confidence`, stop and write a receipt — do not keep mutating.

## How to run the bot

### A. Grok Automations (every night)

1. Open grok.com/automations (or Grok app → Automations).
2. Name: `JuniorCloud overnight build`.
3. Paste `grok_bot/PROMPT.md` as the instruction.
4. Attach / @ mention GitHub connector (`cloudcover95`).
5. Schedule: daily 02:00 America/Denver.
6. Notification: app + email.
7. **Run now** once and read the receipt.

Each run is a *fresh* request with the same instructions and current GitHub state. That is the intended behavior.

### B. Grok Bot computer-use (heavy slices)

Use Bot only for multi-file scaffolding that Automations cannot finish in one shot:
- new package trees under `adaptations/` or `rails/linux/`
- running tests after a push
- reading `MORNING.json` / loop state

Do **not** use Bot to browse random vendor dashboards or to log into third-party guidebooks.

### C. Chat (you, daytime)

Point Grok at `grok_bot/PROMPT.md` + `ports/registry.py` and say which slice (Gemma loader, Astra Work, StoneField, Linux rail).

## Custom LLM routing (inside JuniorLLM)

| Job the bot is doing | Port |
|----------------------|------|
| Field / access / gym conditions | JuniorBitNetFieldCore |
| Interactive chat / general code help | JuniorGemma4-4B |
| Safety, refuse, rigidity | JuniorFable |
| Durable overnight Work, checkpoints | JuniorAstra |
| Sparse long-context (if weights fit) | JuniorKimiK3-edge |
| Night drift / morning brief | JuniorNightTernary |

`ports.registry.pick(task, ram_gb)` is the machine router. Grok Bot must name the port it used in the commit body.

## Done definition for a night

A night is successful if it produces **at least one additive commit** *or* a written `blocked` receipt with reasons — never a silent no-op and never a deletion.
