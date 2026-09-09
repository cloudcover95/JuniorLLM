# Grok Bot controls (how to point the product)

Industry layout (2026): short always-on file + on-demand skills + routine wrapper.
That is how Claude Code (`AGENTS.md`/`SKILL.md`) and Grok Bot (identity + skills + routines) are steered.

| Layer | File | When loaded |
|-------|------|-------------|
| Always-on | `/AGENTS.md` | every session |
| Gotchas | `grok_bot/GOTCHAS.md` | every session |
| Routine | `grok_bot/PROMPT.md` | pasted into Automations / Bot routine |
| Skills | `grok_bot/skills/*/SKILL.md` | only when the slice matches |
| Receipt | `grok_bot/RECEIPT.template.md` | end of every run |

## Wire-up

**Automations (already created):** `JuniorCloud overnight build` daily 02:00 America/Denver.
Replace its prompt with `PROMPT.md` if you edit this file — Automations does not auto-pull git.

**Grok Bot:** New Bot named `JuniorCloud builder`. Description = first 20 lines of `AGENTS.md`. Skill = `PROMPT.md` procedure. Routine = nightly 02:00. Tools = GitHub only.

**Grok Build / terminal:** drop `AGENTS.md` at repo root (done). Deeper files win.

## Test a routine before enabling
Confirm: inputs current, output quality, approval stops, failure states. Turn on only if all four pass.
