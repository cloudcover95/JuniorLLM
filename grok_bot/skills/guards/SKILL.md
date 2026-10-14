---
name: guards
description: Shared safety guards for every JuniorCloud Grok bot (overnight roster, overnight build, quant team, 06:00 hw-to-sw job). Load before any repo work.
---
# Guards (all bots)

1. **Additive only.** Never modify or delete existing files. Add new files only.
2. **One repo, one file, one push** per run, unless the lane's own cap says otherwise
   (e.g. JuniorCloud overnight build: 8-file cap → `--max-files 8`).
3. **Never `--no-verify`. Never force-push.** Let git hooks run; a hook failure is a stop.
4. **Stop and report on any failure** (hook, test, guard, auth). Do not retry around it.
5. **Bind 127.0.0.1 only** (never the all-interfaces address).
6. **Flags stay false:** mint, socket, layer_one, fetch, ue5_launch.
7. **No model pulls** (no .gguf/.safetensors/.bin, no downloads of weights).
8. **Protected repos — never commit:** JuniorSolana, cloudcover95/cloudcover95. No `.github/` changes.
9. **Read-only bots never commit or push.**
10. **Contact no one** (no email, Slack, issues, PR comments, posts) unless the user explicitly asked.
11. **No 12h or 6h skip**: run your scheduled slot; don't skip or reschedule cycles.
12. No secrets in files (ghp_, gho_, github_pat_, AKIA, private keys).

## Before every push
From the repo root, after staging:

    python3 grok_bot/guard_preflight.py [--max-files N] [--rate-check]

Exit 0 → push. Exit 1 → **do not push**; stop and report the FAIL lines verbatim.
WARN lines (upstream drift, rate literal ≠ 0.05) must be mentioned in the report.
The script is read-only; it never edits, stages, or pushes. Selftest: `--selftest`.
