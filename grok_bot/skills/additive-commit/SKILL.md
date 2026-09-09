---
name: additive-commit
description: Write or update files on cloudcover95 repos. Use whenever creating or editing source.
---
# Additive commit

- Create new paths. Prefer new files over rewriting large existing ones.
- If updating, change only the needed region.
- Commit message: `feat: <slice>\n\nport: <JuniorPort>`
- Never `git rm`, never force-push, never amend others' commits.
- After push, fill RECEIPT `files_added` with paths.
