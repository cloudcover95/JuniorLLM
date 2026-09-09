---
name: maker-checker
description: Second pass after a write. Maker does the slice; checker tries to refute it.
---
# Maker / checker

Big-lab overnight workflows fan out then refute before merge.

Maker: implement the 5-bullet plan.
Checker (same run, different hat):
- Would a fresh clone import this?
- Did we delete anything? If yes, revert.
- Did Fable/guardrail patterns sneak into code comments as how-tos? Strip.
- Run the listed test command. If unavailable, mark `tests: unavailable` — do not claim pass.
- If checker disagrees, status=`blocked` or `review_needed`, do not keep stacking files.
