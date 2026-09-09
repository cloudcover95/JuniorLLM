# Tests run on the clone, not on GitHub runners

GitHub Actions on these repos were cargo-cult Ubuntu/macOS jobs.
They are `workflow_dispatch` + `if: false` so they do not mail failures.

```
cd JuniorLLM
PYTHONPATH=. python tests/test_junior_aie.py
PYTHONPATH=. python tests/test_linux_security.py
PYTHONPATH=. python tests/test_state_schema.py
PYTHONPATH=. python scripts/backfill_check.py
```
