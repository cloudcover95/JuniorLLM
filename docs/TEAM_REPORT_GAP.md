# Team report gap

rails/linux/team_report_rail.py calls scripts/team_report_prod.py.
That script now lives in JuniorLLM and delegates to JuniorHome if present.
If Home is not checked out, it writes a local stub and exits 0.

## Status
Live. Additive. Local only.
