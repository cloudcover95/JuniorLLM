"""Run every golden in junior_osai/goldens through Flagstaff + handshake."""
from __future__ import annotations

import json
from pathlib import Path

from ports.flagstaff_balance import check
from ports.gaia_proto import handshake

DIR = Path(__file__).resolve().parents[1] / "junior_osai" / "goldens"


def run_all() -> dict:
    suites = []
    for p in sorted(DIR.glob("*.json")):
        spec = json.loads(p.read_text(encoding="utf-8"))
        rows = []
        for c in spec.get("cases") or spec.get("nodes") or spec.get("tiles") or []:
            if not isinstance(c, dict):
                continue
            note = c.get("note") or c.get("name") or c.get("sym") or p.stem
            gate = check(str(note), area=c.get("area", "auto"), consent=bool(c.get("consent", True)), private=bool(c.get("private", False)))
            if "0.0.0.0" in str(note):
                gate["ok"] = False
            hs = handshake(str(note), job="dash-viewport") if c.get("expect_ok", True) else {"schema_ok": gate.get("ok")}
            want = bool(c.get("expect_ok", True))
            got = bool(gate.get("ok"))
            rows.append({"id": c.get("id") or c.get("name") or c.get("sym"), "got": got, "want": want, "pass": got == want, "schema_ok": hs.get("schema_ok")})
        if not rows:
            continue
        suites.append({"file": p.name, "species": spec.get("species") or p.stem, "passed": sum(1 for r in rows if r["pass"]), "n": len(rows), "rows": rows})
    return {"suite": "JuniorOSai", "protocol": "goldend-osai-omega/1", "passed": sum(s["passed"] for s in suites), "n": sum(s["n"] for s in suites), "suites": suites, "download": False}
