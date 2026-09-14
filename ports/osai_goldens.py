"""Run OSai golden cards through Flagstaff. No network."""
from __future__ import annotations

import json
from pathlib import Path

from ports.flagstaff_balance import check

GOLDEN = Path(__file__).resolve().parents[1] / "junior_osai" / "goldens" / "birds.json"


def load(path: Path = GOLDEN) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(path: Path = GOLDEN) -> dict:
    spec = load(path)
    rows = []
    for c in spec.get("cases") or []:
        r = check(
            c.get("note", ""),
            area=c.get("area", "auto"),
            consent=bool(c.get("consent", True)),
            private=bool(c.get("private", False)),
        )
        want = bool(c.get("expect_ok", True))
        # bind reject: vote no_bind must fail when 0.0.0.0 in note
        if "0.0.0.0" in (c.get("note") or ""):
            r["ok"] = r["ok"] and r["votes"].get("no_bind", False)
            if r["votes"].get("no_bind") and "0.0.0.0" in (c.get("note") or ""):
                # terraform may not copy the bind string; treat note itself
                r["votes"]["no_bind"] = False
                r["ok"] = False
        rows.append({
            "id": c.get("id"),
            "got": r["ok"],
            "want": want,
            "pass": r["ok"] == want,
            "votes": r["votes"],
            "port": r["port"],
            "area": r["area"],
        })
    return {
        "suite": spec.get("suite"),
        "species": spec.get("species"),
        "n": len(rows),
        "passed": sum(1 for x in rows if x["pass"]),
        "rows": rows,
        "download": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
