"""Always-on local status. Core workflows never go missing."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

STATUS = Path.home() / ".juniorhome" / "os" / "local_status.json"

def write(body: dict[str, Any]) -> Path:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return STATUS

def missing(what: str, detail: str = "") -> dict[str, Any]:
    body = {"what": what, "present": False, "detail": detail, "local": True}
    write(body)
    return body

if __name__ == "__main__":
    print(missing("test"))
