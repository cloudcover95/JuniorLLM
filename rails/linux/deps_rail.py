"""Linux rail: deps probe with an always-written status. Loopback only.

Probes the local deps the Linux rails need (python, numpy, a C compiler, make,
the native bit-plane .so, a JuniorHome checkout). If a JuniorHome deps probe
script is present locally, it runs that too. The Home probe path comes from
$JUNIORHOME_DEPS_PROBE, else the first ~/JuniorHome/scripts/*deps*.py match.
Never downloads or installs anything. No bench. No LUT. No GGUF. No model pull.
Status: ~/.juniorhome/os/status/deps.json
Usage: python3 -m rails.linux.deps_rail
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rails.linux import local_status  # noqa: E402

HOME = Path.home() / "JuniorHome"
STATUS = local_status.STATUS.parent / "status" / "deps.json"


def _home_probe() -> Path | None:
    env = os.environ.get("JUNIORHOME_DEPS_PROBE")
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    scripts = HOME / "scripts"
    if not scripts.is_dir():
        return None
    hits = sorted(scripts.glob("*deps*.py"))
    return hits[0] if hits else None


def local_probe() -> dict[str, Any]:
    numpy = importlib.util.find_spec("numpy") is not None
    so = Path(__file__).with_name("libjunior_bitplane.so").is_file()
    return {
        "python": platform.python_version(),
        "numpy": numpy,
        "cc": shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"),
        "make": shutil.which("make"),
        "native_so": so,
        "juniorhome": HOME.is_dir(),
    }


def run() -> dict[str, Any]:
    local = local_probe()
    probe = _home_probe()
    if probe is None:
        home = {"present": False, "detail": "no JuniorHome deps probe found locally"}
    else:
        try:
            out = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=60)
            home = {"present": True, "script": str(probe), "ok": out.returncode == 0,
                    "stdout": out.stdout.strip()[-4000:], "stderr": out.stderr.strip()[-2000:]}
        except Exception as e:
            home = {"present": True, "script": str(probe), "ok": False, "error": f"{type(e).__name__}: {e}"}
    required_ok = local["numpy"]  # the stack rails need numpy; the rest fall back
    ok = required_ok and (home.get("ok", True))
    body = {
        "rail": "deps",
        "local_deps": local,
        "home_probe": home,
        "ok": ok,
        "ts": datetime.now(timezone.utc).isoformat(),
        "bind": "127.0.0.1",
        "install": False,
        "model_pull": False,
        "local": True,
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    body["path"] = str(STATUS)
    return body


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["ok"] else 1)
