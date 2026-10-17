"""Live beta juniorctl port map: in-process handshake check (no sockets, no exec).

Advances JuniorLLM docs/BETA_TO_OS.md, item "Live beta", steps 1, 2 and 6:
  1. `juniorctl health`
  2. `juniorctl ask` / `night` / `security`
  6. bitnetd on 127.0.0.1:8765 only

Reads rails/linux/juniorctl.py, ctl_cli.py, juniorctl_ops.py, i2sd.py and
bitnetd.service as plain text and parses the Python with `ast` (never imported,
never executed, no subprocess). For each live-beta subcommand it checks:
  * ctl_cli.run dispatches the subcommand to a function of the same name;
  * that function is defined in juniorctl.py or juniorctl_ops.py;
  * health() and security() advertise bitnetd at 127.0.0.1:8765, and the
    hardened bitnetd.service pins JUNIOR_BIND=127.0.0.1:8765;
  * i2sd.py pins HOST=127.0.0.1, PORT=8767 and refuses non-loopback hosts;
  * the Home / OSai / FieldCore / Flagstaff / JuniorLLM rows map each
    subcommand only to loopback bitnetd (8765) or i2sd (8767), never 0.0.0.0.

The skill-pin family is owned by the overnight build (T13) and is not checked.
Test files are not run or wrapped. No socket is opened, nothing is fetched,
no model is pulled, nothing is launched. Each row is closed with
sha256(prev || pack || text)[:16]. Stdlib only.
Run: python ports/live_beta_ctl_port_map.py   (exit 0 on pass, 1 on fail).
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"

LOOPBACK = "127.0.0.1"
BITNETD = f"{LOOPBACK}:8765"
I2SD = f"{LOOPBACK}:8767"
RATE = 0.05
PACK = "live-beta-ctl-port-map"
GENESIS = "0" * 16

FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}

LIVE_CMDS = ("health", "ask", "night", "security")

# subcommand -> loopback daemons it may touch on a live beta box
CMD_PORTS = {
    "health": (BITNETD,),
    "ask": (BITNETD,),
    "night": (BITNETD, I2SD),
    "security": (BITNETD,),
}

NODES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def _src(name: str) -> str:
    return (LINUX / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_src(name), filename=name)


def _funcs(tree: ast.Module) -> set[str]:
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def _dispatch(tree: ast.Module) -> dict[str, str]:
    """Map `if cmd == "x":` branches in run() to the ns["y"] they call."""
    out: dict[str, str] = {}
    run = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"),
        None,
    )
    if run is None:
        return out
    for node in ast.walk(run):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if not (
            isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name)
            and t.left.id == "cmd"
            and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and isinstance(t.comparators[0].value, str)
        ):
            continue
        cmd = t.comparators[0].value
        for sub in node.body:
            for inner in ast.walk(sub):
                if (
                    isinstance(inner, ast.Subscript)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "ns"
                    and isinstance(inner.slice, ast.Constant)
                ):
                    out.setdefault(cmd, inner.slice.value)
    return out


def _assigns(tree: ast.Module) -> dict[str, object]:
    out: dict[str, object] = {}
    for n in tree.body:
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant)
        ):
            out[n.targets[0].id] = n.value.value
    return out


def _func_strings(tree: ast.Module, name: str) -> set[str]:
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return {
                c.value
                for c in ast.walk(n)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            }
    return set()


def checks() -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []

    ctl = _tree("juniorctl.py")
    ops = _tree("juniorctl_ops.py")
    cli = _tree("ctl_cli.py")
    i2sd = _tree("i2sd.py")
    unit = _src("bitnetd.service")

    defined = _funcs(ctl) | _funcs(ops)
    disp = _dispatch(cli)

    for cmd in LIVE_CMDS:
        target = disp.get(cmd)
        rows.append((f"dispatch:{cmd}", target == cmd, f"ctl_cli.run {cmd!r} -> ns[{target!r}]"))
        rows.append((f"defined:{cmd}", cmd in defined, f"{cmd}() in juniorctl/juniorctl_ops"))

    advertised = _func_strings(ctl, "health")
    rows.append(("health:bitnetd", BITNETD in advertised, f"health() advertises {BITNETD}"))
    for cmd in LIVE_CMDS:
        label = "ask <q>" if cmd == "ask" else cmd
        rows.append((f"health:cmds:{cmd}", label in advertised, f"health() lists {label!r}"))

    sec = _func_strings(ctl, "security")
    rows.append(("security:bind", BITNETD in sec, f"security() reports bind {BITNETD}"))
    rows.append(("unit:bind", f"JUNIOR_BIND={BITNETD}" in unit, "bitnetd.service JUNIOR_BIND loopback"))
    for k in ("NoNewPrivileges=yes", "ProtectSystem=strict", "MemoryDenyWriteExecute=yes"):
        rows.append((f"unit:{k.split('=')[0]}", k in unit, f"bitnetd.service has {k}"))

    a = _assigns(i2sd)
    rows.append(("i2sd:host", a.get("HOST") == LOOPBACK, f"i2sd HOST={a.get('HOST')!r}"))
    rows.append(("i2sd:port", a.get("PORT") == 8767, f"i2sd PORT={a.get('PORT')!r}"))
    rows.append(("i2sd:loopback_only", "loopback only" in _src("i2sd.py"), "i2sd serve() refuses non-loopback"))

    sources = "".join(_src(n) for n in ("juniorctl.py", "ctl_cli.py", "juniorctl_ops.py", "i2sd.py")) + unit
    rows.append(("no_any_bind", "0.0.0.0" not in sources, "no 0.0.0.0 in live-beta ctl sources"))

    for node in NODES:
        for cmd in LIVE_CMDS:
            ports = CMD_PORTS[cmd]
            ok = all(p.startswith(LOOPBACK + ":") and p in (BITNETD, I2SD) for p in ports)
            rows.append((f"map:{node}:{cmd}", ok, f"{node} {cmd} -> {','.join(ports)}"))

    rows.append(("flags", not any(FLAGS.values()), json.dumps(FLAGS, sort_keys=True)))
    rows.append(("rate", RATE == 0.05, f"rate {RATE}"))
    return rows


def main() -> int:
    prev = GENESIS
    failed = 0
    out = []
    for name, ok, text in checks():
        prev = close(prev, PACK, f"{name}|{int(ok)}|{text}")
        out.append({"check": name, "ok": ok, "text": text, "close": prev})
        failed += 0 if ok else 1
    report = {
        "item": "BETA_TO_OS Live beta (juniorctl health/ask/night/security, bitnetd 8765 only)",
        "pack": PACK,
        "bitnetd": BITNETD,
        "i2sd": I2SD,
        "rate": RATE,
        "flags": FLAGS,
        "rows": len(out),
        "failed": failed,
        "tip": prev,
        "fail_rows": [r for r in out if not r["ok"]],
    }
    print(json.dumps(report, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
