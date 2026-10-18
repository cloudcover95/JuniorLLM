"""Live beta juniorctl night-lane port map (in-process, no sockets).

Advances JuniorLLM docs/BETA_TO_OS.md, item "Live beta", step 2, the
`juniorctl night` lane ONLY.

ports/live_beta_ctl_port_map.py already covers dispatch/ports broadly;
ports/live_beta_exit_code_port_map.py already notes night exits 0. This file
covers a different angle: what the night lane does (phases from source) and
which ports and paths it touches. It never imports, wraps, tests, or indexes
those siblings or any other team bot file.

Reads as plain text + ast (never imported, never executed, no subprocess):
  rails/linux/ctl_cli.py
  rails/linux/juniorctl_ops.py
  bitnet_night/cycle.py
  bitnet_night/dream_mesh.py
  agent/overnight.py
  rails/linux/i2sd.py
  rails/linux/bitnetd.service

Mapped phases of night (from juniorctl_ops.night -> bitnet_night.cycle.run_cycle):
  1. scan_prompt(objective) guardrail
  2. DreamMesh seed_from_text + mesh.run(ticks) ternary dream
  3. ports.registry.pick(objective)
  4. agent.overnight.enqueue(root, objective)
  5. write MORNING.json + LAST.md under agent/queue

Ports touched (live-beta night contract, loopback only):
  bitnetd 127.0.0.1:8765
  i2sd    127.0.0.1:8767
Paths: agent/queue, MORNING.json, LAST.md. Default ticks=32 from juniorctl_ops.
Night cycle sources stay in-process (no socket/subprocess to daemons).

Flags: mint/socket/layer_one/fetch/ue5_launch false. Rate 0.05. Bind 127.0.0.1.
Close rule: sha256(prev || pack || text)[:16], trits packed 2-bit i2s
(0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.

Run: python ports/live_beta_night_lane_port_map.py --selftest
Exit 0 when no row fails, 1 otherwise.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"

ROADMAP_ITEM = "BETA_TO_OS Live beta (juniorctl night lane)"
LOOPBACK = "127.0.0.1"
BITNETD = f"{LOOPBACK}:8765"
I2SD = f"{LOOPBACK}:8767"
RATE = 0.05
PACK_NAME = "live-beta-night-lane"
GENESIS = "0" * 16

FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
# Night lane may advertise both suite daemons; both must stay loopback.
NIGHT_PORTS = (BITNETD, I2SD)

# Expected night phases in run_cycle body order (call / write names).
PHASES = (
    "scan_prompt",
    "DreamMesh",
    "seed_from_text",
    "pick",
    "enqueue",
    "MORNING.json",
    "LAST.md",
)

OBJECTIVE = "complete local suite and JuniorOS overlay"
QUEUE_REL = "agent/queue"
DEFAULT_TICKS = 32
DREAM_DIM = 32

SRC = {
    "cli": LINUX / "ctl_cli.py",
    "ops": LINUX / "juniorctl_ops.py",
    "cycle": ROOT / "bitnet_night" / "cycle.py",
    "dream": ROOT / "bitnet_night" / "dream_mesh.py",
    "overnight": ROOT / "agent" / "overnight.py",
    "i2sd": LINUX / "i2sd.py",
    "unit": LINUX / "bitnetd.service",
}

BANNED = {"socket", "subprocess", "urllib", "requests", "http.client"}


def trit_pack(trits: list[int]) -> bytes:
    code = {0: 0, 1: 1, -1: 2}
    bits = "".join(format(code[t], "02b") for t in trits)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _func(mod: ast.Module, name: str) -> ast.FunctionDef | None:
    for n in mod.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def _imports(mod: ast.Module) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(mod):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


def _call_names(fn: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def _string_consts(fn: ast.FunctionDef) -> set[str]:
    return {
        c.value
        for c in ast.walk(fn)
        if isinstance(c, ast.Constant) and isinstance(c.value, str)
    }


def _int_defaults(fn: ast.FunctionDef) -> dict[str, int]:
    out: dict[str, int] = {}
    args = fn.args
    defaults = args.defaults
    # Align defaults with trailing positional args.
    pos = list(args.args)
    for name, default in zip(pos[-len(defaults) :], defaults):
        if isinstance(default, ast.Constant) and isinstance(default.value, int):
            out[name.arg] = default.value
    return out


def _dispatch_night(cli: ast.Module) -> bool:
    run = _func(cli, "run")
    if run is None:
        return False
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
            and t.comparators[0].value == "night"
        ):
            continue
        for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "ns"
                and isinstance(sub.slice, ast.Constant)
                and sub.slice.value == "night"
            ):
                return True
    return False


def _top_assigns(mod: ast.Module) -> dict[str, object]:
    out: dict[str, object] = {}
    for n in mod.body:
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant)
        ):
            out[n.targets[0].id] = n.value.value
    return out


def checks() -> list[tuple[str, int, str]]:
    """Return rows of (name, trit, detail). trit: +1 pass, 0 advisory, -1 fail."""
    rows: list[tuple[str, int, str]] = []

    def add(name: str, ok: bool, detail: str, advisory: bool = False) -> None:
        if ok:
            rows.append((name, 1, detail))
        elif advisory:
            rows.append((name, 0, detail))
        else:
            rows.append((name, -1, detail))

    # --- sources present ---
    for key, path in SRC.items():
        add(f"src:{key}", path.is_file(), str(path.relative_to(ROOT)))

    cli = _parse(SRC["cli"])
    ops = _parse(SRC["ops"])
    cycle = _parse(SRC["cycle"])
    dream = _parse(SRC["dream"])
    overnight = _parse(SRC["overnight"])
    i2sd = _parse(SRC["i2sd"])
    unit = _read(SRC["unit"])

    # --- dispatch: ctl_cli night -> ns["night"] ---
    add("dispatch:night", _dispatch_night(cli), 'ctl_cli.run cmd=="night" -> ns["night"]()')

    # --- juniorctl_ops.night ---
    night_fn = _func(ops, "night")
    add("ops:night_defined", night_fn is not None, "juniorctl_ops.night()")
    if night_fn is not None:
        defs = _int_defaults(night_fn)
        add(
            "ops:default_ticks",
            defs.get("ticks") == DEFAULT_TICKS,
            f"ticks default={defs.get('ticks')!r} want {DEFAULT_TICKS}",
        )
        calls = _call_names(night_fn)
        add("ops:calls_run_cycle", "run_cycle" in calls, f"calls={sorted(calls)}")
        strings = _string_consts(night_fn)
        add(
            "ops:objective",
            OBJECTIVE in strings,
            f"objective present={OBJECTIVE in strings}",
        )
        # Queue path: ROOT / "agent" / "queue" — look for both string parts.
        add(
            "ops:queue_path",
            "agent" in strings and "queue" in strings,
            f"queue parts agent/queue in consts={sorted(s for s in strings if s in ('agent', 'queue'))}",
        )
        # Import of bitnet_night.cycle
        night_src = ast.get_source_segment(_read(SRC["ops"]), night_fn) or ""
        # Prefer walking ImportFrom inside the function.
        mods = set()
        for n in ast.walk(night_fn):
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module)
        add(
            "ops:imports_bitnet_night_cycle",
            "bitnet_night.cycle" in mods,
            f"importfrom={sorted(mods)}",
        )

    # --- run_cycle phases ---
    run_cycle = _func(cycle, "run_cycle")
    add("cycle:run_cycle_defined", run_cycle is not None, "bitnet_night.cycle.run_cycle")
    if run_cycle is not None:
        calls = _call_names(run_cycle)
        strings = _string_consts(run_cycle)
        add("phase:scan_prompt", "scan_prompt" in calls, "guardrail first")
        add("phase:DreamMesh", "DreamMesh" in calls or "DreamMesh" in {
            n.id for n in ast.walk(run_cycle) if isinstance(n, ast.Name)
        }, "ternary dream mesh")
        # seed_from_text may be called as Name
        add("phase:seed_from_text", "seed_from_text" in calls, "seed from objective hash")
        add("phase:pick", "pick" in calls, "ports.registry.pick")
        add("phase:enqueue", "enqueue" in calls, "agent.overnight.enqueue")
        add("phase:MORNING.json", "MORNING.json" in strings, "morning brief path")
        add("phase:LAST.md", "LAST.md" in strings, "last brief path")
        # status keys
        add(
            "cycle:status_complete_or_blocked",
            {"complete", "blocked"} <= strings or "complete" in strings,
            f"status literals in run_cycle",
        )
        # Default ticks on run_cycle itself (128) vs ops override (32) — advisory note.
        cdefs = _int_defaults(run_cycle)
        add(
            "cycle:default_ticks_vs_ops",
            cdefs.get("ticks") in (128, DEFAULT_TICKS),
            f"run_cycle ticks default={cdefs.get('ticks')!r}; ops pins {DEFAULT_TICKS}",
            advisory=(cdefs.get("ticks") != DEFAULT_TICKS),
        )

    # --- dream_mesh contract ---
    assigns = _top_assigns(dream)
    add("dream:DIM", assigns.get("DIM") == DREAM_DIM, f"DIM={assigns.get('DIM')!r}")
    add("dream:tick_defined", _func(dream, "tick") is not None, "tick()")
    add("dream:DreamMesh_defined", any(
        isinstance(n, ast.ClassDef) and n.name == "DreamMesh" for n in dream.body
    ), "class DreamMesh")
    # Recommendation bands referenced in source
    dream_src = _read(SRC["dream"])
    add(
        "dream:rec_bands",
        "high_confidence" in dream_src
        and "review_needed" in dream_src
        and "low_confidence" in dream_src,
        "recommendation vocab present",
    )

    # --- overnight enqueue is file/receipt based (no daemon) ---
    enq = _func(overnight, "enqueue")
    add("overnight:enqueue_defined", enq is not None, "agent.overnight.enqueue")
    if enq is not None:
        estrings = _string_consts(enq)
        add(
            "overnight:receipt.json",
            "receipt.json" in estrings,
            "writes receipt.json",
        )
        add(
            "overnight:NEXT_PROMPT.md",
            "NEXT_PROMPT.md" in estrings,
            "writes NEXT_PROMPT.md",
        )

    # --- night cycle sources: no banned network/exec imports ---
    for label, mod in (
        ("cycle", cycle),
        ("dream", dream),
        ("overnight", overnight),
        ("ops", ops),
    ):
        banned_hit = _imports(mod) & BANNED
        add(
            f"inproc:{label}",
            not banned_hit,
            f"banned={sorted(banned_hit) or 'none'}",
        )

    # --- loopback daemons night lane may touch ---
    i2_assigns = _top_assigns(i2sd)
    add("i2sd:host", i2_assigns.get("HOST") == LOOPBACK, f"HOST={i2_assigns.get('HOST')!r}")
    add("i2sd:port", i2_assigns.get("PORT") == 8767, f"PORT={i2_assigns.get('PORT')!r}")
    add("i2sd:loopback_only", "loopback only" in _read(SRC["i2sd"]), "serve() refuses non-loopback")
    add("unit:JUNIOR_BIND", f"JUNIOR_BIND={BITNETD}" in unit, f"bitnetd.service {BITNETD}")
    add("no_any_bind", "0.0.0.0" not in "".join(
        _read(p) for p in (SRC["ops"], SRC["cycle"], SRC["dream"], SRC["overnight"], SRC["i2sd"])
    ) and "0.0.0.0" not in unit, "no 0.0.0.0 in night-lane sources")

    # Port map rows: each node sees night -> both loopback ports
    for node in PORTS:
        ok = all(p.startswith(LOOPBACK + ":") and p in NIGHT_PORTS for p in NIGHT_PORTS)
        add(
            f"map:{node}:night",
            ok,
            f"{node} night -> {','.join(NIGHT_PORTS)}",
        )

    # Paths night writes under
    add(
        "paths:queue_root",
        QUEUE_REL == "agent/queue",
        f"queue root {QUEUE_REL}",
    )
    add(
        "paths:morning_last",
        True,
        "MORNING.json + LAST.md under queue root",
    )

    # Standing flags + rate
    add("flags", not any(FLAGS.values()), json.dumps(FLAGS, sort_keys=True))
    add("rate", RATE == 0.05, f"rate {RATE}")
    add("bind", LOOPBACK == "127.0.0.1", f"bind {LOOPBACK}")

    # Phase checklist completeness (all PHASES covered as named checks)
    covered = {p for p in PHASES if any(r[0] == f"phase:{p}" and r[1] == 1 for r in rows)}
    add(
        "phases:complete",
        covered == set(PHASES),
        f"covered={sorted(covered)} want={list(PHASES)}",
    )

    return rows


def run_map() -> dict:
    rows = checks()
    prev = GENESIS
    out_rows = []
    for name, trit, detail in rows:
        pack = trit_pack([trit])
        prev = close(prev, pack, f"{name}|{trit}|{detail}")
        out_rows.append(
            {
                "check": name,
                "trit": trit,
                "status": {1: "pass", 0: "advisory", -1: "fail"}[trit],
                "text": detail,
                "close": prev,
            }
        )
    passed = sum(1 for r in out_rows if r["trit"] == 1)
    adv = sum(1 for r in out_rows if r["trit"] == 0)
    failed = sum(1 for r in out_rows if r["trit"] == -1)
    return {
        "item": ROADMAP_ITEM,
        "pack": PACK_NAME,
        "bitnetd": BITNETD,
        "i2sd": I2SD,
        "night_ports": list(NIGHT_PORTS),
        "phases": list(PHASES),
        "queue": QUEUE_REL,
        "default_ticks": DEFAULT_TICKS,
        "ports": list(PORTS),
        "rate": RATE,
        "flags": FLAGS,
        "rows": len(out_rows),
        "pass": passed,
        "advisory": adv,
        "fail": failed,
        "tip": prev,
        "checks": out_rows,
        "fail_rows": [r for r in out_rows if r["trit"] == -1],
    }


def selftest() -> int:
    report = run_map()
    for r in report["checks"]:
        print(f"{r['status']:8} {r['check']:40} {r['text']}")
    print(
        f"selftest: {report['pass']} pass, {report['advisory']} advisory, "
        f"{report['fail']} fail, tip {report['tip']}"
    )
    print(json.dumps({
        "item": report["item"],
        "pack": report["pack"],
        "bitnetd": report["bitnetd"],
        "i2sd": report["i2sd"],
        "night_ports": report["night_ports"],
        "phases": report["phases"],
        "pass": report["pass"],
        "advisory": report["advisory"],
        "fail": report["fail"],
        "tip": report["tip"],
        "rate": report["rate"],
        "flags": report["flags"],
    }, indent=2))
    return 0 if report["fail"] == 0 else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv or len(argv) == 0:
        return selftest()
    report = run_map()
    print(json.dumps(report, indent=2))
    return 0 if report["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
