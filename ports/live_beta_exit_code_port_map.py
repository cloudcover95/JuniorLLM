"""Live beta exit-code port map: in-process handshake of juniorctl exit codes.

Advances JuniorLLM docs/BETA_TO_OS.md, item "Live beta", exit-code side:
  1. `juniorctl health`
  2. `juniorctl ask` / `night` / `security`
  3. `tests/test_linux_security.py`
  6. bitnetd on 127.0.0.1:8765 only

ports/live_beta_ctl_port_map.py already maps dispatch and ports. This file
covers only what that one leaves open: the exit-code contract. It reads
rails/linux/ctl_cli.py, rails/linux/juniorctl.py and
tests/test_linux_security.py as plain text and parses them with `ast`.
Nothing is imported from the repo, nothing is executed, no subprocess.

For each live-beta subcommand it pulls the `return` that ends its branch in
ctl_cli.run and folds it to the set of exit codes it can produce:
  health -> {0}, ask -> {0}, night -> {0}, no argument -> health -> {0},
  unknown command -> usage -> {2}, security -> {0, 1}.
For security the conditional is walked by a tiny boolean interpreter over
every unit_ok x seccomp combination (no eval), so the table is exact.
juniorctl.py must hand main()'s int to SystemExit, so the code reaches $?.

It then maps test_linux_security's three cases onto the security() keys
they assert and predicts the unittest exit (0 all pass, 1 any fail). A key a
test asserts but `juniorctl security` does not gate on is fine only when
security() returns it as a literal constant with the asserted value; a case
that reads a file juniorctl only checks for existence (seccomp defaultAction)
is reported as an advisory, with the on-disk value read as plain JSON.

Port rows: Home, OSai, FieldCore, Flagstaff and JuniorLLM each read the
exit codes of health/ask/night/security against bitnetd 127.0.0.1:8765
(health and security advertise it) and i2sd 127.0.0.1:8767 (constants in
rails/linux/i2sd.py, read as text). No socket is opened.

Skill-pin family (T13) is owned by the overnight build and is skipped.
Flags: mint false, socket false, layer_one false, fetch false,
ue5_launch false. Rate 0.05. Bind 127.0.0.1. No model pull.
Each row is one trit (+1 pass, 0 advisory, -1 fail), 2-bit packed
(0->0, 1->1, -1->2, MSB first) and closed with sha256(prev || pack || text)[:16].
Stdlib only.

Run: python ports/live_beta_exit_code_port_map.py [--selftest] [--json]
Exit 0 when no row fails, 1 otherwise.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"
TESTS = ROOT / "tests"

LOOPBACK = "127.0.0.1"
BITNETD = f"{LOOPBACK}:8765"
I2SD = f"{LOOPBACK}:8767"
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
GENESIS = "0" * 16

LIVE_BETA = ("health", "ask", "night", "security")
EXPECTED = {
    "health": {0},
    "ask": {0},
    "night": {0},
    "security": {0, 1},
    "<none>": {0},
    "<unknown>": {2},
}
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
# Which loopback lane each subcommand's exit code speaks for.
LANES = {
    "health": (BITNETD,),
    "security": (BITNETD,),
    "ask": (BITNETD, I2SD),
    "night": (BITNETD, I2SD),
}


# ---------------------------------------------------------------- close rule
def pack_trits(trits: list[int]) -> bytes:
    code = {0: 0, 1: 1, -1: 2}
    out = bytearray()
    for i in range(0, len(trits), 4):
        b = 0
        chunk = trits[i:i + 4]
        for j in range(4):
            b = (b << 2) | (code[chunk[j]] if j < len(chunk) else 0)
        out.append(b)
    return bytes(out)


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- ast helpers
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _func(mod: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _cmd_of(test: ast.expr) -> str | None:
    """Return X for `cmd == "X"` or `cmd == "X" and ...`."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        test = test.values[0]
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "cmd" and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)):
        return test.comparators[0].value
    return None


def _branch_return(body: list[ast.stmt]) -> ast.expr | None:
    for stmt in body:
        if isinstance(stmt, ast.Return):
            return stmt.value
    return None


def _default_cmd(run: ast.FunctionDef) -> str | None:
    for stmt in run.body:
        if (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "cmd" and isinstance(stmt.value, ast.IfExp)
                and isinstance(stmt.value.orelse, ast.Constant)):
            return stmt.value.orelse.value
    return None


def _eval(node: ast.expr, env: dict) -> object:
    """Tiny interpreter: Constant, IfExp, BoolOp, UnaryOp not, report["k"]."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.IfExp):
        return _eval(node.body, env) if _eval(node.test, env) else _eval(node.orelse, env)
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval(node.operand, env)
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)):
        return env[node.slice.value]
    raise ValueError(f"unsupported node {type(node).__name__}")


def _keys(node: ast.expr) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant):
            out.append(n.slice.value)
    return sorted(set(out))


def exit_table(expr: ast.expr) -> tuple[set[int], list[tuple[dict, int]]]:
    keys = _keys(expr)
    rows = []
    for combo in itertools.product((True, False), repeat=len(keys)):
        env = dict(zip(keys, combo))
        rows.append((env, int(_eval(expr, env))))
    return {code for _, code in rows}, rows


def security_literals(mod: ast.Module) -> dict:
    fn = _func(mod, "security")
    lits = {}
    if fn is None:
        return lits
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    lits[k.value] = v.value if isinstance(v, ast.Constant) else "<computed>"
    return lits


def test_cases(mod: ast.Module) -> dict:
    """Map test_* -> {"keys": {key: expectation}, "files": [...]}"""
    cases = {}
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            keys, files = {}, []
            for call in ast.walk(node):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
                    continue
                meth = call.func.attr
                if meth in ("assertTrue", "assertFalse") and call.args:
                    a = call.args[0]
                    if isinstance(a, ast.Subscript) and isinstance(a.slice, ast.Constant):
                        keys[a.slice.value] = meth == "assertTrue"
                    for n in ast.walk(a):
                        if isinstance(n, ast.Constant) and isinstance(n.value, str) and "." in n.value:
                            files.append(n.value)
                if meth == "assertEqual" and len(call.args) == 2:
                    a, b = call.args
                    if (isinstance(a, ast.Subscript) and isinstance(a.slice, ast.Constant)
                            and isinstance(b, ast.Constant)):
                        keys[a.slice.value] = b.value
            for n in ast.walk(node):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".json"):
                    files.append(n.value)
            cases[node.name] = {"keys": keys, "files": sorted(set(files))}
    return cases


def i2sd_consts(text: str) -> dict:
    mod = ast.parse(text)
    out = {}
    for stmt in mod.body:
        if (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)):
            out[stmt.targets[0].id] = stmt.value.value
    return out


# ---------------------------------------------------------------- the map
def build(ctl_cli: str, juniorctl: str, test_sec: str, i2sd: str, seccomp: str | None) -> list[dict]:
    rows: list[dict] = []

    def row(name: str, trit: int, detail: str) -> None:
        rows.append({"row": name, "trit": trit, "detail": detail})

    cli = ast.parse(ctl_cli)
    run = _func(cli, "run")
    if run is None:
        row("ctl_cli.run present", -1, "run() not found")
        return rows
    row("ctl_cli.run present", 1, "run(argv, ns) -> int")

    branches: dict[str, ast.expr | None] = {}
    for stmt in run.body:
        if isinstance(stmt, ast.If):
            c = _cmd_of(stmt.test)
            if c in LIVE_BETA:
                branches[c] = _branch_return(stmt.body)
    tail = run.body[-1]
    branches["<unknown>"] = tail.value if isinstance(tail, ast.Return) else None
    default = _default_cmd(run)
    if default in branches:
        branches["<none>"] = branches[default]
    row("no-argument default", 1 if default == "health" else -1, f"cmd defaults to {default!r}")

    sec_rows: list[tuple[dict, int]] = []
    sec_keys: list[str] = []
    for cmd in (*LIVE_BETA, "<none>", "<unknown>"):
        expr = branches.get(cmd)
        if expr is None:
            row(f"exit {cmd}", -1, "no return found in branch")
            continue
        try:
            codes, table = exit_table(expr)
        except (ValueError, KeyError) as e:
            row(f"exit {cmd}", -1, f"cannot fold: {e}")
            continue
        ok = codes == EXPECTED[cmd]
        if cmd == "security":
            sec_rows, sec_keys = table, _keys(expr)
        row(f"exit {cmd}", 1 if ok else -1, f"codes {sorted(codes)} expected {sorted(EXPECTED[cmd])}")

    if sec_rows:
        only_zero = [env for env, code in sec_rows if code == 0]
        exact = len(only_zero) == 1 and all(only_zero[0].values())
        row("security truth table", 1 if exact else -1,
            "; ".join(f"{','.join(f'{k}={int(v)}' for k, v in env.items())}->{c}" for env, c in sec_rows))

    jc = ast.parse(juniorctl)
    hands_off = any(
        isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
        and isinstance(n.exc.func, ast.Name) and n.exc.func.id == "SystemExit"
        and n.exc.args and isinstance(n.exc.args[0], ast.Call)
        and isinstance(n.exc.args[0].func, ast.Name) and n.exc.args[0].func.id == "main"
        for n in ast.walk(jc))
    row("SystemExit(main(argv))", 1 if hands_off else -1, "exit code reaches the shell" if hands_off else "missing")

    lits = security_literals(jc)
    bind_ok = lits.get("bind") == BITNETD
    row("security bind", 1 if bind_ok else -1, f"bind {lits.get('bind')!r}")

    tmod = ast.parse(test_sec)
    cases = test_cases(tmod)
    uses_main = "unittest.main" in test_sec
    row("test_linux_security exit", 1 if uses_main and cases else -1,
        f"unittest.main -> 0 all pass / 1 any fail; {len(cases)} cases")

    for name, info in sorted(cases.items()):
        gated = [k for k in info["keys"] if k in sec_keys]
        ungated = {k: v for k, v in info["keys"].items() if k not in sec_keys}
        bad = [k for k, v in ungated.items()
               if k != "defaultAction" and not (isinstance(lits.get(k), bool) and lits.get(k) == v)]
        trit, notes = 1, []
        if gated:
            notes.append(f"gated by juniorctl security: {','.join(gated)}")
        for k in ungated:
            if k == "defaultAction":
                continue
            notes.append(f"{k} literal {lits.get(k)!r}")
        if "defaultAction" in info["keys"]:
            want = info["keys"]["defaultAction"]
            have = None
            if seccomp is not None:
                try:
                    have = json.loads(seccomp).get("defaultAction")
                except ValueError:
                    have = "<bad json>"
            trit = 0 if have == want else -1
            notes.append(f"defaultAction on disk {have!r} want {want!r}; juniorctl security only checks the file exists")
        if info["files"] and "defaultAction" not in info["keys"]:
            notes.append(f"file checks {','.join(info['files'])} outside juniorctl exit")
            trit = min(trit, 0) if trit == 1 and not gated else trit
        if bad:
            trit = -1
            notes.append(f"ungated non-literal keys {bad}")
        row(f"case {name}", trit, "; ".join(notes) or "no security keys")

    consts = i2sd_consts(i2sd)
    i2_ok = consts.get("HOST") == LOOPBACK and consts.get("PORT") == 8767
    row("i2sd lane", 1 if i2_ok else -1, f"HOST {consts.get('HOST')!r} PORT {consts.get('PORT')!r}")

    for port in PORTS:
        for cmd in LIVE_BETA:
            lanes = LANES[cmd]
            good = all(l.startswith(LOOPBACK + ":") for l in lanes) and "0.0.0.0" not in " ".join(lanes)
            if I2SD in lanes:
                good = good and i2_ok
            if BITNETD in lanes and cmd in ("health", "security"):
                good = good and BITNETD in ctl_cli + juniorctl
            codes = sorted(EXPECTED[cmd])
            row(f"{port} {cmd}", 1 if good else -1, f"exit {codes} over {' '.join(lanes)}")
    return rows


def chain(rows: list[dict]) -> str:
    prev = GENESIS
    for r in rows:
        text = f"{r['row']}|{r['trit']}|{r['detail']}|rate={RATE}"
        prev = close(prev, pack_trits([r["trit"]]), text)
        r["close"] = prev
    return prev


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def live() -> list[dict]:
    seccomp = LINUX / "seccomp-bitnetd.json"
    return build(
        _read(LINUX / "ctl_cli.py"),
        _read(LINUX / "juniorctl.py"),
        _read(TESTS / "test_linux_security.py"),
        _read(LINUX / "i2sd.py"),
        _read(seccomp) if seccomp.is_file() else None,
    )


# ---------------------------------------------------------------- selftest
_FAKE_CLI = '''
def run(argv, ns):
    cmd = argv[1] if len(argv) > 1 else "health"
    if cmd == "health":
        return 0
    if cmd == "security":
        report = ns["security"]()
        return 0 if report["unit_ok"] and report["seccomp"] else 1
    if cmd == "ask":
        return 0
    if cmd == "night":
        return 0
    return 2
'''
_FAKE_CTL = '''
def security():
    return {"unit_ok": True, "seccomp": True, "bind": "127.0.0.1:8765", "privileged": False, "docker_socket": False}
def main(argv):
    return 0
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''
_FAKE_TEST = '''
import unittest
class T(unittest.TestCase):
    def test_unit_hardened(self):
        r = security()
        self.assertTrue(r["unit_ok"], r["missing"])
        self.assertTrue(r["seccomp"])
        self.assertFalse(r["privileged"])
        self.assertFalse(r["docker_socket"])
    def test_seccomp_default_deny(self):
        raw = json.loads((LINUX / "seccomp-bitnetd.json").read_text())
        self.assertEqual(raw["defaultAction"], "SCMP_ACT_ERRNO")
if __name__ == "__main__":
    unittest.main()
'''
_FAKE_I2SD = 'HOST = "127.0.0.1"\nPORT = 8767\n'


def selftest() -> int:
    fails = 0

    def check(name: str, cond: bool) -> None:
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        fails += 0 if cond else 1

    check("pack 4 trits", pack_trits([1, -1, 0, 1]) == bytes([0b01100001]))
    check("pack pads", pack_trits([-1]) == bytes([0b10000000]))
    a = close(GENESIS, b"\x01", "x")
    check("close is 16 hex", len(a) == 16 and int(a, 16) >= 0)
    check("close depends on prev", a != close("1" * 16, b"\x01", "x"))
    check("flags all false", not any(FLAGS.values()))
    check("rate 0.05", RATE == 0.05)

    good = build(_FAKE_CLI, _FAKE_CTL, _FAKE_TEST, _FAKE_I2SD, '{"defaultAction": "SCMP_ACT_ERRNO"}')
    check("good fixture no fail", all(r["trit"] >= 0 for r in good))
    sec = next(r for r in good if r["row"] == "exit security")
    check("security folds to {0,1}", sec["trit"] == 1)
    tt = next(r for r in good if r["row"] == "security truth table")
    check("truth table exact", tt["trit"] == 1 and tt["detail"].count("->1") == 3)

    bad_cli = _FAKE_CLI.replace('report["unit_ok"] and report["seccomp"]', 'report["unit_ok"] or report["seccomp"]')
    bad = build(bad_cli, _FAKE_CTL, _FAKE_TEST, _FAKE_I2SD, '{"defaultAction": "SCMP_ACT_ERRNO"}')
    check("or-gate caught", any(r["row"] == "security truth table" and r["trit"] == -1 for r in bad))

    drift = build(_FAKE_CLI.replace("return 2", "return 0"), _FAKE_CTL, _FAKE_TEST, _FAKE_I2SD, None)
    check("unknown->0 caught", any(r["row"] == "exit <unknown>" and r["trit"] == -1 for r in drift))
    check("missing seccomp caught", any(r["row"] == "case test_seccomp_default_deny" and r["trit"] == -1 for r in drift))

    wide = build(_FAKE_CLI, _FAKE_CTL, _FAKE_TEST, 'HOST = "0.0.0.0"\nPORT = 8767\n', None)
    check("i2sd 0.0.0.0 caught", any(r["row"] == "i2sd lane" and r["trit"] == -1 for r in wide))

    priv = build(_FAKE_CLI, _FAKE_CTL.replace('"privileged": False', '"privileged": flag()'), _FAKE_TEST, _FAKE_I2SD, None)
    check("non-literal privileged caught", any(r["row"] == "case test_unit_hardened" and r["trit"] == -1 for r in priv))

    t1, t2 = chain([dict(r) for r in good]), chain([dict(r) for r in good])
    check("chain deterministic", t1 == t2)

    if (LINUX / "ctl_cli.py").is_file():
        rows = live()
        check("live repo no fail", all(r["trit"] >= 0 for r in rows))
    print(f"selftest: {'pass' if fails == 0 else 'FAIL'} ({fails} failed)")
    return 0 if fails == 0 else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    rows = live()
    tip = chain(rows)
    passed = sum(1 for r in rows if r["trit"] == 1)
    adv = sum(1 for r in rows if r["trit"] == 0)
    failed = sum(1 for r in rows if r["trit"] == -1)
    if "--json" in argv:
        print(json.dumps({"rows": rows, "tip": tip, "rate": RATE, "flags": FLAGS,
                          "bind": LOOPBACK}, indent=2))
    else:
        for r in rows:
            mark = {1: "pass", 0: "adv ", -1: "FAIL"}[r["trit"]]
            print(f"{mark} {r['close']} {r['row']}: {r['detail']}")
        print(f"live beta exit-code map: {passed} pass, {adv} advisory, {failed} fail, tip {tip}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
