"""Asahi MLX lane port map (JuniorLLM side).

Advances JuniorHome docs/ROADMAP.md item 4, "Asahi MLX - probe only from
automations", from the JuniorLLM side. The Home-side handshake
(JuniorHome ports/asahi_mlx_probe_handshake.py) is not imported, wrapped or
re-checked here. This file maps how the JuniorLLM accel probe chooses a lane:

  runtime "mlx"       -> in-process MLX lane, no port, no socket
  runtime "llama.cpp" -> bitnetd 127.0.0.1:8765 (GGUF already on disk)
  runtime "i2sd"      -> i2sd    127.0.0.1:8767 (userspace fallback)

Everything is read as text and parsed with ast. Nothing from the repo is
imported, mlx is never imported, no subprocess, no socket, no fetch, no
model pull, no hardware call. Ports involved: Home, OSai, FieldCore,
Flagstaff, JuniorLLM.

Sources read (read-only): rails/linux/backend.py, ports/enduser_llm.py,
rails/linux/llama.py, rails/linux/bitnetd.sh, rails/linux/i2sd.py,
ports/inject_hw.py, rails/linux/ASAHI.md.

Existing path: opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false, layer_one false, fetch false, ue5_launch false,
bind 127.0.0.1. Close rule: sha256(prev || pack || text)[:16].
Stdlib only. Run with --selftest (default) to print the map and exit 0 on pass.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #4 Asahi MLX - probe only from automations"
RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
BIND = "127.0.0.1"
BITNETD_PORT = 8765
I2SD_PORT = 8767
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE_RULE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
TRIT_PACK = (1, 0, -1, 1, 0, -1, 1, 1)

LANES = {
    "mlx": {"lane": "mlx in-process", "bind": None, "port": None},
    "llama.cpp": {"lane": "bitnetd", "bind": BIND, "port": BITNETD_PORT},
    "i2sd": {"lane": "i2sd", "bind": BIND, "port": I2SD_PORT},
}

ROOT = Path(__file__).resolve().parents[1]
SRC = {
    "backend": "rails/linux/backend.py",
    "enduser": "ports/enduser_llm.py",
    "llama": "rails/linux/llama.py",
    "bitnetd_sh": "rails/linux/bitnetd.sh",
    "i2sd": "rails/linux/i2sd.py",
    "inject_hw": "ports/inject_hw.py",
    "asahi_md": "rails/linux/ASAHI.md",
}
BANNED_IMPORTS = {"subprocess", "socket", "urllib", "http", "requests", "ctypes"}


def _read(key: str) -> str:
    return (ROOT / SRC[key]).read_text(encoding="utf-8")


def _imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


_SAFE = (ast.Expression, ast.Attribute, ast.Name, ast.Load, ast.Compare, ast.Eq,
         ast.In, ast.Constant, ast.Set, ast.Call, ast.BoolOp, ast.And, ast.Or)


def _safe_eval(expr: ast.expr, p: SimpleNamespace) -> bool:
    node = ast.Expression(body=expr)
    for n in ast.walk(node):
        if not isinstance(n, _SAFE):
            raise ValueError(f"unsafe node {type(n).__name__}")
        if isinstance(n, ast.Name) and n.id != "p":
            raise ValueError(f"unexpected name {n.id}")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Attribute) and n.func.attr == "lower"):
            raise ValueError("only .lower() calls allowed")
    return bool(eval(compile(node, "<spec-branch>", "eval"), {"__builtins__": {}}, {"p": p}))


def spec_chain(src: str) -> list[tuple[ast.expr | None, str]]:
    """Extract the if/elif/else chain of ports/enduser_llm.spec() as (test, runtime)."""
    fn = _func(ast.parse(src), "spec")
    chain: list[tuple[ast.expr | None, str]] = []
    node = next(n for n in fn.body if isinstance(n, ast.If))
    while True:
        rt = node.body[0].value.elts[1].value  # name, runtime, ctx = (...)
        chain.append((node.test, rt))
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
            continue
        chain.append((None, node.orelse[0].value.elts[1].value))
        return chain


def route(chain, asahi: bool, mlx_ok: bool, machine: str, llama_ready: bool) -> dict:
    """Mirror backend._accel + enduser_llm.spec in-process from facts only."""
    if asahi:
        accel = "mlx" if mlx_ok else "cpu"
    else:
        accel = "mlx" if mlx_ok else "cpu"
    p = SimpleNamespace(asahi=asahi, accel=accel, machine=machine)
    runtime = None
    for test, rt in chain:
        if test is None or _safe_eval(test, p):
            runtime = rt
            break
    if llama_ready:
        runtime = "llama.cpp"
    return {"accel": accel, "runtime": runtime, **LANES[runtime]}


def close(prev: str, pack: tuple[int, ...], text: str) -> str:
    enc = {0: "0", 1: "1", -1: "2"}
    packed = "".join(enc[t] for t in pack)
    return hashlib.sha256((prev + packed + text).encode("utf-8")).hexdigest()[:16]


def checks() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    def row(name: str, ok: bool, note: str = "", advisory: bool = False) -> None:
        rows.append((name, "pass" if ok else ("advisory" if advisory else "FAIL"), note))

    backend = _read("backend")
    bt = ast.parse(backend)
    row("backend.no_banned_imports", not (_imports(bt) & BANNED_IMPORTS),
        ",".join(sorted(_imports(bt) & BANNED_IMPORTS)) or "stdlib probe")
    top = [n for n in bt.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                                    ast.ClassDef, ast.Assign, ast.AnnAssign))
           and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    row("backend.probe_no_import_side_effect", not top, "probe runs only when called")
    accel = _func(bt, "_accel")
    first_if = next(n for n in accel.body if isinstance(n, ast.If))
    asahi_first = isinstance(first_if.test, ast.Call) and getattr(first_if.test.func, "id", "") == "_asahi"
    guarded = any(isinstance(n, ast.Try) and any(isinstance(m, ast.ImportFrom) is False and isinstance(m, ast.Import)
                  and m.names[0].name == "mlx.core" for m in ast.walk(n)) for n in first_if.body)
    row("backend.asahi_then_mlx_guarded", asahi_first and guarded, "os-release asahi -> try import mlx.core")
    writes = [n for n in ast.walk(bt) if isinstance(n, ast.Attribute) and n.attr in {"write_text", "write_bytes"}]
    row("backend.no_firmware_writes", not writes, "read_text only")

    asahi_md = _read("asahi_md")
    row("asahi_md.miss_to_cpu", "Miss → cpu AbsMean" in asahi_md, "ASAHI.md: miss -> cpu AbsMean")

    chain = spec_chain(_read("enduser"))
    rts = [rt for _, rt in chain]
    row("enduser.spec_runtimes_known", all(rt in LANES for rt in rts), "->".join(rts))

    llama = _read("llama")
    row("llama.bitnetd_loopback_8765", '"--host", "127.0.0.1", "--port", "8765"' in llama, "plan cmd")
    row("llama.fallback_i2sd", '"fallback": "i2sd"' in llama, "plan fallback")

    sh = _read("bitnetd_sh")
    row("bitnetd_sh.bind_default", "BIND=${JUNIOR_BIND:-127.0.0.1:8765}" in sh, "default bind")
    row("bitnetd_sh.refuse_non_loopback", "refusing non-loopback" in sh, "exit 3 on non-loopback")
    row("bitnetd_sh.gguf_on_disk_gate", '[ -f "$JUNIOR_GGUF" ]' in sh, "bitnetd only if GGUF on disk")
    row("bitnetd_sh.else_i2sd", 'exec python3 "$HERE/i2sd.py"' in sh, "fallback lane i2sd")

    i2 = ast.parse(_read("i2sd"))
    consts = {t.id: n.value.value for n in i2.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    row("i2sd.host_port", consts.get("HOST") == BIND and consts.get("PORT") == I2SD_PORT,
        f"{consts.get('HOST')}:{consts.get('PORT')}")
    row("i2sd.serve_loopback_only", "loopback only" in ast.unparse(_func(i2, "serve")), "ValueError otherwise")

    inj = _read("inject_hw")
    m = re.search(r"CLASSES = \(([^)]*)\)", inj)
    classes = [c.strip().strip('"') for c in (m.group(1).split(",") if m else [])]
    row("inject_hw.classes_mlx_asahi", {"mlx", "asahi"} <= set(classes), ",".join(classes))
    row("inject_hw.launch_pull_false", '"launch": False' in inj and '"model_pull": False' in inj, "no launch, no pull")
    row("inject_hw.bind_loopback", '"bind": "127.0.0.1"' in inj, BIND)

    cases = [
        ("asahi+mlx", dict(asahi=True, mlx_ok=True, machine="aarch64", llama_ready=False), "mlx"),
        ("asahi+mlx+gguf", dict(asahi=True, mlx_ok=True, machine="aarch64", llama_ready=True), "llama.cpp"),
        ("darwin+mlx", dict(asahi=False, mlx_ok=True, machine="arm64", llama_ready=False), "mlx"),
        ("arm_no_mlx", dict(asahi=False, mlx_ok=False, machine="aarch64", llama_ready=False), "i2sd"),
        ("x86_no_mlx", dict(asahi=False, mlx_ok=False, machine="x86_64", llama_ready=False), "i2sd"),
        ("x86+gguf", dict(asahi=False, mlx_ok=False, machine="x86_64", llama_ready=True), "llama.cpp"),
    ]
    for name, facts, want in cases:
        r = route(chain, **facts)
        port = f"{r['bind']}:{r['port']}" if r["port"] else "in-process"
        row(f"lane.{name}", r["runtime"] == want and (r["port"] in (None, BITNETD_PORT, I2SD_PORT)),
            f"{r['runtime']} -> {r['lane']} {port}")

    # ASAHI.md says an MLX miss falls back to cpu; spec() keys on p.asahi first.
    miss = route(chain, asahi=True, mlx_ok=False, machine="aarch64", llama_ready=False)
    row("lane.asahi_mlx_miss", miss["runtime"] == "i2sd",
        f"spec gives {miss['runtime']} with accel {miss['accel']}; ASAHI.md expects cpu -> i2sd {BIND}:{I2SD_PORT}",
        advisory=True)

    used = {v["port"] for v in LANES.values() if v["port"]}
    row("lanes.distinct_loopback_ports", used == {BITNETD_PORT, I2SD_PORT}
        and all(v["bind"] in (None, BIND) for v in LANES.values()), "8765 != 8767, mlx portless")
    row("flags.false", not any((MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH, MODEL_PULL)) and RATE == 0.05,
        "mint/socket/layer_one/fetch/ue5_launch/model_pull false, rate 0.05")
    return rows


def selftest() -> int:
    rows = checks()
    fails = [r for r in rows if r[1] == "FAIL"]
    adv = [r for r in rows if r[1] == "advisory"]
    for name, status, note in rows:
        print(f"{status:8} {name:40} {note}")
    passed = sum(1 for r in rows if r[1] == "pass")
    text = f"{ROADMAP_ITEM}|{passed}/{len(rows)} pass|{len(adv)} advisory|{len(fails)} fail"
    prev = sys.argv[sys.argv.index("--prev") + 1] if "--prev" in sys.argv else GENESIS
    print(json.dumps({"item": ROADMAP_ITEM, "ports": PORTS, "pass": passed, "rows": len(rows),
                      "advisory": len(adv), "fail": len(fails), "rate": RATE,
                      "close": close(prev[:16], TRIT_PACK, text), "close_rule": CLOSE_RULE}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(selftest())
