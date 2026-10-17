"""GGUF on-disk gate port map: in-process handshake check (no sockets).

Advances JuniorLLM docs/BETA_TO_OS.md, item "JuniorOS overlay", clause
"I2_S 2B4T only if GGUF already on disk."

The operator sets JUNIOR_GGUF; bots never download. This maps how the JuniorLLM
side reads JUNIOR_GGUF and checks, as plain text and AST only:
  * rails/linux/bitnetd.sh execs bitnet.cpp on 127.0.0.1:8765 only behind
    [ -f "$JUNIOR_GGUF" ], refuses non-loopback binds, and otherwise falls back
    to i2sd.py;
  * rails/linux/i2sd.py pins HOST 127.0.0.1 and PORT 8767;
  * rails/linux/bitnet_cpp.py and rails/linux/llama.py plan() read JUNIOR_GGUF,
    gate "ready" on Path(...).is_file(), and build the 127.0.0.1:8765 command
    only inside "if ready:";
  * rails/linux/bitnetd.service pins JUNIOR_BIND=127.0.0.1:8765 and
    JUNIOR_OFFLINE=1 and does not hardcode a GGUF path;
  * no reader carries a download token or a 0.0.0.0 bind;
  * a pure truth-table model routes Home, OSai, FieldCore, Flagstaff and
    JuniorLLM to bitnetd 127.0.0.1:8765 only when cpp + JUNIOR_GGUF + file on
    disk, else to i2sd 127.0.0.1:8767.

No repo module is imported, no socket is opened, nothing is fetched, no model is
pulled, nothing is launched. Each row is closed with sha256(prev || pack || text)[:16].
Stdlib only. Run: python ports/gguf_disk_gate_port_map.py  (exit 0 on pass).
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"

LOOPBACK = "127.0.0.1"
BITNETD = (LOOPBACK, 8765)
I2SD = (LOOPBACK, 8767)
NODES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
RATE = 0.05
PACK = "gguf_disk_gate|JTR1|one-trit-pack"
GENESIS = "0" * 16
FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
    "download": False,
    "model_pull": False,
}
DOWNLOAD_TOKENS = (
    "curl", "wget", "urllib", "requests", "huggingface", "hf_hub",
    "http://", "https://", "git clone", "snapshot_download",
)


def close(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def _src(name: str) -> str:
    return (LINUX / name).read_text(encoding="utf-8")


def _plan(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "plan":
            return node
    return None


def _strings(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _plan_check(name: str) -> tuple[bool, str]:
    fn = _plan(ast.parse(_src(name)))
    if fn is None:
        return False, f"{name}: no plan()"
    reads_env = "JUNIOR_GGUF" in _strings(fn)
    ready_is_file = False
    cmd_gated = False
    cmd_ok = False
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ready" for t in stmt.targets):
            ready_is_file = any(
                isinstance(n, ast.Attribute) and n.attr == "is_file" for n in ast.walk(stmt.value)
            )
        if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "ready":
            for sub in stmt.body:
                if isinstance(sub, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cmd" for t in sub.targets):
                    cmd_gated = True
                    s = _strings(sub.value)
                    cmd_ok = all(x in s for x in ("-m", "--host", LOOPBACK, "--port", str(BITNETD[1])))
    ungated = [
        s for s in fn.body
        if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cmd" for t in s.targets)
        and not (isinstance(s.value, ast.Constant) and s.value.value is None)
    ]
    ok = reads_env and ready_is_file and cmd_gated and cmd_ok and not ungated
    return ok, f"{name}: env={reads_env} is_file={ready_is_file} gated={cmd_gated} cmd={cmd_ok} ungated={len(ungated)}"


def _i2sd_consts() -> dict[str, object]:
    out: dict[str, object] = {}
    for node in ast.parse(_src("i2sd.py")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
    return out


def _shell_check() -> tuple[bool, str]:
    sh = _src("bitnetd.sh")
    lines = sh.splitlines()
    gate = next((i for i, ln in enumerate(lines) if '-f "$JUNIOR_GGUF"' in ln), -1)
    cpp_exec = next((i for i, ln in enumerate(lines) if "exec" in ln and "$JUNIOR_GGUF" in ln), -1)
    fi = next((i for i, ln in enumerate(lines) if i > gate and ln.strip() == "fi"), -1)
    fallback = next((i for i, ln in enumerate(lines) if "exec" in ln and "i2sd.py" in ln), -1)
    exec_ok = cpp_exec >= 0 and f"--host {LOOPBACK} --port {BITNETD[1]}" in lines[cpp_exec]
    refuse = "refusing non-loopback" in sh and f"{LOOPBACK}:{BITNETD[1]}" in sh
    ok = 0 <= gate < cpp_exec < fi < fallback and exec_ok and refuse
    return ok, f"bitnetd.sh: gate@{gate} exec@{cpp_exec} fi@{fi} i2sd_fallback@{fallback} loopback={exec_ok} refuse={refuse}"


def route(has_cpp: bool, gguf_set: bool, on_disk: bool) -> tuple[str, int]:
    """Pure model of bitnetd.sh: bitnet.cpp only if the GGUF is already on disk."""
    if has_cpp and gguf_set and on_disk:
        return BITNETD
    return I2SD


def checks() -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    rows.append(("bitnetd.sh gate", *_shell_check()))
    c = _i2sd_consts()
    rows.append(("i2sd 127.0.0.1:8767", c.get("HOST") == I2SD[0] and c.get("PORT") == I2SD[1],
                 f"i2sd.py HOST={c.get('HOST')} PORT={c.get('PORT')}"))
    for name in ("bitnet_cpp.py", "llama.py"):
        rows.append((f"{name} plan gate", *_plan_check(name)))
    svc = _src("bitnetd.service")
    svc_ok = (f"JUNIOR_BIND={LOOPBACK}:{BITNETD[1]}" in svc and "JUNIOR_OFFLINE=1" in svc
              and "JUNIOR_GGUF" not in svc)
    rows.append(("bitnetd.service", svc_ok, "bind 127.0.0.1:8765, offline 1, no GGUF path hardcoded"))
    readers = ("bitnetd.sh", "bitnet_cpp.py", "llama.py", "bitnetd.service", "i2sd.py")
    for name in readers:
        low = _src(name).lower()
        hits = [t for t in DOWNLOAD_TOKENS if re.search(re.escape(t), low)]
        any_bind = "0.0.0.0" in low
        rows.append((f"{name} no fetch/0.0.0.0", not hits and not any_bind,
                     f"download_tokens={hits} any_bind={any_bind}"))
    table = list(itertools.product((False, True), repeat=3))
    to_bitnetd = [t for t in table if route(*t) == BITNETD]
    rows.append(("truth table", to_bitnetd == [(True, True, True)],
                 f"bitnetd only on cpp+env+disk; {len(table) - len(to_bitnetd)}/8 -> i2sd"))
    for node in NODES:
        on = route(True, True, True)
        off = route(True, True, False)
        ok = on == BITNETD and off == I2SD and on[0] == off[0] == LOOPBACK
        rows.append((f"{node} port map", ok, f"{node}: gguf on disk -> {on[0]}:{on[1]}, absent -> {off[0]}:{off[1]}"))
    rows.append(("flags", not any(FLAGS.values()), json.dumps(FLAGS, sort_keys=True)))
    rows.append(("rate", RATE == 0.05, f"rate {RATE}"))
    return rows


def main() -> int:
    prev = GENESIS
    out = []
    for name, ok, text in checks():
        prev = close(prev, PACK, f"{name}|{int(ok)}|{text}")
        out.append({"check": name, "ok": ok, "text": text, "close": prev})
    passed = all(r["ok"] for r in out)
    print(json.dumps({
        "item": "BETA_TO_OS JuniorOS overlay: I2_S 2B4T only if GGUF already on disk",
        "pack": PACK,
        "bitnetd": f"{BITNETD[0]}:{BITNETD[1]}",
        "i2sd": f"{I2SD[0]}:{I2SD[1]}",
        "rate": RATE,
        "flags": FLAGS,
        "rows": out,
        "passed": sum(r["ok"] for r in out),
        "total": len(out),
        "tip": prev,
        "ok": passed,
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
