#!/usr/bin/env python3
"""Live beta step 5: backfill_check step-order writer map (handshake + port map).

Advances docs/BETA_TO_OS.md, section "Live beta", step 5:
"`python scripts/backfill_check.py` exits 0" -- read as the fifth step of a
sequence, after steps 1-4 have already run on the same fresh clone.

Already landed for step 5 and not redone here: the exit code (9244444,
adfb204), coverage of the 12-path EXPECTED list (829a789), the fresh-clone
.git/index side (c93a5ec, 5032022), the HEAD-tree key (e82c972) and the
quant-error substance gate (JuniorQuant dbafb2e).  This file takes the
step-order side: backfill_check asks Path.is_file() *after* steps 1-4, so its
0 only attests the clone if no earlier step can create, truncate, move or
delete an EXPECTED path (or a bitnetd/i2sd port source).  If one can, step 5
is order-dependent: it may pass only because step 1-4 wrote the file, or a
step may remove what step 5 then reports missing.

In-process, read-only, with ast and no imports of repo code, this script:

  1. builds the repo-local transitive import closure of the step 1-3
     entrypoints (rails/linux/juniorctl.py, tests/test_junior_aie.py,
     tests/test_linux_security.py), following lazy imports inside functions
     and `from x import *`; step 4 lives in JuniorClimbs and is an advisory;
  2. finds every write site in that closure: open() in w/a/x/+ mode,
     write_text/write_bytes, unlink/rmdir/rename/replace/touch, os.remove /
     os.unlink / os.rename / os.replace, shutil.rmtree/move/copy*;
  3. resolves string literals naming each site's target (the call's
     receiver or path argument, or else same-function assignments to the
     bare name it uses) and maps them to EXPECTED and to
     the port sources: a delete/move hitting a target is -1, a create or
     truncate hitting a target is 0 (step 5 then attests generated output),
     no writer hitting it is +1;  a writer whose target is wholly dynamic is
     an advisory counted once per module;
  4. port map: bitnetd 127.0.0.1:8765 (bitnetd.sh, bitnetd.service) and i2sd
     127.0.0.1:8767 (i2sd.py) are read statically for their literal next to
     127.0.0.1 and must not be in any step 1-4 writer's reach.

Ports touched: bitnetd 127.0.0.1:8765, i2sd 127.0.0.1:8767 (static reads).
Parties: Home, OSai, FieldCore, Flagstaff, JuniorLLM.  One trit pack (+1
pass, 0 advisory, -1 fail), base rate 0.05, closed with
sha256(prev || pack || text)[:16].  No sockets, no subprocess, no shell, no
download, no model pull.  Stdlib only.  mint, socket, layer_one, fetch and
ue5_launch are all false.  Bind 127.0.0.1 only.

    python3 ports/live_beta_backfill_step_order_writer_map.py
    python3 ports/live_beta_backfill_step_order_writer_map.py --json
    python3 ports/live_beta_backfill_step_order_writer_map.py --selftest

Exit 0 when no -1 row, else 1.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE = 0.05
BIND = "127.0.0.1"
GENESIS = "0" * 16
PARTIES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False,
         "ue5_launch": False, "bind": BIND}
ITEM = "BETA_TO_OS Live beta step 5 (step-order writer side)"

ENTRYPOINTS = {
    "step1 juniorctl health": "rails/linux/juniorctl.py",
    "step2 juniorctl ask/night/security": "rails/linux/juniorctl.py",
    "step3 test_junior_aie": "tests/test_junior_aie.py",
    "step3 test_linux_security": "tests/test_linux_security.py",
}
STEP4_FOREIGN = "JuniorClimbs scripts/stonefield_stdlib_probe.py"
PORT_SOURCES = {
    "rails/linux/bitnetd.sh": 8765,
    "rails/linux/bitnetd.service": 8765,
    "rails/linux/i2sd.py": 8767,
}
DESTROY = {"unlink", "rmdir", "rename", "replace", "remove", "rmtree", "move"}
CREATE = {"write_text", "write_bytes", "touch", "copy", "copyfile", "copy2",
          "copytree", "mkdir"}
OS_MODS = {"os", "shutil"}


def read_expected(root: Path) -> list[str]:
    src = (root / "scripts/backfill_check.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "EXPECTED" for t in node.targets):
            return [e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def mod_to_path(root: Path, mod: str) -> Path | None:
    base = root.joinpath(*mod.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def imports_of(root: Path, path: Path, tree: ast.AST) -> set[Path]:
    out: set[Path] = set()
    pkg = path.relative_to(root).parent.parts
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(pkg[: len(pkg) - node.level + 1])
                head = ".".join(base + ([node.module] if node.module else []))
            else:
                head = node.module or ""
            names = [head] + [f"{head}.{a.name}" for a in node.names if a.name != "*"]
        for n in names:
            parts = n.split(".")
            for i in range(len(parts), 0, -1):
                p = mod_to_path(root, ".".join(parts[:i]))
                if p:
                    out.add(p)
                    break
    return out


def closure(root: Path, starts: list[str]) -> dict[Path, ast.AST]:
    seen: dict[Path, ast.AST] = {}
    todo = [root / s for s in starts if (root / s).is_file()]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        seen[p] = tree
        todo.extend(q for q in imports_of(root, p, tree) if q not in seen)
    return seen


def _strs(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def write_sites(tree: ast.AST) -> list[dict]:
    """Return write sites: {kind, op, line, literals}."""
    sites = []
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    owner: dict[int, ast.AST] = {}
    for f in funcs:
        for n in ast.walk(f):
            owner.setdefault(id(n), f)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn, op, kind = node.func, None, None
        if isinstance(fn, ast.Name) and fn.id == "open":
            mode = None
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and set(mode) & set("wax+"):
                op, kind = "open:" + mode, "create"
        elif isinstance(fn, ast.Attribute):
            a = fn.attr
            is_os = isinstance(fn.value, ast.Name) and fn.value.id in OS_MODS
            if a == "open":
                mode = None
                if node.args and isinstance(node.args[0], ast.Constant):
                    mode = node.args[0].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if isinstance(mode, str) and set(mode) & set("wax+"):
                    op, kind = "open:" + mode, "create"
            elif a in DESTROY and (is_os or a in {"unlink", "rmdir", "rename",
                                                  "replace", "rmtree", "move"}):
                if a == "replace" and not is_os and len(node.args) == 2 and all(
                        isinstance(x, ast.Constant) and isinstance(x.value, str)
                        for x in node.args):
                    continue  # str.replace("a", "b")
                op, kind = a, "destroy"
            elif a in CREATE and a != "mkdir":
                op, kind = a, "create"
        if not op:
            continue
        # Literals that name the target: the call itself (receiver + args),
        # plus same-function assignments to a bare-name receiver or argument.
        tgt = fn.value if isinstance(fn, ast.Attribute) and not (
            isinstance(fn.value, ast.Name) and fn.value.id in OS_MODS) else (
            node.args[0] if node.args else node)
        lits = _strs(tgt)
        f = owner.get(id(node))
        names = set() if lits else {
            n.id for n in ast.walk(tgt) if isinstance(n, ast.Name)}
        scope = f if f is not None else tree
        for n in ast.walk(scope):
            if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in names for t in n.targets):
                lits += _strs(n.value)
        sites.append({"op": op, "kind": kind, "line": node.lineno,
                      "literals": sorted(set(lits))})
    return sites


def hits(target: str, literals: list[str]) -> bool:
    name = Path(target).name
    for s in literals:
        s2 = s.strip().lstrip("./")
        if s2 == target or s2.endswith("/" + target) or s2 == name:
            return True
        if "/" in target and s2 and target.endswith("/" + s2) and len(s2) > 3:
            return True
    return False


def port_literal(root: Path, rel: str, port: int) -> bool:
    p = root / rel
    if not p.is_file():
        return False
    txt = p.read_text(errors="replace")
    return bool(re.search(r"127\.0\.0\.1\D{0,40}" + str(port), txt, re.S)) \
        and "0.0.0.0" not in txt


def survey(root: Path) -> list[dict]:
    rows: list[dict] = []

    def row(t: int, key: str, note: str) -> None:
        rows.append({"trit": t, "key": key, "note": note})

    expected = read_expected(root)
    row(1 if len(expected) else -1, "expected.read",
        f"{len(expected)} EXPECTED paths read statically from backfill_check.py")
    starts = sorted(set(ENTRYPOINTS.values()))
    missing_entry = [s for s in starts if not (root / s).is_file()]
    row(-1 if missing_entry else 1, "closure.entrypoints",
        "steps 1-3 entrypoints present" if not missing_entry
        else f"missing {missing_entry}")
    mods = closure(root, starts)
    row(1 if mods else -1, "closure.size",
        f"{len(mods)} repo-local modules reachable from steps 1-3")
    row(0, "closure.step4", f"{STEP4_FOREIGN} is outside this repo; its writes "
        "are not mapped here (it must not touch JuniorLLM paths)")

    sites = []
    for p, tree in sorted(mods.items()):
        rel = p.relative_to(root).as_posix()
        for s in write_sites(tree):
            s["module"] = rel
            sites.append(s)
    dyn_mods = sorted({s["module"] for s in sites if not s["literals"]})
    row(1 if not sites else 0, "closure.writers",
        f"{len(sites)} write sites in {len({s['module'] for s in sites})} modules")
    if dyn_mods:
        row(0, "closure.dynamic", "write targets with no literal (unmapped): "
            + ", ".join(dyn_mods[:8]) + (" ..." if len(dyn_mods) > 8 else ""))

    targets = [("expected", t) for t in expected] + \
              [("port", t) for t in PORT_SOURCES]
    for kind, t in targets:
        h = [s for s in sites if hits(t, s["literals"])]
        destroy = [s for s in h if s["kind"] == "destroy"]
        where = "; ".join(f"{s['module']}:{s['line']} {s['op']}" for s in h[:3])
        if destroy:
            row(-1, f"{kind}.{t}", "step 1-3 closure can delete/move it: " + where)
        elif h:
            row(0, f"{kind}.{t}", "step 1-3 closure can create/truncate it "
                "(step 5 would attest generated output): " + where)
        else:
            row(1, f"{kind}.{t}", "no step 1-3 writer reaches it; step 5 "
                "is order-independent here")

    for rel, port in PORT_SOURCES.items():
        ok = port_literal(root, rel, port)
        row(1 if ok else -1, f"portmap.{Path(rel).name}",
            f"{BIND}:{port} literal present, no wildcard bind" if ok
            else f"{BIND}:{port} literal missing or wildcard present")
    return rows


def pack_trits(trits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            v = v * 3 + (t + 1)
        out.append(v)
    return bytes(out)


def close(rows: list[dict], prev: str = GENESIS) -> dict:
    trits = [r["trit"] for r in rows]
    pack = pack_trits(trits)
    text = "\n".join(f"{r['trit']:+d} {r['key']} {r['note']}" for r in rows)
    seal = hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]
    adv = trits.count(0)
    return {"item": ITEM, "parties": list(PARTIES), "rate": RATE,
            "pass": trits.count(1), "advisory": adv, "fail": trits.count(-1),
            "advisory_share": round(adv / len(trits), 3) if trits else 0.0,
            "prev": prev, "pack": pack.hex(), "close": seal, "flags": FLAGS,
            "rows": rows}


def head_prev(root: Path) -> str:
    g = root / ".git"
    try:
        h = (g / "HEAD").read_text().strip()
        if h.startswith("ref:"):
            ref = h.split(None, 1)[1]
            f = g / ref
            if f.is_file():
                return f.read_text().strip()[:16]
            pr = g / "packed-refs"
            if pr.is_file():
                for line in pr.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line[:16]
            return GENESIS
        return h[:16]
    except OSError:
        return GENESIS


def selftest() -> int:
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        for rel in ("scripts", "rails/linux", "tests", "docs", "pkg"):
            (r / rel).mkdir(parents=True, exist_ok=True)
        (r / "scripts/backfill_check.py").write_text(
            'EXPECTED = ["docs/A.md", "STATE.md", "docs/C.md"]\n')
        (r / "pkg/__init__.py").write_text("")
        (r / "pkg/w.py").write_text(
            "from pathlib import Path\n"
            "def go():\n    Path('STATE.md').write_text('x')\n"
            "def rm():\n    import os\n    os.remove('docs/C.md')\n"
            "def s():\n    return 'a'.replace('a', 'b')\n")
        (r / "rails/linux/juniorctl.py").write_text(
            "def night():\n    from pkg.w import go\n    go()\n")
        (r / "tests/test_junior_aie.py").write_text("import pkg.w\n")
        (r / "tests/test_linux_security.py").write_text("x = 1\n")
        (r / "rails/linux/bitnetd.sh").write_text("BIND=127.0.0.1:8765\n")
        (r / "rails/linux/bitnetd.service").write_text("JUNIOR_BIND=127.0.0.1:8765\n")
        (r / "rails/linux/i2sd.py").write_text('HOST = "0.0.0.0"\nPORT = 8767\n')
        rows = {x["key"]: x["trit"] for x in survey(r)}
        checks = [
            rows.get("expected.read") == 1,
            rows.get("closure.size") == 1,
            rows.get("expected.STATE.md") == 0,
            rows.get("expected.docs/C.md") == -1,
            rows.get("expected.docs/A.md") == 1,
            rows.get("portmap.bitnetd.sh") == 1,
            rows.get("portmap.i2sd.py") == -1,
        ]
        c1 = close([{"trit": 1, "key": "k", "note": "n"}])
        c2 = close([{"trit": 1, "key": "k", "note": "n"}])
        checks.append(c1["close"] == c2["close"] and len(c1["close"]) == 16)
        checks.append(pack_trits([1, 0, -1]) == bytes([2 + 3 * 1 + 9 * 0]))
        checks.append(not any(v for k, v in FLAGS.items() if k != "bind"))
        ok = all(checks)
        print(f"selftest {'pass' if ok else 'FAIL'} {sum(checks)}/{len(checks)}")
        return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    res = close(survey(ROOT), head_prev(ROOT))
    if "--json" in argv:
        print(json.dumps(res, indent=2))
    else:
        for r in res["rows"]:
            print(f"{r['trit']:+d} {r['key']}: {r['note']}")
        print(f"{res['item']}: {res['pass']} pass, {res['advisory']} advisory, "
              f"{res['fail']} fail, rate {RATE}, prev {res['prev']}, "
              f"close {res['close']}")
    return 1 if res["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
