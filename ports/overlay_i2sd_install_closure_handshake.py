#!/usr/bin/env python3
"""JuniorOS overlay: i2sd install-closure handshake + port map.

Advances docs/BETA_TO_OS.md, section "JuniorOS overlay", the item
"hardened `bitnetd.service`" on its fallback lane: once the overlay is
installed, does the unit's ExecStart chain actually reach a runnable i2sd on
127.0.0.1:8767 when bitnet.cpp / a GGUF is not on disk?  (os-release.junior,
PATH, Debian/Alpine, MemoryMax, user-unit parity, the no-kernel-blob receipt
and the GGUF on-disk gate already landed elsewhere and are not re-derived.)

Angle: the install closure.  rails/linux/KERNEL.md lists i2sd as part of the
userspace overlay, and bitnetd.sh falls back to `exec python3 "$HERE/i2sd.py"`.
This script statically builds the virtual DEST tree that
rails/linux/install-overlay.sh (and, separately, debian/stage_overlay.sh)
writes, then walks the chain in-process:

  bitnetd.service ExecStart -> DEST/usr/local/bin/junior-bitnetd (bitnetd.sh)
    -> $HERE resolved to the installed dir -> $HERE/i2sd.py present?
    -> PYTHONPATH default ($HERE/../..) and sys.path[0] -> i2sd's imports
       (junior_bitnet.i2s) resolvable inside the installed tree?
  plus the staged helpers (backend.py, bitnet_cpp.py) and their imports,
  bitnet_cpp.plan()'s fallback interpreter, and the port each lane binds
  versus what the unit advertises in JUNIOR_BIND.

Ports touched: bitnetd 127.0.0.1:8765 (bitnet.cpp lane, unit JUNIOR_BIND) and
i2sd 127.0.0.1:8767 (fallback lane).  Hand-off between Home, OSai, FieldCore,
Flagstaff and JuniorLLM is one trit pack (+1 pass, 0 advisory, -1 fail)
closed with sha256(prev || pack || text)[:16].  In-process only: no sockets,
no subprocess, no shell run, no download, no model pull.  Stdlib only.

    python ports/overlay_i2sd_install_closure_handshake.py
    python ports/overlay_i2sd_install_closure_handshake.py --json
    python ports/overlay_i2sd_install_closure_handshake.py --selftest

Exit 1 if any row is -1 (non-loopback bind, wildcard, missing ExecStart
wrapper, i2sd off 127.0.0.1:8767, or a flag flipped).  Install gaps are
advisories (0), reported with the smallest additive fix.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = "rails/linux"

FLAGS = {
    "bind": "127.0.0.1",
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
    "model_pull": False,
    "rate": 0.05,
    "pool": "held",
    "opt_in": True,
}
PORTS = {"bitnetd": ("127.0.0.1", 8765), "i2sd": ("127.0.0.1", 8767)}
ITEM = 'BETA_TO_OS.md / JuniorOS overlay / hardened bitnetd.service (i2sd fallback install closure)'
GENESIS = "0" * 16
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

SOURCES = {
    "install": f"{LINUX}/install-overlay.sh",
    "stage": f"{LINUX}/debian/stage_overlay.sh",
    "bitnetd_sh": f"{LINUX}/bitnetd.sh",
    "unit": f"{LINUX}/bitnetd.service",
    "i2sd": f"{LINUX}/i2sd.py",
    "bitnet_cpp": f"{LINUX}/bitnet_cpp.py",
    "backend": f"{LINUX}/backend.py",
    "kernel_md": f"{LINUX}/KERNEL.md",
}
# installed name -> repo source key, for walking imports of staged python
PY_SOURCE_BY_NAME = {"i2sd.py": "i2sd", "bitnet_cpp.py": "bitnet_cpp",
                     "backend.py": "backend"}

CP_RE = re.compile(r'^\s*cp\s+(?:-\w+\s+)*(.+?)\s*$')
HEREDOC_RE = re.compile(r'^\s*cat\s+>\s*"?\$(DEST|OUT)/([^"\s]+)"?\s*<<')
SUBINSTALL_RE = re.compile(r'^\s*sh\s+"?\$HERE/install-overlay\.sh"?')
EXEC_PY_RE = re.compile(r'^\s*exec\s+python3\s+"?\$HERE/([^"\s]+)"?')
EXEC_CPP_RE = re.compile(r'^\s*exec\s+.*--host\s+(\S+)\s+--port\s+(\d+)')
PYPATH_RE = re.compile(r'PYTHONPATH:-\$\(CDPATH=\s*cd\s+--\s+"\$HERE/([^"]+)"')
BIND_DEFAULT_RE = re.compile(r'JUNIOR_BIND:-([0-9a-zA-Z.:]+)\}')


# ---------------------------------------------------------------- loading

def load(root: Path) -> dict:
    out = {}
    for k, rel in SOURCES.items():
        p = root / rel
        out[k] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
    out["_pkgs"] = sorted(
        str(p.relative_to(root)) for p in (root / "junior_bitnet").glob("*.py")
    ) if (root / "junior_bitnet").is_dir() else []
    return out


def _code_lines(text: str):
    """Non-comment, non-heredoc-body lines of a POSIX sh script."""
    out, body_end = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if body_end is not None:
            if line == body_end:
                body_end = None
            continue
        if not line or line.startswith("#"):
            continue
        m = re.search(r"<<-?\s*'?\"?(\w+)'?\"?", line)
        if m:
            body_end = m.group(1)
        out.append(line)
    return out


def _unquote(tok: str) -> str:
    return tok.strip().strip('"').strip("'")


def manifest(script: str, var: str, base: dict | None = None) -> dict:
    """Virtual tree {'/abs/installed/path': 'src-name'} written by a script."""
    tree = dict(base or {})
    for line in _code_lines(script):
        m = HEREDOC_RE.match(line)
        if m and m.group(1) == var:
            tree["/" + m.group(2)] = "<heredoc>"
            continue
        m = CP_RE.match(line)
        if not m:
            continue
        toks = [_unquote(t) for t in re.findall(r'"[^"]*"|\S+', m.group(1))]
        if len(toks) < 2 or not toks[-1].startswith("$" + var + "/"):
            continue
        dst = "/" + toks[-1].split("/", 1)[1]
        srcs = toks[:-1]
        many = len(srcs) > 1 or dst.endswith("/")
        for src in srcs:
            name = src.rsplit("/", 1)[-1]
            target = dst.rstrip("/") + "/" + name if many else dst
            # `cp x "$OUT/usr/lib/junioros/"` style with one src and trailing dir
            tree[target] = name
    return tree


def unit_value(unit: str, key: str) -> list:
    vals = []
    for ln in unit.splitlines():
        ln = ln.strip()
        if ln.startswith(key + "="):
            vals.append(ln.split("=", 1)[1].strip())
    return vals


def unit_env(unit: str) -> dict:
    env = {}
    for v in unit_value(unit, "Environment"):
        if "=" in v:
            k, val = v.split("=", 1)
            env[k.strip()] = val.strip()
    return env


def py_consts(src: str) -> dict:
    out = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
    return out


def py_imports(src: str) -> list:
    """Absolute, non-stdlib modules a file imports at top level or in defs."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    std = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    out = []
    for m in mods:
        top = m.split(".")[0]
        if top in std or top in {"mlx", "torch", "numpy"}:
            continue  # optional accelerators are guarded by try/except upstream
        if m not in out:
            out.append(m)
    return out


def resolve_module(mod: str, roots: list, tree: dict) -> str | None:
    parts = mod.split(".")
    for r in roots:
        cand_file = posixpath.join(r, *parts) + ".py"
        cand_pkg = posixpath.join(r, *parts, "__init__.py")
        if cand_file in tree or cand_pkg in tree:
            return cand_file if cand_file in tree else cand_pkg
    return None


def string_literals(src: str) -> list:
    try:
        return [n.value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    except SyntaxError:
        return []


# ---------------------------------------------------------------- checks

def check(s: dict) -> dict:
    rows = []

    def row(key, trit, note):
        rows.append({"key": key, "trit": int(trit), "note": note})

    install_tree = manifest(s["install"], "DEST")
    stage_base = install_tree if any(SUBINSTALL_RE.match(l) for l in _code_lines(s["stage"])) else {}
    stage_tree = manifest(s["stage"], "OUT", stage_base)

    # --- wildcard / loopback guards (hard)
    wild = []
    for k in ("install", "stage", "bitnetd_sh"):
        # refusal guards (grep -Fq '0.0.0.0' ... exit 3) are not binds
        if any("0.0.0.0" in ln and "grep" not in ln for ln in _code_lines(s[k])):
            wild.append(k)
    if any("0.0.0.0" in ln.split("#", 1)[0] for ln in s["unit"].splitlines()):
        wild.append("unit")
    for k in ("i2sd", "bitnet_cpp", "backend"):
        if "0.0.0.0" in string_literals(s[k]):
            wild.append(k)
    row("bind.no_wildcard", -1 if wild else 1,
        "no 0.0.0.0 in overlay chain" if not wild else f"0.0.0.0 in {wild}")

    env = unit_env(s["unit"])
    ub = env.get("JUNIOR_BIND", "")
    u_host, _, u_port = ub.rpartition(":")
    row("unit.bind_loopback", 1 if u_host in LOOPBACK and u_port == "8765" else -1,
        f"bitnetd.service JUNIOR_BIND={ub or '<unset>'}")

    # --- ExecStart -> installed wrapper (hard)
    execs = unit_value(s["unit"], "ExecStart")
    exec_path = execs[0].split()[0] if execs else ""
    wrapper_src = install_tree.get(exec_path)
    row("unit.execstart_installed", 1 if wrapper_src == "bitnetd.sh" else -1,
        f"ExecStart={exec_path or '<none>'} -> {wrapper_src or 'not installed'}")

    # --- bitnet.cpp lane: 127.0.0.1:8765, only with binary + GGUF on disk
    cpp = None
    for ln in _code_lines(s["bitnetd_sh"]):
        m = EXEC_CPP_RE.match(ln)
        if m:
            cpp = (m.group(1), int(m.group(2)))
    row("lane.cpp_port", 1 if cpp == PORTS["bitnetd"] else -1,
        f"bitnetd.sh cpp exec {cpp[0]}:{cpp[1]}" if cpp else "no cpp exec line")
    shipped_bin = [p for p in stage_tree if re.search(r"(bitnet-cpp|llama-cli|\.gguf)$", p)]
    row("lane.cpp_not_shipped", 1 if not shipped_bin else -1,
        "overlay ships no bitnet.cpp binary or GGUF (no model pull)"
        if not shipped_bin else f"shipped {shipped_bin}")

    # --- i2sd listen port (hard)
    c = py_consts(s["i2sd"])
    i2 = (c.get("HOST"), c.get("PORT"))
    row("lane.i2sd_port", 1 if i2 == PORTS["i2sd"] else -1,
        f"i2sd.py HOST={i2[0]} PORT={i2[1]}")

    # --- fallback exec target resolution against the installed tree
    wrapper_dir = posixpath.dirname(exec_path) if exec_path else "/usr/local/bin"
    fb = None
    for ln in _code_lines(s["bitnetd_sh"]):
        m = EXEC_PY_RE.match(ln)
        if m:
            fb = posixpath.join(wrapper_dir, m.group(1))
    i2sd_where_install = sorted(p for p, src in install_tree.items() if src == "i2sd.py")
    i2sd_where_stage = sorted(p for p, src in stage_tree.items() if src == "i2sd.py")
    row("fallback.target_in_install", 1 if fb and fb in install_tree else 0,
        f"junior-bitnetd execs {fb}; install-overlay.sh installs i2sd.py at "
        f"{i2sd_where_install or 'nowhere'}")
    row("fallback.target_in_stage", 1 if fb and fb in stage_tree else 0,
        f"junior-bitnetd execs {fb}; stage_overlay.sh puts i2sd.py at "
        f"{i2sd_where_stage or 'nowhere'}")

    # --- PYTHONPATH default + sys.path[0] -> i2sd imports
    m = PYPATH_RE.search(s["bitnetd_sh"])
    pp = posixpath.normpath(posixpath.join(wrapper_dir, m.group(1))) if m else None
    if env.get("PYTHONPATH"):
        pp = env["PYTHONPATH"].split(":")[0]  # bitnetd.sh honours ${PYTHONPATH:-...}
    roots = [r for r in (posixpath.dirname(fb) if fb else None, pp) if r]
    i2_imports = py_imports(s["i2sd"])
    for tree_name, tree in (("install", install_tree), ("stage", stage_tree)):
        missing = [mod for mod in i2_imports if not resolve_module(mod, roots, tree)]
        row(f"fallback.imports_{tree_name}", 1 if not missing else 0,
            f"PYTHONPATH default {pp}, sys.path[0] {roots[0] if roots else None}; "
            + ("all i2sd imports resolve" if not missing else
               f"unresolved in {tree_name} tree: {missing}"))

    # --- staged helper python and its imports (stage tree only)
    staged_py = sorted(p for p, src in stage_tree.items()
                       if src in PY_SOURCE_BY_NAME and p.startswith("/usr/lib/junioros/"))
    unresolved = {}
    for p in staged_py:
        src = s[PY_SOURCE_BY_NAME[stage_tree[p]]]
        miss = [mod for mod in py_imports(src)
                if not resolve_module(mod, [posixpath.dirname(p)], stage_tree)]
        if miss:
            unresolved[posixpath.basename(p)] = miss
    row("stage.helpers_import_closure", 1 if not unresolved else 0,
        "staged helpers import only staged modules" if not unresolved
        else f"staged /usr/lib/junioros helpers import outside the tree: {unresolved}")

    # --- bitnet_cpp.plan() fallback interpreter vs bitnetd.sh shebang
    lits = string_literals(s["bitnet_cpp"])
    fb_cmd = next((x for x in lits if "bitnetd.sh" in x), "")
    shebang = s["bitnetd_sh"].splitlines()[0] if s["bitnetd_sh"] else ""
    mismatch = fb_cmd.startswith("python3") and shebang.startswith("#!/bin/sh")
    row("plan.fallback_interpreter", 0 if mismatch else 1,
        f"bitnet_cpp.plan fallback '{fb_cmd}' vs bitnetd.sh shebang '{shebang}'"
        + (" (python3 cannot run a sh script)" if mismatch else ""))

    # --- port the unit advertises vs port the fallback lane listens on
    reads_bind = "JUNIOR_BIND" in s["i2sd"]
    drift = (u_port == str(PORTS["bitnetd"][1]) and i2 == PORTS["i2sd"]
             and not reads_bind)
    row("ports.unit_vs_fallback", 0 if drift else 1,
        "fallback lane listens 127.0.0.1:8767 and ignores JUNIOR_BIND, while the "
        "unit advertises 127.0.0.1:8765 (clients of the unit see nothing on 8765)"
        if drift else "fallback lane honours the unit's JUNIOR_BIND or matches it")

    # --- KERNEL.md documented overlay vs installed tree
    doc = s["kernel_md"]
    listed = [n for n in ("i2sd", "juniorctl", "os_route", "bitnet.cpp")
              if re.search(r"[├└]\s*" + re.escape(n) + r"\b", doc)]
    row("doc.kernel_md_lists_i2sd", 1 if "i2sd" in listed else -1,
        f"KERNEL.md userspace overlay lists {listed}")
    names_install = {posixpath.basename(p) for p in install_tree}
    names_stage = {posixpath.basename(p) for p in stage_tree}
    for n in ("i2sd", "juniorctl", "os_route"):
        if n not in listed:
            continue
        hit_i = any(x.startswith(n) for x in names_install)
        hit_s = any(x.startswith(n) for x in names_stage)
        row(f"doc_vs_install.{n}", 1 if hit_i else 0,
            f"{n}: install-overlay.sh {'yes' if hit_i else 'no'}, "
            f"stage_overlay.sh {'yes' if hit_s else 'no'}")

    # --- flags (hard)
    flags_ok = (FLAGS["bind"] == "127.0.0.1" and FLAGS["rate"] == 0.05 and not any(
        FLAGS[k] for k in ("mint", "socket", "layer_one", "fetch", "ue5_launch", "model_pull")))
    row("flags", 1 if flags_ok else -1,
        "mint socket layer_one fetch ue5_launch model_pull false, rate 0.05, bind 127.0.0.1")

    fix = []
    if any(r["key"] == "fallback.target_in_install" and r["trit"] == 0 for r in rows):
        fix.append(f'install-overlay.sh: cp "$HERE/i2sd.py" "$DEST{fb}"')
    if any(r["key"] == "fallback.imports_install" and r["trit"] == 0 for r in rows):
        fix.append("install-overlay.sh: stage junior_bitnet/{__init__,i2s,math}.py under "
                   "$DEST/usr/lib/junioros/junior_bitnet/ and set "
                   "Environment=PYTHONPATH=/usr/lib/junioros in both bitnetd units")
    if mismatch:
        fix.append("bitnet_cpp.plan: fallback 'sh rails/linux/bitnetd.sh'")
    return {"rows": rows, "fix": fix,
            "install_tree": sorted(install_tree), "stage_tree": sorted(stage_tree)}


# ---------------------------------------------------------------- close

def pack(rows: list) -> bytes:
    return bytes(r["trit"] + 1 for r in rows)


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def report(s: dict, prev: str = GENESIS) -> dict:
    res = check(s)
    rows = res["rows"]
    text = "\n".join(f"{r['trit']:+d} {r['key']} {r['note']}" for r in rows)
    pk = pack(rows)
    return {
        "item": ITEM,
        "ports": {k: f"{h}:{p}" for k, (h, p) in PORTS.items()},
        "flags": FLAGS,
        "rows": rows,
        "fix": res["fix"],
        "pass": sum(r["trit"] == 1 for r in rows),
        "advisory": sum(r["trit"] == 0 for r in rows),
        "fail": sum(r["trit"] == -1 for r in rows),
        "pack": pk.hex(),
        "prev": prev,
        "close": close(prev, pk, text),
    }


# ---------------------------------------------------------------- selftest

def _fixed(s: dict) -> dict:
    """Same tree with the additive install fixes applied in memory."""
    g = dict(s)
    g["install"] = s["install"].replace(
        'cp "$HERE/juniorctl.py"',
        'cp "$HERE/i2sd.py" "$DEST/usr/local/bin/i2sd.py"\n'
        'cp "$HERE/../../junior_bitnet/__init__.py" '
        '"$DEST/usr/lib/junioros/junior_bitnet/__init__.py"\n'
        'cp "$HERE/../../junior_bitnet/i2s.py" "$DEST/usr/lib/junioros/junior_bitnet/i2s.py"\n'
        'cp "$HERE/juniorctl.py"')
    g["unit"] = s["unit"].replace(
        "Environment=JUNIOR_OFFLINE=1",
        "Environment=JUNIOR_OFFLINE=1\nEnvironment=PYTHONPATH=/usr/lib/junioros")
    return g


def selftest() -> int:
    ok = True
    s = load(ROOT)
    real = report(s)
    if real["fail"]:
        print("selftest: real tree has fail rows", file=sys.stderr)
        ok = False
    keys = {r["key"]: r["trit"] for r in real["rows"]}

    good = {r["key"]: r["trit"] for r in report(_fixed(s))["rows"]}
    for k in ("fallback.target_in_install", "fallback.imports_install"):
        if good.get(k) != 1:
            print(f"selftest: fixed install did not pass {k} ({good.get(k)})", file=sys.stderr)
            ok = False

    bad = dict(s)
    bad["i2sd"] = s["i2sd"].replace('"127.0.0.1"', '"0.0.0.0"', 1)
    bad["unit"] = s["unit"].replace("ExecStart=/usr/local/bin/junior-bitnetd",
                                    "ExecStart=/opt/missing/bitnetd")
    bk = {r["key"]: r["trit"] for r in report(bad)["rows"]}
    for k in ("bind.no_wildcard", "lane.i2sd_port", "unit.execstart_installed"):
        if bk.get(k) != -1:
            print(f"selftest: bad fixture did not fail {k} ({bk.get(k)})", file=sys.stderr)
            ok = False

    a, b = report(s, GENESIS), report(s, GENESIS)
    c2 = report(s, "f" * 16)
    if a["close"] != b["close"] or a["close"] == c2["close"] or len(a["close"]) != 16:
        print("selftest: close rule", file=sys.stderr)
        ok = False
    if len(bytes.fromhex(a["pack"])) != len(a["rows"]):
        print("selftest: pack length", file=sys.stderr)
        ok = False
    if FLAGS["rate"] != 0.05 or FLAGS["socket"] or FLAGS["mint"] or FLAGS["fetch"]:
        print("selftest: flags", file=sys.stderr)
        ok = False
    print(f"selftest {'pass' if ok else 'FAIL'}: real {real['pass']} pass / "
          f"{real['advisory']} advisory / {real['fail']} fail, close {real['close']}, "
          f"rows {len(keys)}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prev", default=GENESIS, help="previous close (16 hex)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    r = report(load(Path(a.root)), a.prev)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(ITEM)
        for row in r["rows"]:
            print(f"{row['trit']:+d} {row['key']}: {row['note']}")
        for f in r["fix"]:
            print(f"fix: {f}")
        print(f"{r['pass']} pass, {r['advisory']} advisory, {r['fail']} fail; "
              f"close {r['close']}")
    return 1 if r["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
