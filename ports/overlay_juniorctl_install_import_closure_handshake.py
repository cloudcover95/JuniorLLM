#!/usr/bin/env python3
"""JuniorOS overlay: installed `juniorctl` import-closure handshake + port map.

Advances docs/BETA_TO_OS.md, section "JuniorOS overlay", the item
"`juniorctl` on PATH" on its install side: once install-overlay.sh (and
debian/stage_overlay.sh) have run, can the copy at DEST/usr/local/bin/juniorctl
actually answer `juniorctl health` / `juniorctl security` and hand off to
bitnetd 127.0.0.1:8765 and i2sd 127.0.0.1:8767?  The PATH pin itself
(profile.d, path.env), os-release.junior, Debian/Alpine, MemoryMax, user-unit
parity, tarball owner/mode, seccomp closure, sysctl and the i2sd install
closure already landed elsewhere and are not re-derived.  The skill-pin
family is owned by the overnight build (T13) and is not checked.

Angle: the Python import closure of the installed entry point.  This script
builds the virtual DEST tree from the `cp` / `cat >` lines of the two
installers (text only, nothing is run), then parses rails/linux/juniorctl.py
with `ast` (never imported) and asks, for the installed location:

  * where ROOT = parents[2] and LINUX = parent resolve
    (/usr and /usr/local/bin);
  * whether every module-level `from rails.linux... import` resolves on the
    sys.path the installed script gets (sys.path[0] = /usr/local/bin plus any
    PYTHONPATH the overlay sets) -- if not, juniorctl dies with
    ModuleNotFoundError before health() runs;
  * whether health()'s `ports.registry` resolves after _path() adds ROOT;
  * whether health()'s CONTAINER_SECURITY.md and security()'s
    LINUX/bitnetd.service and LINUX/seccomp-bitnetd.json exist in the tree;
  * that the same closure resolves in the fresh-clone tree (baseline);
  * that health() advertises bitnetd 127.0.0.1:8765, the hardened unit pins
    JUNIOR_BIND=127.0.0.1:8765, i2sd.py pins 127.0.0.1:8767, and nothing in
    juniorctl.py names 0.0.0.0.

Ports touched: juniorctl health -> bitnetd 127.0.0.1:8765 -> i2sd
127.0.0.1:8767.  Hand-off between Home, OSai, FieldCore, Flagstaff and
JuniorLLM is one trit pack (+1 pass, 0 advisory, -1 fail) closed with
sha256(prev || pack || text)[:16], rate 0.05.  In-process only: no sockets,
no subprocess, no shell run, no download, no model pull.  Stdlib only.

    python ports/overlay_juniorctl_install_import_closure_handshake.py
    python ports/overlay_juniorctl_install_import_closure_handshake.py --json
    python ports/overlay_juniorctl_install_import_closure_handshake.py --selftest

Exit 1 if any row is -1 (wildcard bind, port drift, juniorctl not installed
on PATH, baseline closure broken, or a flag flipped).  Install-tree gaps are
advisories (0), reported with the smallest additive fix.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import posixpath
import re
import shlex
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
}
PORTS = {"bitnetd": ("127.0.0.1", 8765), "i2sd": ("127.0.0.1", 8767)}
NODES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ITEM = "BETA_TO_OS.md / JuniorOS overlay / juniorctl on PATH (installed import closure)"
GENESIS = "0" * 16
INSTALLED = "/usr/local/bin/juniorctl"

SOURCES = {
    "install": f"{LINUX}/install-overlay.sh",
    "stage": f"{LINUX}/debian/stage_overlay.sh",
    "juniorctl": f"{LINUX}/juniorctl.py",
    "unit": f"{LINUX}/bitnetd.service",
    "i2sd": f"{LINUX}/i2sd.py",
}


# ---------------------------------------------------------------- load

def load(root: Path = ROOT) -> dict:
    s = {}
    for k, rel in SOURCES.items():
        p = root / rel
        s[k] = p.read_text(encoding="utf-8") if p.is_file() else ""
    s["repo_files"] = None  # filled lazily from the real tree
    s["root"] = str(root)
    return s


def repo_has(s: dict, rel: str) -> bool:
    if s.get("repo_files") is not None:
        return rel in s["repo_files"]
    return (Path(s["root"]) / rel).is_file()


# ---------------------------------------------------------------- virtual tree

def _strip(tok: str, var: str) -> str | None:
    for pre in (f"${var}", "${" + var + "}"):
        if tok.startswith(pre):
            return "/" + tok[len(pre):].lstrip("/")
    return None


def tree_from(script: str, var: str) -> set:
    """Destinations written by `cp` and `cat > ` lines (text parse only)."""
    out = set()
    for line in script.splitlines():
        line = line.strip()
        if line.startswith("cp "):
            try:
                toks = shlex.split(line)[1:]
            except ValueError:
                continue
            toks = [t for t in toks if not t.startswith("-")]
            if len(toks) < 2:
                continue
            dest = _strip(toks[-1], var)
            if dest is None:
                continue
            if toks[-1].endswith("/"):
                for src in toks[:-1]:
                    out.add(posixpath.join(dest, posixpath.basename(src)))
            else:
                out.add(dest)
        m = re.match(r'cat\s*>\s*"?([^"\s]+)"?', line)
        if m:
            dest = _strip(m.group(1), var)
            if dest:
                out.add(dest)
    return out


def installed_tree(s: dict) -> set:
    return tree_from(s["install"], "DEST") | tree_from(s["stage"], "OUT")


def overlay_pythonpath(s: dict) -> list:
    """PYTHONPATH entries the overlay itself sets (profile.d, units, path.env)."""
    hits = []
    for k in ("install", "stage", "unit"):
        for m in re.finditer(r"PYTHONPATH=\"?([^\"\s]+)", s[k]):
            for part in m.group(1).split(":"):
                if part.startswith("/") and part not in hits:
                    hits.append(part)
    return hits


# ---------------------------------------------------------------- ast

STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def _project(mod: str) -> bool:
    return mod.split(".")[0] not in STDLIB


def top_level_froms(tree: ast.Module) -> list:
    return [n.module for n in tree.body
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0
            and _project(n.module)]


def func(tree: ast.Module, name: str):
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def inner_froms(fn) -> list:
    if fn is None:
        return []
    return [n.module for n in ast.walk(fn)
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0
            and _project(n.module)]


def linux_files(fn) -> list:
    """String constants joined onto LINUX (LINUX / "x") inside a function."""
    out = []
    if fn is None:
        return out
    for n in ast.walk(fn):
        if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                and isinstance(n.left, ast.Name) and n.left.id == "LINUX"
                and isinstance(n.right, ast.Constant) and isinstance(n.right.value, str)):
            out.append(n.right.value)
    return out


def resolves(mod: str, dirs: list, has) -> bool:
    rel = mod.replace(".", "/")
    for d in dirs:
        for cand in (f"{rel}.py", f"{rel}/__init__.py"):
            if has(posixpath.join(d, cand)):
                return True
    return False


# ---------------------------------------------------------------- check

def check(s: dict) -> dict:
    rows = []

    def row(key, trit, note):
        rows.append({"key": key, "trit": trit, "note": note})

    src = s["juniorctl"]
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        row("juniorctl.parse", -1, f"juniorctl.py does not parse: {e}")
        return {"rows": rows, "fix": []}
    row("juniorctl.parse", 1, "rails/linux/juniorctl.py parsed with ast (not imported)")

    vt = installed_tree(s)
    on_path = INSTALLED in vt
    row("install.on_path", 1 if on_path else -1,
        f"install-overlay.sh writes {INSTALLED}" if on_path
        else f"{INSTALLED} not written by install-overlay.sh")
    exec_ok = 'chmod 0755' in s["install"] and '"$DEST/usr/local/bin/juniorctl"' in s["install"]
    row("install.exec_bit", 1 if exec_ok else 0, "chmod 0755 on installed juniorctl"
        if exec_ok else "no chmod 0755 for installed juniorctl")
    shebang = src.splitlines()[0] if src else ""
    row("install.shebang", 1 if shebang.startswith("#!") and "python3" in shebang else 0,
        f"shebang {shebang!r}")

    p_root = posixpath.dirname(posixpath.dirname(posixpath.dirname(INSTALLED)))
    p_linux = posixpath.dirname(INSTALLED)
    row("install.root_resolution", 0 if p_root == "/usr" else 1,
        f"ROOT=parents[2] -> {p_root}, LINUX=parent -> {p_linux} "
        "(not a JuniorLLM tree)" if p_root == "/usr" else f"ROOT -> {p_root}")

    pyp = overlay_pythonpath(s)
    base_dirs = [p_linux] + pyp
    vhas = lambda p: p in vt  # noqa: E731

    tops = top_level_froms(tree)
    missing_top = [m for m in tops if not resolves(m, base_dirs, vhas)]
    row("installed.module_level_imports", 0 if missing_top else 1,
        (f"{len(missing_top)}/{len(tops)} module-level imports unresolvable on "
         f"sys.path {base_dirs}: {missing_top}; juniorctl exits ModuleNotFoundError "
         "before health() runs") if missing_top
        else f"all {len(tops)} module-level imports resolve in the installed tree")

    h = func(tree, "health")
    sec = func(tree, "security")
    h_mods = inner_froms(h)
    h_missing = [m for m in h_mods if not resolves(m, base_dirs + [p_root], vhas)]
    row("installed.health_imports", 0 if h_missing else 1,
        f"health() imports {h_mods}; unresolvable after _path() adds {p_root}: {h_missing}"
        if h_missing else f"health() imports {h_mods} resolve")

    want = [("health", f) for f in linux_files(h)] + [("security", f) for f in linux_files(sec)]
    for fn_name, f in want:
        at = posixpath.join(p_linux, f)
        present = at in vt
        elsewhere = sorted(p for p in vt if posixpath.basename(p) == f)
        row(f"installed.{fn_name}.{f}", 1 if present else 0,
            f"{fn_name}() reads LINUX/{f} -> {at}: "
            + ("present" if present else
               "absent" + (f", installed at {elsewhere}" if elsewhere else ", not installed")))

    # baseline: same closure from a fresh clone
    rhas = lambda p: repo_has(s, p.lstrip("/"))  # noqa: E731
    repo_dirs = ["/"]
    base_missing = [m for m in tops + h_mods if not resolves(m, repo_dirs, rhas)]
    base_files = [f for _, f in want if not rhas(f"/{LINUX}/{f}")]
    row("baseline.fresh_clone_closure", -1 if (base_missing or base_files) else 1,
        f"fresh-clone tree misses modules {base_missing} files {base_files}"
        if (base_missing or base_files)
        else "fresh clone resolves every module-level, health() import and LINUX file")

    # ports
    hb, hp = PORTS["bitnetd"]
    adv = f"{hb}:{hp}"
    h_src = ast.get_source_segment(src, h) or "" if h else ""
    row("ports.health_bitnetd", 1 if f'"bitnetd": "{adv}"' in h_src else -1,
        f"health() advertises bitnetd {adv}")
    row("ports.unit_bind", 1 if f"JUNIOR_BIND={adv}" in s["unit"] else -1,
        f"bitnetd.service pins JUNIOR_BIND={adv}")
    ih, ip = PORTS["i2sd"]
    i2_ok = f'HOST = "{ih}"' in s["i2sd"] and f"PORT = {ip}" in s["i2sd"]
    row("ports.i2sd", 1 if i2_ok else -1, f"i2sd.py pins {ih}:{ip}")
    wild = "0.0.0.0" in src
    row("ports.no_wildcard", -1 if wild else 1,
        "juniorctl.py names 0.0.0.0" if wild else "juniorctl.py names no 0.0.0.0")

    # handshake map Home/OSai/FieldCore/Flagstaff/JuniorLLM
    reach = not missing_top and not h_missing
    for n in NODES:
        row(f"handoff.{n}", 1 if reach else 0,
            f"{n}: juniorctl health -> bitnetd {adv} -> i2sd {ih}:{ip} "
            + ("reachable from installed overlay" if reach
               else "reachable from a fresh clone only"))

    flags_ok = (FLAGS["bind"] == "127.0.0.1" and FLAGS["rate"] == 0.05 and not any(
        FLAGS[k] for k in ("mint", "socket", "layer_one", "fetch", "ue5_launch", "model_pull")))
    row("flags", 1 if flags_ok else -1,
        "mint socket layer_one fetch ue5_launch model_pull false, rate 0.05, bind 127.0.0.1")

    fix = []
    if missing_top or h_missing:
        fix.append("install-overlay.sh: stage rails/__init__.py, rails/linux/{juniorctl*,ctl_*}.py "
                   "and ports/registry.py under $DEST/usr/lib/junioros/ and install "
                   "/usr/local/bin/juniorctl as a sh wrapper: "
                   "PYTHONPATH=/usr/lib/junioros exec python3 /usr/lib/junioros/rails/linux/juniorctl.py \"$@\"")
    if any(r["key"].startswith(("installed.health.", "installed.security.")) and r["trit"] == 0
           for r in rows):
        fix.append("install-overlay.sh: cp bitnetd.service, seccomp-bitnetd.json and "
                   "CONTAINER_SECURITY.md next to the installed juniorctl.py (LINUX dir)")
    return {"rows": rows, "fix": fix, "installed_tree": sorted(vt), "pythonpath": pyp}


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

FIX_JCTL = '''#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from rails.linux.juniorctl_ops import *
LINUX = Path(__file__).resolve().parent
def health() -> dict:
    from ports.registry import list_ports
    return {"bitnetd": "127.0.0.1:8765", "security": str(LINUX / "CONTAINER_SECURITY.md")}
def security() -> dict:
    unit = (LINUX / "bitnetd.service").read_text()
    return {}
'''


def _fixture(installed_ok: bool) -> dict:
    unit = "[Service]\nEnvironment=JUNIOR_BIND=127.0.0.1:8765\n"
    i2 = 'HOST = "127.0.0.1"\nPORT = 8767\n'
    inst = 'cp "$HERE/juniorctl.py" "$DEST/usr/local/bin/juniorctl"\n' \
           'chmod 0755 "$DEST/usr/local/bin/juniorctl"\n'
    if installed_ok:
        inst += ('cp "$HERE/juniorctl_ops.py" "$DEST/usr/local/bin/rails/linux/juniorctl_ops.py"\n'
                 'cp "$HERE/registry.py" "$DEST/usr/ports/registry.py"\n'
                 'cp "$HERE/CONTAINER_SECURITY.md" "$HERE/bitnetd.service" "$DEST/usr/local/bin/"\n')
    return {"install": inst, "stage": "", "juniorctl": FIX_JCTL, "unit": unit, "i2sd": i2,
            "root": "/nonexistent",
            "repo_files": {"rails/linux/juniorctl_ops.py", "ports/registry.py",
                           "rails/linux/CONTAINER_SECURITY.md", "rails/linux/bitnetd.service"}}


def selftest() -> int:
    ok = True

    def t(cond, msg):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok = ok and cond

    good = report(_fixture(True))
    t(good["fail"] == 0 and good["advisory"] == 1,
      f"fixed install: only the /usr ROOT advisory ({good['pass']}/{good['advisory']}/{good['fail']})")
    bad = report(_fixture(False))
    keys0 = {r["key"] for r in bad["rows"] if r["trit"] == 0}
    t({"installed.module_level_imports", "installed.health_imports"} <= keys0 and bad["fail"] == 0,
      "bare install: module-level and health() imports flagged as advisories")
    t(any(k.startswith("handoff.") for k in keys0), "bare install: hand-off rows downgraded to 0")
    w = _fixture(True)
    w["juniorctl"] = FIX_JCTL.replace("127.0.0.1:8765", "0.0.0.0:8765")
    wr = report(w)
    t(wr["fail"] >= 2, "wildcard bind in juniorctl is a hard fail")
    d = _fixture(True)
    d["i2sd"] = 'HOST = "127.0.0.1"\nPORT = 9999\n'
    t(report(d)["fail"] == 1, "i2sd port drift is a hard fail")
    b = _fixture(True)
    b["repo_files"] = set()
    t(report(b)["fail"] >= 1, "broken fresh-clone baseline is a hard fail")
    t(report(_fixture(True))["close"] == good["close"], "close is deterministic")
    t(report(_fixture(True), prev="1" * 16)["close"] != good["close"], "close chains on prev")
    t(tree_from('cp "$HERE/a.py" "$HERE/b.py" "$OUT/usr/lib/x/"', "OUT")
      == {"/usr/lib/x/a.py", "/usr/lib/x/b.py"}, "multi-source cp into a dir is parsed")
    t(FLAGS["rate"] == 0.05 and not FLAGS["socket"] and not FLAGS["mint"], "flags pinned")
    real = report(load())
    t(real["fail"] == 0, f"real tree has no hard fail ({real['pass']}/{real['advisory']}/{real['fail']})")
    print("selftest", "pass" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    r = report(load(), prev=a.prev)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(r["item"])
        for row in r["rows"]:
            print(f"{row['trit']:+d} {row['key']}: {row['note']}")
        for f in r["fix"]:
            print("fix:", f)
        print(f"pass {r['pass']} advisory {r['advisory']} fail {r['fail']} close {r['close']}")
    return 1 if r["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
