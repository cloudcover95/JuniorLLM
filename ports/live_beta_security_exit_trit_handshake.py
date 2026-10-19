"""Live beta security exit-trit handshake: repo layout vs installed overlay.

Advances JuniorLLM docs/BETA_TO_OS.md, item "Live beta", the end-to-end
exit-code trits plus `tests/test_linux_security.py` (step 3, with the
`juniorctl security` gate it leans on). Steps 2/4/5/6, night, ask and health
are out of scope here; ports/live_beta_exit_code_port_map.py already folds
ctl_cli.run's exit table, so this file does not redo that. It answers a
different question: does the exit code a Home/OSai/FieldCore/Flagstaff/
JuniorLLM caller sees still mean "secure" once you leave the repo checkout?

It reads rails/linux/{juniorctl.py, ctl_cli.py, install-overlay.sh,
bitnetd.service, seccomp-bitnetd.json, i2sd.py, oci/config.json,
oci/README.md} and tests/test_linux_security.py as plain text (ast / json /
regex). Nothing from the repo is imported or executed. No subprocess, no
socket, no network, no model pull.

Rows, one trit each (+1 holds, 0 advisory / not exercised, -1 broken):
  A. Artifact trail for each test_linux_security case: the file the case
     reads in the repo, and where install-overlay.sh puts it (or not).
  B. Exit-code sources that share code 1 on the security path, so a 1 from
     `juniorctl security` can't be read as "gate failed" end to end.
  C. install-overlay.sh exit codes as trits (0 +1, 2 usage -1, 3 refusal -1,
     set -eu step failure -1) and its loopback guard vs the 127.0.0.1 rule.
  D. seccomp wiring: OCI config.json, the podman line in oci/README.md
     resolved from both layouts, archMap coverage, --network=none vs the
     host reaching bitnetd 127.0.0.1:8765.
  E. i2sd 127.0.0.1:8767: bind constants, and whether any security exit
     gate covers it (it doesn't, so its trit is 0, not +1).
  F. One port row each for Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Hard invariants (-1, exit 1): a 0.0.0.0 bind, a seccomp defaultAction
that allows, a HARDENING key missing from bitnetd.service, a flag set true.

Flags: mint false, socket false, layer_one false, fetch false,
ue5_launch false. Rate 0.05. Bind 127.0.0.1. Skill-pin family (T13) is
owned by the overnight build and skipped. Trits are 2-bit packed
(0->0, 1->1, -1->2, MSB first) and chained with
close = sha256(prev || pack || text)[:16]. Stdlib only.

Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM;
bitnetd 127.0.0.1:8765, i2sd 127.0.0.1:8767.

Run: python ports/live_beta_security_exit_trit_handshake.py [--selftest] [--json]
Exit 0 when no row is -1, 1 otherwise.
"""
from __future__ import annotations

import ast
import hashlib
import json
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"
TESTS = ROOT / "tests"

LOOPBACK = "127.0.0.1"
BITNETD = f"{LOOPBACK}:8765"
I2SD = f"{LOOPBACK}:8767"
RATE = 0.05
GENESIS = "0" * 16
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md Live beta: end-to-end exit-code trits "
    "plus tests/test_linux_security.py"
)
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False,
         "ue5_launch": False}
ALLOW_ACTIONS = {"SCMP_ACT_ALLOW", "SCMP_ACT_LOG"}
DENY_ACTIONS = {"SCMP_ACT_ERRNO", "SCMP_ACT_KILL", "SCMP_ACT_KILL_PROCESS",
                "SCMP_ACT_KILL_THREAD", "SCMP_ACT_TRAP"}
# Repo path of each artifact -> path install-overlay.sh would give it, filled
# from the installer's cp lines at run time.
REPO_REL = "rails/linux"


# ---------------------------------------------------------------- close rule
def pack_trits(trits: list[int]) -> bytes:
    code = {0: 0, 1: 1, -1: 2}
    out, acc, n = bytearray(), 0, 0
    for t in trits:
        acc = (acc << 2) | code[t]
        n += 1
        if n == 4:
            out.append(acc)
            acc, n = 0, 0
    if n:
        out.append(acc << (2 * (4 - n)))
    return bytes(out)


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- parsers
def juniorctl_facts(src: str) -> dict:
    """HARDENING, LINUX anchor, and module-level `from rails...` imports."""
    mod = ast.parse(src)
    hardening: list[str] = []
    linux_anchor = None
    top_imports: list[str] = []
    for node in mod.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "HARDENING" and isinstance(node.value, (ast.Tuple, ast.List)):
                hardening = [e.value for e in node.value.elts
                             if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if name == "LINUX":
                linux_anchor = ast.unparse(node.value)
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.split(".")[0] == "rails":
            top_imports.append(node.module)
    found: list[tuple[int, int, str]] = []
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == "security":
            for sub in ast.walk(node):
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Div) \
                        and isinstance(sub.right, ast.Constant):
                    found.append((sub.lineno, sub.col_offset, str(sub.right.value)))
    sec_reads = [name for _, _, name in sorted(found)]
    return {"hardening": hardening, "linux_anchor": linux_anchor,
            "top_imports": top_imports, "security_reads": sec_reads}


def gate_keys(ctl_src: str) -> list[str]:
    """Keys the `security` branch of ctl_cli.run gates its exit on."""
    mod = ast.parse(ctl_src)
    for node in ast.walk(mod):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) \
                and any(isinstance(c, ast.Constant) and c.value == "security"
                        for c in node.test.comparators):
            for st in node.body:
                if isinstance(st, ast.Return) and isinstance(st.value, ast.IfExp):
                    return [s.slice.value for s in ast.walk(st.value.test)
                            if isinstance(s, ast.Subscript)
                            and isinstance(s.slice, ast.Constant)]
    return []


def test_artifacts(test_src: str) -> dict:
    """Each test case -> files it reads under LINUX, and whether it calls security()."""
    mod = ast.parse(test_src)
    out: dict = {}
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            files, calls_sec = [], False
            for sub in ast.walk(node):
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Div) \
                        and isinstance(sub.left, ast.Name) and sub.left.id == "LINUX" \
                        and isinstance(sub.right, ast.Constant):
                    files.append(sub.right.value)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                        and sub.func.id == "security":
                    calls_sec = True
            out[node.name] = {"files": files, "calls_security": calls_sec}
    return out


_CP = re.compile(r'^\s*cp\s+"\$HERE/([^"]+)"\s+"\$DEST/([^"]+)"', re.M)
_EXIT = re.compile(r"\bexit\s+(\d+)")
_CASE = re.compile(r'case\s+"\$host"\s+in\s*\n\s*([^\n)]+)\)', re.M)


def installer_facts(sh: str) -> dict:
    cps = {src: dst for src, dst in _CP.findall(sh)}
    exits = sorted({int(x) for x in _EXIT.findall(sh)})
    m = _CASE.search(sh)
    allowed = [h.strip() for h in m.group(1).split("|")] if m else []
    bind_default = re.search(r"JUNIOR_BIND:-([0-9.:]+)", sh)
    return {"cp": cps, "exits": exits, "set_eu": bool(re.search(r"^set -eu", sh, re.M)),
            "loopback_case": allowed,
            "bind_default": bind_default.group(1) if bind_default else None,
            "wildcard_guard": "0.0.0.0" in sh and "refusing" in sh}


def i2sd_facts(src: str) -> dict:
    mod = ast.parse(src)
    host = port = None
    sets: list[list[str]] = []
    for node in ast.walk(mod):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant):
            if node.targets[0].id == "HOST":
                host = node.value.value
            if node.targets[0].id == "PORT":
                port = node.value.value
        if isinstance(node, ast.Set):
            vals = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if vals:
                sets.append(sorted(vals))
    return {"host": host, "port": port, "allow_sets": sets}


def readme_seccomp_ref(md: str) -> str | None:
    m = re.search(r"seccomp=(\S+)", md)
    return m.group(1) if m else None


def parse_unit(text: str) -> dict:
    return {"text": text, "wildcard": "0.0.0.0" in text,
            "bind": BITNETD in text}


# ---------------------------------------------------------------- build
def build(src: dict) -> list[dict]:
    rows: list[dict] = []

    def row(name: str, trit: int, detail: str) -> None:
        rows.append({"row": name, "trit": trit, "detail": detail})

    jc = juniorctl_facts(src["juniorctl"])
    gk = gate_keys(src["ctl_cli"])
    cases = test_artifacts(src["test_sec"])
    inst = installer_facts(src["installer"])
    unit = parse_unit(src["unit"])
    sec = json.loads(src["seccomp"]) if src["seccomp"] else {}
    oci = json.loads(src["oci_config"]) if src["oci_config"] else {}
    i2 = i2sd_facts(src["i2sd"])
    ref = readme_seccomp_ref(src["oci_readme"])

    # flags + hard invariants
    for k, v in FLAGS.items():
        row(f"flag {k}", 1 if v is False else -1, f"{k}={str(v).lower()}")
    row("rate", 1 if RATE == 0.05 else -1, f"rate {RATE}")
    missing = [k for k in jc["hardening"] if k not in unit["text"]]
    row("unit HARDENING present", -1 if missing or not jc["hardening"] else 1,
        f"{len(jc['hardening'])} keys, missing {missing or 'none'}")
    row("unit bind loopback", -1 if unit["wildcard"] or not unit["bind"] else 1,
        f"bitnetd.service carries JUNIOR_BIND={BITNETD}, no 0.0.0.0")
    action = sec.get("defaultAction")
    row("seccomp defaultAction denies",
        1 if action in DENY_ACTIONS else -1,
        f"defaultAction={action}")
    row("security gate keys", 1 if gk else -1,
        f"ctl_cli security exits 0 iff {' and '.join(gk) or '?'}")

    # A. artifact trail per test case, repo layout vs installed layout
    linux_dir = f"{REPO_REL}/"
    for case, info in sorted(cases.items()):
        for f in info["files"]:
            dst = inst["cp"].get(f)
            present = f in src["linux_files"]
            if not present:
                row(f"A {case} {f}", -1, "missing in repo layout")
                continue
            if dst:
                row(f"A {case} {f}", 1,
                    f"repo {linux_dir}{f} -> installed /{dst}")
            elif f == "install-overlay.sh":
                row(f"A {case} {f}", 1,
                    f"repo {linux_dir}{f} present; the installer runs from the "
                    f"checkout and is not meant to be installed")
            else:
                row(f"A {case} {f}", 0,
                    f"repo {linux_dir}{f} present; install-overlay.sh never "
                    f"copies it, so the installed overlay has no copy for this case")
        if info["calls_security"]:
            row(f"A {case} via security()", 1,
                f"reads {', '.join(jc['security_reads'])} relative to "
                f"LINUX = {jc['linux_anchor']}")

    # B. exit-code 1 collision on the security path
    installed_ctl = inst["cp"].get("juniorctl.py")
    anchor_is_file_dir = (jc["linux_anchor"] or "").endswith(".parent")
    sources = ["ctl_cli gate false (unit_ok/seccomp)"]
    if jc["top_imports"]:
        sources.append("ModuleNotFoundError from module-level "
                       f"`from {jc['top_imports'][0]} import` outside a checkout")
    if anchor_is_file_dir and installed_ctl:
        inst_dir = posixpath.dirname("/" + installed_ctl)
        unread = [r for r in jc["security_reads"]
                  if posixpath.join(inst_dir, r) != "/" + inst["cp"].get(r, "")]
        if unread:
            verb = "is" if len(unread) == 1 else "are"
            sources.append(f"FileNotFoundError: LINUX resolves to {inst_dir}, "
                           f"where {' and '.join(unread)} {verb} not installed")
    sources.append("unittest failure or import error in test_linux_security")
    row("B repo-layout security exit", 1,
        "PYTHONPATH=. rails/linux/juniorctl.py security: files beside LINUX, "
        "gate decides, predicted exit 0 -> +1")
    row("B installed security exit", 0 if len(sources) > 2 else 1,
        f"/{installed_ctl} security: exit 1 has {len(sources)} sources "
        f"({'; '.join(sources)}); predicted 1 before the gate runs, so it reads 0 "
        "(environment), not -1 (insecure)")
    ungated = [c for c, i in cases.items()
               if not i["calls_security"]]
    row("B test cases outside gate", 0 if ungated else 1,
        f"{len(ungated)} of {len(cases)} cases can fail while the gate exits 0: "
        f"{', '.join(sorted(ungated)) or 'none'}")

    # C. installer exit trits + loopback guard
    trit_of = {0: (1, "installed"), 1: (-1, "set -eu step failed"),
               2: (-1, "usage (DEST unset)"), 3: (-1, "refused (non-loopback or wildcard/docker.sock)")}
    codes = list(inst["exits"]) + ([1] if inst["set_eu"] else [])
    for c in sorted(set(codes)):
        t, why = trit_of.get(c, (-1, "unmapped nonzero"))
        row(f"C installer exit {c} mapped", 1 if c in trit_of else 0,
            f"exit {c} -> trit {t:+d} ({why})")
    row("C installer bind default",
        1 if inst["bind_default"] == BITNETD else -1,
        f"JUNIOR_BIND default {inst['bind_default']}")
    extra = [h for h in inst["loopback_case"] if h != LOOPBACK]
    row("C installer loopback guard", 0 if extra else 1,
        f"case admits {inst['loopback_case']}; {extra} pass the guard though the "
        f"rule is {LOOPBACK} only" if extra else f"admits {LOOPBACK} only")
    row("C installer wildcard guard", 1 if inst["wildcard_guard"] else 0,
        "post-copy grep refuses 0.0.0.0 with exit 3")

    # D. seccomp wiring
    row("D oci config.json seccomp",
        1 if "seccomp" in oci.get("linux", {}) else 0,
        "linux.seccomp present" if "seccomp" in oci.get("linux", {}) else
        "config.json has no linux.seccomp; runc/crun on the bundle alone run "
        "without the profile, only the podman --security-opt line applies it")
    if ref:
        repo_res = posixpath.normpath(posixpath.join(f"{REPO_REL}/oci", ref))
        oci_dst = inst["cp"].get("oci/README.md", "")
        inst_res = posixpath.normpath(posixpath.join(posixpath.dirname(oci_dst), ref)) \
            if oci_dst else None
        sec_dst = inst["cp"].get("seccomp-bitnetd.json")
        row("D README seccomp ref repo", 1 if repo_res == f"{REPO_REL}/seccomp-bitnetd.json" else 0,
            f"{ref} from {REPO_REL}/oci -> {repo_res}")
        if inst_res is not None:
            row("D README seccomp ref installed", 1 if inst_res == sec_dst else 0,
                f"{ref} from /{posixpath.dirname(oci_dst)} -> /{inst_res}; "
                f"profile installed at /{sec_dst}")
    arches = [a.get("architecture") for a in sec.get("archMap", [])]
    row("D seccomp archMap", 1 if "SCMP_ARCH_AARCH64" in arches else 0,
        f"archMap {arches}; aarch64 hosts (Asahi) get no arch-specific map")
    names = [n for s in sec.get("syscalls", []) for n in s.get("names", [])]
    needed = ["socket", "bind", "listen", "accept4"]
    lacking = [n for n in needed if n not in names]
    row("D seccomp allows loopback serve", -1 if lacking else 1,
        f"allow list has {', '.join(needed)}" if not lacking else f"missing {lacking}")
    dupes = sorted({n for n in names if names.count(n) > 1})
    row("D seccomp duplicate names", 0 if dupes else 1,
        f"duplicates {dupes}" if dupes else "none")
    net_none = "--network=none" in src["oci_readme"]
    row("D podman network vs 8765", 0 if net_none else 1,
        f"--network=none isolates the container loopback, so Home/OSai on the "
        f"host can't reach bitnetd {BITNETD} from the rootless run"
        if net_none else "no network isolation flag")

    # E. i2sd
    i2_ok = i2["host"] == LOOPBACK and i2["port"] == 8767
    row("E i2sd bind", 1 if i2_ok else -1, f"HOST={i2['host']} PORT={i2['port']}")
    wide = sorted({h for s in i2["allow_sets"] for h in s if h != LOOPBACK})
    row("E i2sd loopback sets", 0 if wide else 1,
        f"allow sets also admit {wide}" if wide else "127.0.0.1 only")
    covered = any("i2sd" in r for r in jc["security_reads"]) or "i2sd.py" in inst["cp"]
    row("E i2sd security gate", 1 if covered else 0,
        f"no security() read, gate key, seccomp profile or installer copy for "
        f"i2sd {I2SD}; its security exit trit is 0 (not exercised)"
        if not covered else "covered")

    # F. port rows
    a_rows = [r for r in rows if r["row"].startswith(("A ", "B ", "D ", "E "))]
    worst = min((r["trit"] for r in a_rows), default=1)
    for port in PORTS:
        row(f"F port {port}", worst if worst < 0 else 1,
            f"{port} reads `juniorctl security` exit against bitnetd {BITNETD} "
            f"(gated) and i2sd {I2SD} (ungated); repo layout +1, installed 0")
    return rows


def chain(rows: list[dict]) -> str:
    prev = GENESIS
    for r in rows:
        text = f"{r['row']}|{r['trit']}|{r['detail']}"
        prev = close(prev, pack_trits([r["trit"]]), text)
        r["close"] = prev
    return prev


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def live_sources(root: Path = ROOT) -> dict:
    linux = root / "rails" / "linux"
    return {
        "juniorctl": _read(linux / "juniorctl.py"),
        "ctl_cli": _read(linux / "ctl_cli.py"),
        "test_sec": _read(root / "tests" / "test_linux_security.py"),
        "installer": _read(linux / "install-overlay.sh"),
        "unit": _read(linux / "bitnetd.service"),
        "seccomp": _read(linux / "seccomp-bitnetd.json"),
        "oci_config": _read(linux / "oci" / "config.json"),
        "oci_readme": _read(linux / "oci" / "README.md"),
        "i2sd": _read(linux / "i2sd.py"),
        "linux_files": sorted(p.name for p in linux.iterdir() if p.is_file())
        if linux.is_dir() else [],
    }


# ---------------------------------------------------------------- selftest
FIX = {
    "juniorctl": (
        "from pathlib import Path\n"
        "LINUX = Path(__file__).resolve().parent\n"
        "HARDENING = ('NoNewPrivileges=yes', 'JUNIOR_BIND=127.0.0.1:8765')\n"
        "def security():\n"
        "    unit = (LINUX / 'bitnetd.service').read_text()\n"
        "    s = LINUX / 'seccomp-bitnetd.json'\n"
        "    return {'unit_ok': True, 'seccomp': s.is_file()}\n"
        "from rails.linux.juniorctl_ops import *\n"
    ),
    "ctl_cli": (
        "def run(argv, ns):\n"
        "    cmd = argv[1]\n"
        "    if cmd == 'security':\n"
        "        report = ns['security']()\n"
        "        return 0 if report['unit_ok'] and report['seccomp'] else 1\n"
        "    return 2\n"
    ),
    "test_sec": (
        "class T:\n"
        "    def test_unit_hardened(self):\n"
        "        r = security()\n"
        "    def test_seccomp_default_deny(self):\n"
        "        x = (LINUX / 'seccomp-bitnetd.json')\n"
        "    def test_installer_exists(self):\n"
        "        a = (LINUX / 'install-overlay.sh'); b = (LINUX / 'CONTAINER_SECURITY.md')\n"
    ),
    "installer": (
        "set -eu\nBIND=${JUNIOR_BIND:-127.0.0.1:8765}\n"
        "case \"$host\" in\n  127.0.0.1|::1)\n    ;;\n  *)\n    exit 3\nesac\n"
        "[ -n \"$DEST\" ] || exit 2\n"
        "cp \"$HERE/bitnetd.service\" \"$DEST/etc/systemd/system/bitnetd.service\"\n"
        "cp \"$HERE/juniorctl.py\" \"$DEST/usr/local/bin/juniorctl\"\n"
        "cp \"$HERE/oci/README.md\" \"$DEST/usr/lib/junioros/oci/README.md\"\n"
        "cp \"$HERE/seccomp-bitnetd.json\" \"$DEST/usr/lib/junioros/oci/seccomp-bitnetd.json\"\n"
        "grep 0.0.0.0 x && echo refusing && exit 3\nexit 0\n"
    ),
    "unit": "NoNewPrivileges=yes\nEnvironment=JUNIOR_BIND=127.0.0.1:8765\n",
    "seccomp": json.dumps({"defaultAction": "SCMP_ACT_ERRNO",
                           "archMap": [{"architecture": "SCMP_ARCH_X86_64"}],
                           "syscalls": [{"names": ["socket", "bind", "listen",
                                                   "accept4", "exit", "exit"],
                                         "action": "SCMP_ACT_ALLOW"}]}),
    "oci_config": json.dumps({"linux": {"namespaces": []}}),
    "oci_readme": "podman run --security-opt seccomp=../seccomp-bitnetd.json --network=none x\n",
    "i2sd": "HOST = '127.0.0.1'\nPORT = 8767\nOK = {'127.0.0.1', '::1'}\n",
    "linux_files": ["bitnetd.service", "seccomp-bitnetd.json", "install-overlay.sh",
                    "CONTAINER_SECURITY.md", "juniorctl.py"],
}


def selftest() -> int:
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    check("pack 4 trits", pack_trits([1, -1, 0, 1]) == bytes([0b01100001]))
    check("pack pads", pack_trits([-1]) == bytes([0b10000000]))
    c = close(GENESIS, b"\x01", "x")
    check("close 16 hex", len(c) == 16 and int(c, 16) >= 0)
    check("close depends on prev", c != close("1" * 16, b"\x01", "x"))
    check("close rule", c == hashlib.sha256(GENESIS.encode() + b"\x01" + b"x").hexdigest()[:16])

    jc = juniorctl_facts(FIX["juniorctl"])
    check("hardening parsed", jc["hardening"] == ["NoNewPrivileges=yes", "JUNIOR_BIND=127.0.0.1:8765"])
    check("top import seen", jc["top_imports"] == ["rails.linux.juniorctl_ops"])
    check("security reads", jc["security_reads"] == ["bitnetd.service", "seccomp-bitnetd.json"])
    check("gate keys", gate_keys(FIX["ctl_cli"]) == ["unit_ok", "seccomp"])
    ta = test_artifacts(FIX["test_sec"])
    check("cases", sorted(ta) == ["test_installer_exists", "test_seccomp_default_deny",
                                  "test_unit_hardened"])
    check("case files", ta["test_installer_exists"]["files"] == ["install-overlay.sh",
                                                                  "CONTAINER_SECURITY.md"])
    inst = installer_facts(FIX["installer"])
    check("installer exits", inst["exits"] == [0, 2, 3] and inst["set_eu"])
    check("installer case", inst["loopback_case"] == ["127.0.0.1", "::1"])

    rows = build(dict(FIX))
    by = {r["row"]: r for r in rows}
    check("fixture no -1", all(r["trit"] >= 0 for r in rows))
    check("doc not installed -> 0",
          by["A test_installer_exists CONTAINER_SECURITY.md"]["trit"] == 0)
    check("seccomp installed -> +1",
          by["A test_seccomp_default_deny seccomp-bitnetd.json"]["trit"] == 1)
    check("installed exit -> 0", by["B installed security exit"]["trit"] == 0)
    check("README ref breaks installed", by["D README seccomp ref installed"]["trit"] == 0)
    check("README ref ok repo", by["D README seccomp ref repo"]["trit"] == 1)
    check("i2sd ungated -> 0", by["E i2sd security gate"]["trit"] == 0)
    check("::1 guard advisory", by["C installer loopback guard"]["trit"] == 0)

    bad = dict(FIX)
    bad["seccomp"] = json.dumps({"defaultAction": "SCMP_ACT_ALLOW", "syscalls": []})
    bad["unit"] = "Environment=JUNIOR_BIND=0.0.0.0:8765\n"
    bad["i2sd"] = "HOST = '0.0.0.0'\nPORT = 8767\n"
    rb = {r["row"]: r for r in build(bad)}
    check("allow default -> -1", rb["seccomp defaultAction denies"]["trit"] == -1)
    check("wildcard unit -> -1", rb["unit bind loopback"]["trit"] == -1)
    check("missing hardening -> -1", rb["unit HARDENING present"]["trit"] == -1)
    check("i2sd 0.0.0.0 -> -1", rb["E i2sd bind"]["trit"] == -1)
    check("ports inherit -1", rb["F port Home"]["trit"] == -1)

    live = build(live_sources())
    tip = chain(live)
    check("live rows", len(live) > 20)
    check("live no -1", all(r["trit"] >= 0 for r in live))
    check("live chain deterministic", tip == chain(build(live_sources())))
    print(f"selftest {'pass' if not fails else 'FAIL'}: "
          f"{len(fails)} failed, tip {tip}")
    return 0 if not fails else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    rows = build(live_sources())
    tip = chain(rows)
    n = {t: sum(1 for r in rows if r["trit"] == t) for t in (1, 0, -1)}
    if "--json" in argv:
        print(json.dumps({"item": ROADMAP_ITEM, "rows": rows, "tip": tip,
                          "pass": n[1], "advisory": n[0], "fail": n[-1],
                          "flags": FLAGS, "rate": RATE, "bind": LOOPBACK},
                         indent=2))
    else:
        mark = {1: "+1", 0: " 0", -1: "-1"}
        for r in rows:
            print(f"{mark[r['trit']]} {r['close']} {r['row']}: {r['detail']}")
        print(f"{ROADMAP_ITEM}: {n[1]} pass, {n[0]} advisory, {n[-1]} fail, tip {tip}")
    return 0 if n[-1] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
