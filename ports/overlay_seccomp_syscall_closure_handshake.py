#!/usr/bin/env python3
"""JuniorOS overlay: hardened-bitnetd seccomp syscall-closure handshake + port map.

Advances docs/BETA_TO_OS.md, section "JuniorOS overlay", the item
"hardened `bitnetd.service`" on its sandbox-filter side: the overlay ships two
filters for the same daemon, the systemd unit's SystemCallFilter
(rails/linux/bitnetd.service) and the rootless OCI profile
rails/linux/seccomp-bitnetd.json (staged by install-overlay.sh to
/usr/lib/junioros/oci/, named by oci/README.md and oci/rootless.py).  Does the
OCI allow-list actually close over the syscalls the i2sd fallback
(rails/linux/i2sd.py, 127.0.0.1:8767) needs, and do the two filters agree?
(os-release.junior, PATH, Debian/Alpine, MemoryMax, user-unit parity, the
no-kernel-blob receipt, the GGUF gate, restart/ordering/sandbox reach, archMap
and the i2sd install closure landed elsewhere and are not re-derived.)

How: in-process only.  AST-read i2sd.py to see which stdlib server it builds
(ThreadingTCPServer -> one pthread per connection; serve_forever ->
socketserver._ServerSelector), read the running interpreter's own selector
class and libc (platform.libc_ver) as evidence, and fold that into a static
syscall model for CPython on Linux x86_64 (glibc >= 2.34 creates threads with
clone3 and only falls back to clone on ENOSYS; musl uses clone).  Then diff
that set against the OCI allow-list, the OCI defaultAction/errno, and the
unit's @system-service groups and denials.

Ports touched: bitnetd 127.0.0.1:8765 (unit JUNIOR_BIND, bitnet.cpp lane) and
i2sd 127.0.0.1:8767 (fallback lane).  Hand-off between Home, OSai, FieldCore,
Flagstaff and JuniorLLM is one trit pack (+1 pass, 0 advisory, -1 fail)
closed with sha256(prev || pack || text)[:16].  No sockets, no subprocess, no
shell run, no download, no model pull.  Stdlib only.

    python ports/overlay_seccomp_syscall_closure_handshake.py
    python ports/overlay_seccomp_syscall_closure_handshake.py --json
    python ports/overlay_seccomp_syscall_closure_handshake.py --selftest

Exit 1 if any row is -1 (non-loopback or wildcard bind, OCI profile missing or
default-allow, an escape syscall allowed, or a flag flipped).  Closure gaps
are advisories (0), reported with the smallest additive fix.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import socketserver
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
LANES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ITEM = ("BETA_TO_OS.md / JuniorOS overlay / hardened bitnetd.service "
        "(OCI seccomp syscall closure for the i2sd fallback)")
GENESIS = "0" * 16
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

SOURCES = {
    "unit": f"{LINUX}/bitnetd.service",
    "seccomp": f"{LINUX}/seccomp-bitnetd.json",
    "i2sd": f"{LINUX}/i2sd.py",
    "bitnetd_sh": f"{LINUX}/bitnetd.sh",
    "install": f"{LINUX}/install-overlay.sh",
    "oci_readme": f"{LINUX}/oci/README.md",
}

# Static model: syscalls CPython 3 on Linux x86_64 issues to start, import
# from a directory tree and run a loopback stream server.  Kept small and
# conservative: only calls that are not optional on the normal path.
BASE = {
    "read", "write", "openat", "close", "fstat", "newfstatat", "lseek",
    "getdents64", "mmap", "mprotect", "munmap", "brk", "rt_sigaction",
    "rt_sigprocmask", "ioctl", "getrandom", "futex", "set_tid_address",
    "readlink", "getcwd", "exit_group", "execve",
}
GLIBC_START = {"set_robust_list", "rseq"}   # glibc registers both at startup
SERVER = {"socket", "setsockopt", "bind", "listen", "accept4", "getsockname",
          "sendto", "recvfrom"}
THREAD_GLIBC = {"clone3", "set_robust_list", "rseq", "madvise", "exit"}
THREAD_MUSL = {"clone", "set_tid_address", "madvise", "exit"}
POLL = {"poll", "ppoll"}            # either one satisfies PollSelector
EPOLL = {"epoll_create1", "epoll_ctl", "epoll_wait", "epoll_pwait"}
SELECT = {"select", "pselect6"}

# Syscalls that let a sandboxed daemon leave its sandbox; never allowed.
ESCAPE = {"ptrace", "mount", "umount2", "pivot_root", "chroot", "init_module",
          "finit_module", "delete_module", "kexec_load", "kexec_file_load",
          "bpf", "setns", "unshare", "reboot", "swapon", "swapoff",
          "process_vm_writev", "open_by_handle_at", "perf_event_open",
          "keyctl", "add_key", "request_key", "userfaultfd"}

# Static slice of systemd's documented filter groups, only for the syscalls
# this check reasons about (systemd.exec(5) SystemCallFilter= sets).
SD_GROUPS = {
    "@basic-io": {"read", "write", "close", "lseek", "pread64", "pwrite64",
                  "readv", "writev", "dup", "dup2", "dup3"},
    "@file-system": {"openat", "fstat", "newfstatat", "getdents64", "getcwd",
                     "readlink", "statx", "access", "mkdir", "unlink"},
    "@io-event": {"poll", "ppoll", "select", "pselect6", "epoll_create1",
                  "epoll_ctl", "epoll_wait", "epoll_pwait", "eventfd2"},
    "@process": {"clone", "clone3", "fork", "vfork", "execve", "exit",
                 "exit_group", "wait4", "kill", "set_tid_address", "prctl",
                 "arch_prctl", "rseq"},
    "@network-io": {"socket", "setsockopt", "bind", "listen", "accept4",
                    "getsockname", "sendto", "recvfrom", "connect"},
    "@default": {"brk", "futex", "set_robust_list", "rt_sigreturn", "getrandom",
                 "mmap", "munmap", "mprotect", "madvise"},
    "@signal": {"rt_sigaction", "rt_sigprocmask"},
    "@privileged": {"chroot", "mount", "init_module", "kexec_load", "bpf",
                    "setns", "unshare", "reboot", "swapon", "ptrace"},
    "@resources": {"setrlimit", "sched_setaffinity", "setpriority"},
    "@mount": {"mount", "umount2", "pivot_root"},
    "@reboot": {"reboot", "kexec_load", "kexec_file_load"},
    "@swap": {"swapon", "swapoff"},
    "@raw-io": {"ioperm", "iopl"},
    "@module": {"init_module", "finit_module", "delete_module"},
    "@debug": {"ptrace", "process_vm_writev", "perf_event_open"},
    "@keyring": {"keyctl", "add_key", "request_key"},
}
SD_GROUPS["@system-service"] = set().union(*(
    SD_GROUPS[g] for g in ("@basic-io", "@file-system", "@io-event", "@process",
                           "@network-io", "@default", "@signal", "@resources",
                           "@keyring")), {"ioctl"})


# ---------------------------------------------------------------- load

def load(root: Path) -> dict:
    s = {}
    for k, rel in SOURCES.items():
        p = root / rel
        s[k] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
    return s


def runtime() -> dict:
    """In-process evidence from this interpreter (no syscalls traced)."""
    lib, ver = platform.libc_ver()
    sel = getattr(socketserver, "_ServerSelector", None)
    return {"libc": lib or "unknown", "libc_ver": ver or "",
            "server_selector": getattr(sel, "__name__", "unknown"),
            "machine": platform.machine() or "unknown"}


# ---------------------------------------------------------------- parse

def unit_env(unit: str) -> dict:
    env = {}
    for m in re.finditer(r"^Environment=(\w+)=(\S+)", unit, re.M):
        env[m.group(1)] = m.group(2)
    return env


def unit_filter(unit: str) -> tuple:
    """(allowed, denied) after folding SystemCallFilter lines in order."""
    allow, deny = set(), set()
    for m in re.finditer(r"^SystemCallFilter=(.*)$", unit, re.M):
        val = m.group(1).strip()
        neg = val.startswith("~")
        for tok in val.lstrip("~").split():
            names = SD_GROUPS.get(tok, {tok} if not tok.startswith("@") else set())
            (deny if neg else allow).update(names)
    return allow, deny


def seccomp(text: str) -> dict:
    try:
        d = json.loads(text) if text.strip() else None
    except ValueError:
        d = None
    if not isinstance(d, dict):
        return {"ok": False, "allow": set(), "default": "", "errno": None,
                "socket_args": False}
    allow, socket_args = set(), False
    for rule in d.get("syscalls", []):
        if rule.get("action") == "SCMP_ACT_ALLOW":
            names = set(rule.get("names", []))
            allow |= names
            if "socket" in names and rule.get("args"):
                socket_args = True
    return {"ok": True, "allow": allow, "default": d.get("defaultAction", ""),
            "errno": d.get("defaultErrnoRet"), "socket_args": socket_args}


def i2sd_shape(src: str) -> dict:
    """AST: server base class, serve_forever use, HOST/PORT literals."""
    out = {"threading_server": False, "serve_forever": False,
           "host": None, "port": None, "parsed": False}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    out["parsed"] = True
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                name = ast.unparse(b)
                if name.endswith(("ThreadingTCPServer", "ThreadingMixIn",
                                  "ThreadingUnixStreamServer")):
                    out["threading_server"] = True
        if isinstance(node, ast.Attribute) and node.attr == "serve_forever":
            out["serve_forever"] = True
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant):
            if node.targets[0].id == "HOST":
                out["host"] = node.value.value
            if node.targets[0].id == "PORT":
                out["port"] = node.value.value
    return out


def required(shape: dict, rt: dict, libc: str) -> tuple:
    """(must-have set, list of alternatives where any one member suffices)."""
    need = set(BASE) | SERVER | (set() if libc == "musl" else GLIBC_START)
    alts = []
    if shape["threading_server"]:
        need |= THREAD_MUSL if libc == "musl" else THREAD_GLIBC
    if shape["serve_forever"]:
        sel = rt["server_selector"]
        alts.append(EPOLL if sel == "EpollSelector" else
                    SELECT if sel == "SelectSelector" else POLL)
    return need, alts


# ---------------------------------------------------------------- check

def row(rows: list, key: str, trit: int, note: str) -> None:
    rows.append({"key": key, "trit": trit, "note": note})


def check(s: dict, rt: dict | None = None) -> dict:
    rt = rt or runtime()
    rows, fix = [], []

    # flags and ports
    flags_ok = (FLAGS["rate"] == 0.05 and FLAGS["bind"] == "127.0.0.1" and not any(
        FLAGS[k] for k in ("mint", "socket", "layer_one", "fetch", "ue5_launch",
                           "model_pull")))
    row(rows, "flags", 1 if flags_ok else -1,
        "rate 0.05, mint/socket/layer_one/fetch/ue5_launch/model_pull false")

    env = unit_env(s["unit"])
    bind = env.get("JUNIOR_BIND", "")
    host, _, port = bind.rpartition(":")
    row(rows, "lane.bitnetd_bind", 1 if host in LOOPBACK and port == "8765" else -1,
        f"unit JUNIOR_BIND={bind or 'missing'} (want 127.0.0.1:8765)")
    shape = i2sd_shape(s["i2sd"])
    row(rows, "lane.i2sd_bind",
        1 if shape["host"] in LOOPBACK and shape["port"] == 8767 else -1,
        f"i2sd HOST={shape['host']} PORT={shape['port']} (want 127.0.0.1:8767)")
    wild = [k for k in ("unit", "i2sd", "bitnetd_sh", "seccomp") if "0.0.0.0" in s[k]]
    row(rows, "bind.no_wildcard", -1 if wild else 1,
        "no 0.0.0.0 in unit/i2sd/bitnetd.sh/seccomp" if not wild else f"wildcard in {wild}")

    # OCI profile present, staged, referenced, default-deny
    sc = seccomp(s["seccomp"])
    row(rows, "oci.profile_parses", 1 if sc["ok"] else -1,
        f"{SOURCES['seccomp']} {'parses' if sc['ok'] else 'missing or invalid'}, "
        f"{len(sc['allow'])} allowed")
    staged = "seccomp-bitnetd.json" in s["install"]
    row(rows, "oci.profile_staged", 1 if staged else 0,
        "install-overlay.sh stages it under /usr/lib/junioros/oci/" if staged
        else "install-overlay.sh does not stage it")
    named = "seccomp=../seccomp-bitnetd.json" in s["oci_readme"]
    row(rows, "oci.profile_named", 1 if named else 0,
        "oci/README.md podman line passes --security-opt seccomp=../seccomp-bitnetd.json"
        if named else "oci/README.md does not pass the profile")
    deny_default = sc["default"] in ("SCMP_ACT_ERRNO", "SCMP_ACT_KILL",
                                     "SCMP_ACT_KILL_PROCESS", "SCMP_ACT_TRAP")
    row(rows, "oci.default_deny", 1 if deny_default else -1,
        f"defaultAction={sc['default'] or 'missing'}")
    esc = sorted(ESCAPE & sc["allow"])
    row(rows, "oci.no_escape_syscalls", -1 if esc else 1,
        "no ptrace/mount/bpf/setns/unshare/kexec/module/keyring allowed" if not esc
        else f"escape syscalls allowed: {esc}")

    # closure: glibc (Debian) and musl (Alpine) lanes
    missing_all = set()
    for libc in ("glibc", "musl"):
        need, alts = required(shape, rt, libc)
        miss = sorted(need - sc["allow"])
        for alt in alts:
            if not alt & sc["allow"]:
                miss.append("|".join(sorted(alt)))
        missing_all.update(miss)
        lane = "Debian/glibc" if libc == "glibc" else "Alpine/musl"
        row(rows, f"closure.{libc}", 0 if miss else 1,
            f"{lane}: python3 i2sd.py under the OCI profile "
            + (f"is denied {miss}" if miss else "has every modelled syscall"))

    # clone3 + EPERM: glibc only falls back to clone on ENOSYS
    if shape["threading_server"] and "clone3" not in sc["allow"]:
        enosys = sc["errno"] == 38
        row(rows, "closure.clone3_errno", 1 if enosys else 0,
            "clone3 denied with ENOSYS (38), glibc falls back to clone" if enosys else
            f"clone3 denied with {'EPERM (default)' if sc['errno'] is None else sc['errno']}; "
            "glibc >= 2.34 pthread_create fails instead of falling back, so "
            "ThreadingTCPServer cannot start a handler thread")

    # unit lane: @system-service covers the same set, and the denials do not cut it
    u_allow, u_deny = unit_filter(s["unit"])
    need_g, alts_g = required(shape, rt, "glibc")
    u_eff = u_allow - u_deny
    u_miss = sorted(need_g - u_eff) + ["|".join(sorted(a)) for a in alts_g if not a & u_eff]
    row(rows, "unit.filter_closure", 0 if u_miss else 1,
        "unit SystemCallFilter=@system-service minus its ~ list keeps every modelled syscall"
        if not u_miss else f"unit filter denies {u_miss}")
    drift = sorted((need_g & u_eff) - sc["allow"])
    row(rows, "parity.unit_vs_oci", 0 if drift else 1,
        "unit and OCI filters agree on the modelled set" if not drift else
        f"systemd lane allows but OCI lane denies {drift}")
    fam = re.search(r"^RestrictAddressFamilies=(.*)$", s["unit"], re.M)
    if fam and "socket" in sc["allow"]:
        row(rows, "parity.address_families", 1 if sc["socket_args"] else 0,
            f"unit RestrictAddressFamilies={fam.group(1).strip()}; OCI socket "
            + ("is arg-filtered" if sc["socket_args"] else
               "has no args filter, so AF_NETLINK/AF_PACKET etc. pass the OCI lane"))

    row(rows, "runtime.evidence", 1,
        f"this interpreter: {rt['libc']} {rt['libc_ver']} {rt['machine']}, "
        f"socketserver selector {rt['server_selector']}")

    if missing_all:
        add = sorted({n for m in missing_all for n in m.split("|")})
        fix.append("seccomp-bitnetd.json: add to the SCMP_ACT_ALLOW names "
                   + ", ".join(f'"{n}"' for n in add))
    if any(r["key"] == "closure.clone3_errno" and r["trit"] == 0 for r in rows):
        fix.append('or keep clone3 out and add {"names": ["clone3"], '
                   '"action": "SCMP_ACT_ERRNO", "errnoRet": 38} so glibc falls back to clone')
    if any(r["key"] == "parity.address_families" and r["trit"] == 0 for r in rows):
        fix.append('seccomp-bitnetd.json: move "socket" to its own rule with args '
                   "allowing only AF_UNIX(1)/AF_INET(2)/AF_INET6(10), matching the unit")
    return {"rows": rows, "fix": fix, "runtime": rt}


# ---------------------------------------------------------------- close

def pack(rows: list) -> bytes:
    return bytes(r["trit"] + 1 for r in rows)


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def report(s: dict, prev: str = GENESIS, rt: dict | None = None) -> dict:
    res = check(s, rt)
    rows = res["rows"]
    text = "\n".join(f"{r['trit']:+d} {r['key']} {r['note']}" for r in rows)
    pk = pack(rows)
    return {
        "item": ITEM,
        "lanes": list(LANES),
        "ports": {k: f"{h}:{p}" for k, (h, p) in PORTS.items()},
        "flags": FLAGS,
        "runtime": res["runtime"],
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

FIXED_RT = {"libc": "glibc", "libc_ver": "2.41", "server_selector": "PollSelector",
            "machine": "x86_64"}


def _fixed(s: dict) -> dict:
    """Same tree with the additive seccomp fix applied in memory."""
    g = dict(s)
    d = json.loads(s["seccomp"])
    names = d["syscalls"][0]["names"]
    for n in ("lseek", "getdents64", "poll", "ppoll", "set_robust_list", "clone3"):
        if n not in names:
            names.append(n)
    names.remove("socket")
    d["syscalls"].append({"names": ["socket"], "action": "SCMP_ACT_ALLOW",
                          "args": [{"index": 0, "value": 2, "op": "SCMP_CMP_EQ"}]})
    g["seccomp"] = json.dumps(d)
    return g


def selftest() -> int:
    ok = True
    s = load(ROOT)
    real = report(s, rt=FIXED_RT)
    if real["fail"]:
        print("selftest: real tree has fail rows", file=sys.stderr)
        ok = False
    keys = {r["key"]: r["trit"] for r in real["rows"]}

    good = {r["key"]: r["trit"] for r in report(_fixed(s), rt=FIXED_RT)["rows"]}
    for k in ("closure.glibc", "closure.musl", "parity.unit_vs_oci",
              "parity.address_families", "unit.filter_closure"):
        if good.get(k) != 1:
            print(f"selftest: fixed profile did not pass {k} ({good.get(k)})",
                  file=sys.stderr)
            ok = False
    if "closure.clone3_errno" in good:
        print("selftest: clone3 row should vanish once clone3 is allowed", file=sys.stderr)
        ok = False

    bad = dict(s)
    d = json.loads(s["seccomp"])
    d["syscalls"][0]["names"] += ["ptrace", "unshare"]
    bad["seccomp"] = json.dumps(d)
    bad["unit"] = s["unit"].replace("JUNIOR_BIND=127.0.0.1:8765", "JUNIOR_BIND=0.0.0.0:8765")
    bad["i2sd"] = s["i2sd"].replace("PORT = 8767", "PORT = 8765", 1)
    bk = {r["key"]: r["trit"] for r in report(bad, rt=FIXED_RT)["rows"]}
    for k in ("oci.no_escape_syscalls", "lane.bitnetd_bind", "bind.no_wildcard",
              "lane.i2sd_bind"):
        if bk.get(k) != -1:
            print(f"selftest: bad fixture did not fail {k} ({bk.get(k)})", file=sys.stderr)
            ok = False
    allow_all = dict(s)
    d2 = json.loads(s["seccomp"])
    d2["defaultAction"] = "SCMP_ACT_ALLOW"
    allow_all["seccomp"] = json.dumps(d2)
    if {r["key"]: r["trit"] for r in report(allow_all, rt=FIXED_RT)["rows"]}.get(
            "oci.default_deny") != -1:
        print("selftest: default-allow profile not failed", file=sys.stderr)
        ok = False

    musl = report(s, rt=dict(FIXED_RT, libc="musl"))
    if any(r["key"] == "closure.musl" and "clone3" in r["note"] for r in musl["rows"]):
        print("selftest: musl lane should not need clone3", file=sys.stderr)
        ok = False

    a, b = report(s, GENESIS, FIXED_RT), report(s, GENESIS, FIXED_RT)
    c2 = report(s, "f" * 16, FIXED_RT)
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
        for rw in r["rows"]:
            print(f"{rw['trit']:+d} {rw['key']}: {rw['note']}")
        for f in r["fix"]:
            print(f"fix: {f}")
        print(f"{r['pass']} pass, {r['advisory']} advisory, {r['fail']} fail; "
              f"close {r['close']}")
    return 1 if r["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
