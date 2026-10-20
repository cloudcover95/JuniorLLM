#!/usr/bin/env python3
"""JuniorOS overlay: staged-tarball owner/mode landing handshake + port map.

Advances docs/BETA_TO_OS.md, section "JuniorOS overlay", the clause
"Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on
PATH" on its delivery side: rails/linux/ROADMAP.md row "1b Overlay" ships the
overlay as `debian/stage_overlay.sh -> tar.gz`.  When that tarball lands on a
host as root, who owns the root-loaded files (the bitnetd unit, the
junior-bitnetd ExecStart, juniorctl on PATH, the profile.d PATH pin), what
modes do they carry, and what does the archive's "./" entry do to the landing
root?  (os-release.junior, PATH, Debian/Alpine split, MemoryMax, user-unit
parity, the no-kernel-blob receipt, the GGUF gate, restart/ordering/sandbox
reach, second-tier hardening, the i2sd install closure and the seccomp
closure landed elsewhere and are not re-derived.)

How: in-process only.  Parse install-overlay.sh and debian/stage_overlay.sh
(mkdir -p, cp, heredoc and chmod lines, plus the final tar line), replay that
staging into a private temp dir with shutil/os under the running umask (cp
without -p keeps source mode & ~umask; mkdir -p and heredocs get 0777/0666 &
~umask), then build the archive in memory with tarfile the way
`tar -C "$OUT" -czf "$OUT.tar.gz" .` does (a "./" root entry, owner = builder
uid/gid unless --owner/--group are given).  Read the members back and check
names, types, special bits, exec bits, ownership and loopback binds.  The
Debian preseed late_command example (cp -a into /opt/junioros) is mapped
against the archive's / rooted layout.

Ports touched: bitnetd 127.0.0.1:8765 (unit JUNIOR_BIND, profile.d pin,
path.env) and i2sd 127.0.0.1:8767 (staged /usr/lib/junioros/i2sd.py).
Hand-off between Home, OSai, FieldCore, Flagstaff and JuniorLLM is one trit
pack (+1 pass, 0 advisory, -1 fail) closed with sha256(prev || pack ||
text)[:16].  No sockets, no subprocess, no shell run, no download, no model
pull.  Stdlib only.

    python ports/overlay_tarball_owner_mode_handshake.py
    python ports/overlay_tarball_owner_mode_handshake.py --json
    python ports/overlay_tarball_owner_mode_handshake.py --selftest

Exit 1 if any row is -1 (wildcard or non-loopback bind, unsafe member name,
link/device member, setuid/setgid/world-writable member, a missing cp source,
the ExecStart or juniorctl not executable, or a flag flipped).  Ownership and
landing gaps are advisories (0), reported with the smallest additive fix.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
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
LANES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ITEM = ('BETA_TO_OS "JuniorOS overlay": staged tarball (ROADMAP 1b) owner/mode '
        "landing for bitnetd.service, junior-bitnetd and juniorctl on PATH")
GENESIS = "0" * 16

# Paths a root-run boot or login actually loads from the overlay.
ROOT_LOADED = (
    "etc/systemd/system/bitnetd.service",
    "usr/local/bin/junior-bitnetd",
    "usr/local/bin/juniorctl",
    "etc/profile.d/junioros.sh",
    "usr/lib/systemd/user/bitnetd.service",
)
EXECS = ("usr/local/bin/junior-bitnetd", "usr/local/bin/juniorctl")

_Q = re.compile(r'"([^"]*)"')


# ---------------------------------------------------------------- parse

def _rel(tok: str, var: str) -> str | None:
    p = f"${var}/"
    return tok[len(p):] if tok.startswith(p) else None


def parse_install(text: str) -> dict:
    """mkdir/cp/heredoc/chmod ops from install-overlay.sh, relative to DEST."""
    ops = {"mkdir": [], "cp": [], "heredoc": [], "chmod": []}
    m = re.search(r"mkdir -p((?:\s*\\\n\s*\"[^\"]+\")+)", text)
    if m:
        ops["mkdir"] = [r for r in (_rel(t, "DEST") for t in _Q.findall(m.group(1))) if r]
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("cp "):
            toks = _Q.findall(s)
            if len(toks) >= 2:
                dst = _rel(toks[-1], "DEST")
                for t in toks[:-1]:
                    src = _rel(t, "HERE")
                    if src and dst:
                        ops["cp"].append((src, dst))
        elif s.startswith("chmod "):
            mm = re.match(r"chmod (\d{3,4}) (.+)$", s)
            if mm:
                for t in _Q.findall(mm.group(2)):
                    r = _rel(t, "DEST")
                    if r:
                        ops["chmod"].append((int(mm.group(1), 8), r))
    for mm in re.finditer(r"cat > \"\$DEST/([^\"]+)\" <<'EOF'\n(.*?)\nEOF\n", text, re.S):
        ops["heredoc"].append((mm.group(1), mm.group(2) + "\n"))
    return ops


def parse_stage(text: str) -> dict:
    """cp ops (HERE = rails/linux) and the tar line from debian/stage_overlay.sh."""
    cps, tar_line = [], ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("cp "):
            toks = _Q.findall(s)
            dst = _rel(toks[-1], "OUT") if toks else None
            for t in toks[:-1]:
                src = _rel(t, "HERE")
                if src and dst is not None:
                    d = dst + Path(src).name if dst.endswith("/") else dst
                    cps.append((src, d))
        elif s.startswith("tar "):
            tar_line = s
    return {"cp": cps, "tar": tar_line}


def parse_preseed(text: str) -> dict:
    cp = re.search(r"cp -a\s+(\S+)\s+(\S+)", text)
    return {
        "cp_a": bool(cp),
        "target": cp.group(2) if cp else "",
        "tar_extract": "tar " in text and "-x" in text,
        "enable": "systemctl enable bitnetd.service" in text,
    }


# ---------------------------------------------------------------- stage in-process

def _umask() -> int:
    u = os.umask(0)
    os.umask(u)
    return u


def stage_members(root: Path, inst: dict, stg: dict) -> tuple[list, list]:
    """Replay the staging in a private temp dir and tar it in memory."""
    um = _umask()
    missing = []
    tmp = Path(tempfile.mkdtemp(prefix="ov-own-"))
    out = tmp / "junioros-overlay"
    try:
        os.makedirs(out)
        for d in inst["mkdir"]:
            os.makedirs(out / d, exist_ok=True)
        lin = root / LINUX
        for src, dst in inst["cp"] + stg["cp"]:
            sp = lin / src
            if not sp.is_file():
                missing.append(src)
                continue
            shutil.copyfile(sp, out / dst)
            os.chmod(out / dst, stat.S_IMODE(sp.stat().st_mode) & ~um)
        for dst, body in inst["heredoc"]:
            (out / dst).write_text(body)
        for mode, dst in inst["chmod"]:
            if (out / dst).exists():
                os.chmod(out / dst, mode)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.GNU_FORMAT) as tf:
            tf.add(str(out), arcname=".")
        buf.seek(0)
        members = []
        with tarfile.open(fileobj=buf, mode="r:gz") as tf:
            for ti in tf.getmembers():
                text = ""
                if ti.isfile():
                    fh = tf.extractfile(ti)
                    text = fh.read().decode("utf-8", "replace") if fh else ""
                members.append(_m(ti.name, _kind(ti), ti.mode, ti.uid, ti.gid,
                                  ti.uname, text))
        return members, missing
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _kind(ti: tarfile.TarInfo) -> str:
    if ti.isdir():
        return "dir"
    if ti.isfile():
        return "file"
    if ti.issym():
        return "symlink"
    if ti.islnk():
        return "hardlink"
    return "special"


def _m(name, kind, mode, uid, gid, uname="", text="") -> dict:
    return {"name": name, "kind": kind, "mode": mode, "uid": uid, "gid": gid,
            "uname": uname, "text": text}


def _norm(name: str) -> str:
    n = name[2:] if name.startswith("./") else name
    return "" if n == "." else n


def load(root: Path) -> dict:
    lin = root / LINUX
    rd = lambda p: (lin / p).read_text() if (lin / p).is_file() else ""
    s = {"install": rd("install-overlay.sh"), "stage": rd("debian/stage_overlay.sh"),
         "preseed": rd("debian/preseed.fragment"), "roadmap": rd("ROADMAP.md")}
    inst, stg = parse_install(s["install"]), parse_stage(s["stage"])
    members, missing = stage_members(root, inst, stg)
    s.update(members=members, missing=missing, n_cp=len(inst["cp"]) + len(stg["cp"]))
    try:
        import pwd
        uname = pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError):
        uname = "?"
    s["rt"] = {"uid": os.getuid(), "gid": os.getgid(), "user": uname,
               "umask": f"{_umask():04o}"}
    return s


# ---------------------------------------------------------------- check

def _row(key, trit, note):
    return {"key": key, "trit": trit, "note": note}


def check(s: dict) -> dict:
    rows, fix = [], []
    mem = {_norm(m["name"]): m for m in s["members"]}
    rt = s["rt"]

    ok = (FLAGS["rate"] == 0.05 and FLAGS["bind"] == "127.0.0.1" and not any(
        FLAGS[k] for k in ("mint", "socket", "layer_one", "fetch", "ue5_launch", "model_pull")))
    rows.append(_row("flags.locked", 1 if ok else -1,
                     "rate 0.05, mint/socket/layer_one/fetch/ue5_launch/model_pull false"))

    rows.append(_row("roadmap.1b", 1 if re.search(r"1b Overlay.*stage_overlay\.sh.*tar\.gz",
                                                   s["roadmap"]) else 0,
                     "rails/linux/ROADMAP.md row 1b names stage_overlay.sh -> tar.gz"))

    rows.append(_row("stage.cp_closure", -1 if s["missing"] else 1,
                     f"{s['n_cp']} cp sources resolve under {LINUX}" if not s["missing"]
                     else "missing cp sources: " + ", ".join(s["missing"])))

    want = [p for p in ROOT_LOADED + ("usr/lib/junioros/i2sd.py", "etc/junioros/path.env")]
    absent = [p for p in want if p not in mem]
    rows.append(_row("tar.root_loaded_present", -1 if absent else 1,
                     "unit, user unit, junior-bitnetd, juniorctl, profile.d pin, path.env "
                     "and i2sd.py are archive members" if not absent
                     else "absent from archive: " + ", ".join(absent)))

    unit = mem.get("etc/systemd/system/bitnetd.service", {}).get("text", "")
    rows.append(_row("lane.bitnetd_bind",
                     1 if "JUNIOR_BIND=127.0.0.1:8765" in unit else -1,
                     "archived system unit pins JUNIOR_BIND=127.0.0.1:8765"))
    pin = mem.get("etc/profile.d/junioros.sh", {}).get("text", "")
    env = mem.get("etc/junioros/path.env", {}).get("text", "")
    rows.append(_row("lane.bitnetd_pin",
                     1 if "127.0.0.1:8765" in pin and "JUNIOR_BIND=127.0.0.1:8765" in env else -1,
                     "profile.d pin and path.env default JUNIOR_BIND to 127.0.0.1:8765"))
    i2 = mem.get("usr/lib/junioros/i2sd.py", {}).get("text", "")
    rows.append(_row("lane.i2sd_bind",
                     1 if 'HOST = "127.0.0.1"' in i2 and re.search(r"^PORT = 8767$", i2, re.M)
                     else -1, "archived i2sd.py binds 127.0.0.1:8767"))

    wild = [n for n, m in mem.items() if "0.0.0.0" in m["text"]]
    rows.append(_row("bind.no_wildcard", -1 if wild else 1,
                     "no archive member mentions 0.0.0.0" if not wild
                     else "wildcard in: " + ", ".join(wild)))

    bad_names = [m["name"] for m in s["members"]
                 if m["name"].startswith("/") or ".." in Path(m["name"]).parts]
    rows.append(_row("tar.names_safe", -1 if bad_names else 1,
                     "all member names relative, no '..'" if not bad_names
                     else "unsafe names: " + ", ".join(bad_names)))

    odd = [f"{n or '.'}({m['kind']})" for n, m in mem.items() if m["kind"] not in ("dir", "file")]
    rows.append(_row("tar.types", -1 if odd else 1,
                     "dirs and regular files only (no symlink, hardlink, device)" if not odd
                     else "non-regular members: " + ", ".join(odd)))

    special = [n or "." for n, m in mem.items() if m["mode"] & (stat.S_ISUID | stat.S_ISGID)]
    rows.append(_row("tar.no_suid", -1 if special else 1,
                     "no setuid/setgid member" if not special
                     else "setuid/setgid: " + ", ".join(special)))
    ww = [n or "." for n, m in mem.items() if m["mode"] & stat.S_IWOTH]
    rows.append(_row("tar.no_world_write", -1 if ww else 1,
                     "no world-writable member" if not ww
                     else "world-writable: " + ", ".join(ww)))

    noexec = [p for p in EXECS if p in mem and (mem[p]["mode"] & 0o111) != 0o111]
    rows.append(_row("tar.exec_bits", -1 if noexec else 1,
                     "junior-bitnetd (unit ExecStart) and juniorctl are 0755" if not noexec
                     else "not a+x: " + ", ".join(noexec)))
    xdata = [n for n, m in mem.items() if m["kind"] == "file" and n not in EXECS
             and m["mode"] & 0o111]
    rows.append(_row("tar.data_noexec", 0 if xdata else 1,
                     "unit, env, json and staged .py data carry no exec bit" if not xdata
                     else "exec bit on data: " + ", ".join(xdata)))

    nonroot = sorted({(m["uid"], m["gid"], m["uname"]) for n, m in mem.items()
                      if n in ROOT_LOADED or n == ""} - {(0, 0, "root")})
    nonroot = [t for t in nonroot if t[0] or t[1]]
    if nonroot:
        u = ", ".join(f"uid {a}/gid {b} ({c or '?'})" for a, b, c in nonroot)
        rows.append(_row("tar.owner_root", 0,
                         f"root-loaded members are owned by the builder, {u}; GNU tar run "
                         "as root defaults to --same-owner, so the bitnetd unit, its "
                         "ExecStart and juniorctl on PATH land writable by that uid"))
    else:
        rows.append(_row("tar.owner_root", 1, "root-loaded members archived as 0/0"))

    tl = s.get("stage", "")
    tar_line = parse_stage(tl)["tar"]
    flags_ok = all(f in tar_line for f in ("--owner=0", "--group=0", "--numeric-owner"))
    rows.append(_row("stage.owner_flags", 1 if flags_ok else 0,
                     f"tar line: {tar_line or '(none)'}"))
    if not flags_ok:
        fix.append('debian/stage_overlay.sh: tar --owner=0 --group=0 --numeric-owner '
                   '-C "$OUT" -czf "$OUT.tar.gz" .')

    rootent = mem.get("")
    if rootent is not None and re.search(r'-czf\s+"\$OUT\.tar\.gz"\s+\.\s*$', tar_line):
        rows.append(_row("tar.root_entry", 0,
                         f"archive carries a './' entry (mode {rootent['mode']:04o}, uid "
                         f"{rootent['uid']}); tar's default --overwrite-dir re-applies that "
                         "owner and mode to the landing root (/ or /target) on extract"))
        fix.append("extract with tar --no-overwrite-dir -C <root> -xzf junioros-overlay.tar.gz")
    else:
        rows.append(_row("tar.root_entry", 1, "no './' entry re-owns the landing root"))

    ps = parse_preseed(s.get("preseed", ""))
    if ps["cp_a"] and ps["target"].rstrip("/") not in ("", "/", "/target"):
        rows.append(_row("land.preseed_root", 0,
                         f"preseed late_command cp -a lands the tree at {ps['target']} but the "
                         "archive is / rooted (unit ExecStart=/usr/local/bin/junior-bitnetd, "
                         "etc/systemd/system/bitnetd.service), so 'systemctl enable "
                         "bitnetd.service' finds no unit; cp -a also keeps the builder uid"))
        fix.append("debian/preseed.fragment: extract the staged tar.gz at /target with "
                   "--no-overwrite-dir instead of cp -a into /opt/junioros")
    else:
        rows.append(_row("land.preseed_root", 1, "preseed lands the overlay at the root"))

    return {"rows": rows, "fix": fix, "runtime": rt}


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

def _pin(s: dict, uid: int, gid: int, uname: str) -> dict:
    g = dict(s)
    g["members"] = [dict(m, uid=uid, gid=gid, uname=uname) for m in s["members"]]
    g["rt"] = {"uid": uid, "gid": gid, "user": uname, "umask": "0022"}
    return g


def selftest() -> int:
    ok = True
    s = _pin(load(ROOT), 1000, 1000, "builder")
    real = report(s)
    if real["fail"]:
        print("selftest: real tree has fail rows: " + ", ".join(
            r["key"] for r in real["rows"] if r["trit"] == -1), file=sys.stderr)
        ok = False
    rk = {r["key"]: r["trit"] for r in real["rows"]}
    if rk.get("tar.owner_root") != 0:
        print("selftest: builder-owned archive should be advisory", file=sys.stderr)
        ok = False

    good = _pin(s, 0, 0, "root")
    good["stage"] = s["stage"].replace(
        'tar -C "$OUT" -czf "$OUT.tar.gz" .',
        'tar --owner=0 --group=0 --numeric-owner -C "$OUT" -czf "$OUT.tar.gz" -- etc usr')
    good["members"] = [m for m in good["members"] if _norm(m["name"]) != ""]
    good["preseed"] = s["preseed"].replace("cp -a /media/usb/junioros-overlay/. /opt/junioros/",
                                           "tar --no-overwrite-dir -C /target -xzf ov.tar.gz")
    gk = {r["key"]: r["trit"] for r in report(good)["rows"]}
    for k in ("tar.owner_root", "stage.owner_flags", "tar.root_entry", "land.preseed_root"):
        if gk.get(k) != 1:
            print(f"selftest: fixed fixture did not pass {k} ({gk.get(k)})", file=sys.stderr)
            ok = False

    bad = dict(s)
    ms = [dict(m) for m in s["members"]]
    for m in ms:
        n = _norm(m["name"])
        if n == "etc/systemd/system/bitnetd.service":
            m["text"] = m["text"].replace("127.0.0.1:8765", "0.0.0.0:8765")
        if n == "usr/local/bin/juniorctl":
            m["mode"] = 0o4644
        if n == "etc/junioros/path.env":
            m["mode"] = 0o666
    ms.append(_m("./usr/local/bin/python3", "symlink", 0o777, 1000, 1000, "builder"))
    ms.append(_m("../etc/shadow", "file", 0o644, 1000, 1000, "builder"))
    bad["members"] = ms
    bad["missing"] = ["bitnetd.service"]
    bk = {r["key"]: r["trit"] for r in report(bad)["rows"]}
    for k in ("lane.bitnetd_bind", "bind.no_wildcard", "tar.no_suid", "tar.no_world_write",
              "tar.exec_bits", "tar.types", "tar.names_safe", "stage.cp_closure"):
        if bk.get(k) != -1:
            print(f"selftest: bad fixture did not fail {k} ({bk.get(k)})", file=sys.stderr)
            ok = False

    a, b, c2 = report(s), report(s), report(s, "f" * 16)
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
          f"rows {len(rk)}")
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
        print(json.dumps({k: v for k, v in r.items()}, indent=2))
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
