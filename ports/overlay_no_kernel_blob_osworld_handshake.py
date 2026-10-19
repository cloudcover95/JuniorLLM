#!/usr/bin/env python3
"""JuniorOS overlay: "No OSWorld score. No kernel blob." handshake + port map.

Advances docs/BETA_TO_OS.md, section "JuniorOS overlay", the closing clause
"No OSWorld score. No kernel blob." (the thinnest overlay limit; the
os-release.junior, PATH, Debian/Alpine, MemoryMax and user-unit parts already
landed elsewhere and are not re-derived here).

Angle: the install manifest, not the identity file.  The script statically
reads what rails/linux/install-overlay.sh and rails/linux/debian/stage_overlay.sh
actually write under DEST/OUT (every `cp` target and every heredoc target),
plus debian/packages.list and debian/preseed.fragment, and checks that:

  * no staged path lands in /boot, /lib/modules or a firmware dir,
  * no staged file is a kernel artifact by name (vmlinuz, bzImage, initrd,
    *.ko, System.map, *.dtb, *.efi) or by content (ELF / NUL bytes),
  * no overlay script runs a kernel-mutating command (modprobe, insmod,
    update-initramfs, dkms, kexec, grub-install, ...),
  * no package list pulls a linux-image / linux-headers / firmware package,
  * kconfig.junior is staged only as text under usr/lib/junioros, never as
    /boot/config-*,
  * the PID-adjacent services stay userspace on loopback: bitnetd.service
    ExecStart is an overlay-installed wrapper on 127.0.0.1:8765, and i2sd is
    a userspace file on 127.0.0.1:8767,
  * every OSWorld mention in the overlay docs / identity text is a negation
    and carries no numeric score, and os-release.junior has no score key.

Hand-off between Home, OSai, FieldCore, Flagstaff and JuniorLLM: the result
is one trit pack (+1 pass, 0 advisory, -1 fail) closed with
sha256(prev || pack || text)[:16].  Everything is in-process: no sockets, no
subprocess, no download, no model pull.  Stdlib only.

    python ports/overlay_no_kernel_blob_osworld_handshake.py
    python ports/overlay_no_kernel_blob_osworld_handshake.py --json
    python ports/overlay_no_kernel_blob_osworld_handshake.py --selftest

Exit 1 if any row is -1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
PORTS = {"bitnetd": "127.0.0.1:8765", "i2sd": "127.0.0.1:8767"}
ITEM = 'BETA_TO_OS.md / JuniorOS overlay / "No OSWorld score. No kernel blob."'
GENESIS = "0" * 16

KERNEL_DIRS = ("boot/", "lib/modules/", "usr/lib/modules/",
               "lib/firmware/", "usr/lib/firmware/", "efi/")
KERNEL_NAME = re.compile(
    r"(^|/)(vmlinu[xz][^/]*|bzImage|zImage|Image(\.gz)?|initrd[^/]*|"
    r"initramfs[^/]*|System\.map[^/]*|[^/]+\.ko(\.(xz|zst|gz))?|[^/]+\.dtb|"
    r"[^/]+\.efi|config-[0-9][^/]*)$")
KERNEL_CMD = re.compile(
    r"\b(modprobe|insmod|rmmod|depmod|update-initramfs|mkinitcpio|dracut|"
    r"dkms|kexec|grub-install|update-grub|grub-mkconfig|make-kpkg|bootctl|"
    r"flash-kernel)\b")
KERNEL_PKG = re.compile(
    r"\b(linux-image|linux-headers|linux-modules|linux-firmware|"
    r"raspberrypi-kernel|firmware-[a-z]|kernel-|dkms)\S*")
NEGATION = re.compile(r"\b(no|not|never|don'?t|do\s+\*\*not\*\*|without)\b",
                      re.I)
SCORE = re.compile(r"(\d+(\.\d+)?\s*%|score\s*[:=]\s*\d|\b\d+(\.\d+)?\s*/\s*100\b)",
                   re.I)

CP_RE = re.compile(r'^\s*cp\s+(?:-\w+\s+)*(.+?)\s*$')
HEREDOC_RE = re.compile(r'^\s*cat\s+>\s*"?(\$(?:DEST|OUT)/[^"\s]+)"?\s*<<')


# ---------------------------------------------------------------- parsing

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


def parse_manifest(script: str, dest_var: str):
    """Return [(src, dst_rel)] for every cp / heredoc target under $DEST_VAR."""
    rows = []
    for line in _code_lines(script):
        m = HEREDOC_RE.match(line)
        if m:
            dst = m.group(1).split("/", 1)[1]
            rows.append(("<heredoc>", dst))
            continue
        m = CP_RE.match(line)
        if not m:
            continue
        toks = [_unquote(t) for t in re.findall(r'"[^"]*"|\S+', m.group(1))]
        if len(toks) < 2:
            continue
        dst = toks[-1]
        if not dst.startswith("$" + dest_var + "/"):
            continue
        dst_rel = dst.split("/", 1)[1]
        for src in toks[:-1]:
            name = src.rsplit("/", 1)[-1]
            target = dst_rel if not dst_rel.endswith("/") else dst_rel + name
            if dst_rel.endswith("junioros/") or dst_rel.endswith("junioros"):
                target = dst_rel.rstrip("/") + "/" + name
            rows.append((src, target))
    return rows


def _src_path(src: str, here: Path) -> Path | None:
    if src == "<heredoc>":
        return None
    return here / src.replace("$HERE/", "", 1) if src.startswith("$HERE/") else None


def _is_text(p: Path) -> bool:
    data = p.read_bytes()
    return not data.startswith(b"\x7fELF") and b"\x00" not in data


# ---------------------------------------------------------------- checks

def run_checks(root: Path):
    rl = root / "rails" / "linux"
    deb = rl / "debian"
    res = []

    def row(key, trit, note):
        res.append((key, trit, note))

    def read(p: Path) -> str:
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""

    inst = read(rl / "install-overlay.sh")
    stage = read(deb / "stage_overlay.sh")
    manifest = [("install", s, d) for s, d in parse_manifest(inst, "DEST")]
    manifest += [("stage", s, d) for s, d in parse_manifest(stage, "OUT")]
    row("manifest.parsed", 1 if len(manifest) >= 8 else -1,
        f"{len(manifest)} staged targets read from install-overlay.sh + stage_overlay.sh")

    bad_dir = [d for _, _, d in manifest if d.startswith(KERNEL_DIRS)]
    row("manifest.no_boot_modules_firmware", -1 if bad_dir else 1,
        "kernel dir: " + ", ".join(bad_dir) if bad_dir else
        "nothing lands in /boot, /lib/modules or firmware")

    bad_name = [d for _, s, d in manifest
                if KERNEL_NAME.search(d) or KERNEL_NAME.search(s.rsplit("/", 1)[-1])]
    row("manifest.no_kernel_artifact_name", -1 if bad_name else 1,
        "kernel-named: " + ", ".join(bad_name) if bad_name else
        "no vmlinuz/bzImage/initrd/*.ko/System.map/*.dtb/*.efi staged")

    missing, blobs = [], []
    for _who, s, d in manifest:
        # install-overlay.sh and stage_overlay.sh both resolve $HERE to rails/linux
        p = _src_path(s, rl)
        if p is None:
            continue
        if not p.is_file():
            missing.append(s)
        elif not _is_text(p):
            blobs.append(s)
    row("manifest.sources_exist", -1 if missing else 1,
        "missing: " + ", ".join(missing) if missing else "every cp source is in the tree")
    row("manifest.sources_text_not_blob", -1 if blobs else 1,
        "binary: " + ", ".join(blobs) if blobs else "every staged source is text (no ELF, no NUL)")

    cmd_hits = []
    for name, text in (("install-overlay.sh", inst), ("stage_overlay.sh", stage),
                       ("preseed.fragment", read(deb / "preseed.fragment"))):
        for line in text.splitlines():
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            if KERNEL_CMD.search(ln):
                cmd_hits.append(f"{name}: {ln}")
    row("scripts.no_kernel_mutating_cmd", -1 if cmd_hits else 1,
        "; ".join(cmd_hits) if cmd_hits else
        "no modprobe/insmod/update-initramfs/dkms/kexec/grub in overlay scripts")

    pkgs = [l.strip() for l in read(deb / "packages.list").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    for line in read(deb / "preseed.fragment").splitlines():
        if line.strip().startswith("d-i pkgsel/include"):
            pkgs += line.split("string", 1)[-1].split()
    bad_pkg = [p for p in pkgs if KERNEL_PKG.search(p)]
    row("packages.no_kernel_package", -1 if bad_pkg else (1 if pkgs else 0),
        "kernel pkg: " + ", ".join(bad_pkg) if bad_pkg else
        f"{len(pkgs)} packages, none linux-image/headers/firmware" if pkgs else
        "no package list found")

    kc_targets = [d for _, s, d in manifest if s.endswith("kconfig.junior")]
    kc = read(rl / "kconfig.junior")
    kc_ok = (kc_targets and all(t.startswith("usr/lib/junioros/") for t in kc_targets)
             and all(l.startswith(("#", "CONFIG_")) or not l.strip()
                     for l in kc.splitlines())
             and "vmlinuz" in kc.splitlines()[0].lower() if kc else False)
    row("kconfig.text_fragment_not_boot_config",
        1 if kc_ok else (0 if not kc_targets else -1),
        f"staged to {', '.join(kc_targets)} as a want-list; never /boot/config-*"
        if kc_ok else f"kconfig targets {kc_targets or 'none'}")

    unit = read(rl / "bitnetd.service")
    exe = re.search(r"^ExecStart=(\S+)", unit, re.M)
    exe_rel = exe.group(1).lstrip("/") if exe else ""
    installed = {d for _, _, d in manifest}
    bind = re.search(r"JUNIOR_BIND=(\S+)", unit)
    ok = exe_rel in installed and bind and bind.group(1) == PORTS["bitnetd"]
    row("port.bitnetd_userspace_loopback", 1 if ok else -1,
        f"ExecStart /{exe_rel} is overlay-installed, bind {bind.group(1) if bind else '?'}"
        if exe else "no ExecStart")
    row("port.no_wildcard_in_staged_units",
        -1 if "0.0.0.0" in unit + read(rl / "os-release.junior") else 1,
        "no 0.0.0.0 in bitnetd.service / os-release.junior")

    i2 = read(rl / "i2sd.py")
    i2_ok = re.search(r'^HOST\s*=\s*"127\.0\.0\.1"', i2, re.M) and \
        re.search(r"^PORT\s*=\s*8767\b", i2, re.M)
    i2_staged = [d for _, s, d in manifest if s.endswith("i2sd.py")]
    row("port.i2sd_userspace_loopback", 1 if i2_ok else -1,
        f"i2sd.py HOST 127.0.0.1 PORT 8767, staged as {', '.join(i2_staged) or 'nothing'}")
    i2_inst = [d for w, s, d in manifest if w == "install" and "i2sd" in (s + d)]
    row("port.i2sd_installed_by_overlay", 1 if i2_inst else 0,
        "install-overlay.sh installs i2sd" if i2_inst else
        "advisory: KERNEL.md lists i2sd in the userspace overlay, but only "
        "stage_overlay.sh copies it (no unit, not on PATH)")

    osr = read(rl / "os-release.junior")
    score_keys = [l.split("=", 1)[0] for l in osr.splitlines()
                  if "=" in l and not l.startswith("#")
                  and re.search(r"osworld|score|kernel", l.split("=", 1)[0], re.I)]
    row("osworld.os_release_no_score_key", -1 if score_keys else 1,
        "keys: " + ", ".join(score_keys) if score_keys else "no OSWorld/score/kernel key")
    osr_neg = any("osworld" in l.lower() and NEGATION.search(l)
                  for l in osr.splitlines() if l.startswith("#"))
    osr_staged = any(d.endswith("os-release.junior") for _, _, d in manifest)
    row("osworld.negation_travels_with_install", 1 if osr_neg and osr_staged else 0,
        "installed /etc/os-release.junior carries the 'No OSWorld score' line"
        if osr_neg and osr_staged else "negation not carried into the installed identity text")

    scan = sorted((root / "docs").glob("*.md")) + sorted(rl.rglob("*.md")) + \
        [rl / "os-release.junior", root / "grok_bot" / "skills" / "linux-os" / "SKILL.md"]
    mentions, claims = 0, []
    for p in scan:
        if not p.is_file():
            continue
        for n, line in enumerate(read(p).splitlines(), 1):
            if "osworld" not in line.lower():
                continue
            mentions += 1
            if not NEGATION.search(line) or SCORE.search(line):
                claims.append(f"{p.relative_to(root)}:{n}")
    row("osworld.every_mention_is_negation", -1 if claims else (1 if mentions else 0),
        "claim-like: " + ", ".join(claims) if claims else
        f"{mentions} OSWorld mentions, all negations, no numeric score")

    sources = {
        "docs/BETA_TO_OS.md": read(root / "docs" / "BETA_TO_OS.md"),
        "rails/linux/KERNEL.md": read(rl / "KERNEL.md"),
        "rails/linux/ROADMAP.md": read(rl / "ROADMAP.md"),
        "rails/linux/debian/WHY_NOT_VMLINUZ.md": read(deb / "WHY_NOT_VMLINUZ.md"),
        "rails/linux/install-overlay.sh": inst,
        "rails/linux/kconfig.junior": kc,
    }
    agree = [k for k, t in sources.items()
             if re.search(r"(no kernel blob|not a (custom )?vmlinuz|do not ship vmlinuz|"
                          r"not a custom kernel|not a vmlinuz)", t, re.I)]
    row("docs.no_kernel_blob_agreement",
        1 if len(agree) == len(sources) else (0 if len(agree) >= 3 else -1),
        f"{len(agree)}/{len(sources)} overlay sources state no kernel blob")
    return res


# ---------------------------------------------------------------- close

def pack_trits(trits):
    """Five trits per byte, base 3, digit = trit + 1."""
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            v = v * 3 + (t + 1)
        out.append(v)
    return bytes(out)


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def report(root: Path, prev: str = GENESIS):
    rows = run_checks(root)
    trits = [t for _, t, _ in rows]
    pack = pack_trits(trits)
    text = "\n".join(f"{t:+d} {k} {n}" for k, t, n in rows)
    return {
        "item": ITEM,
        "ports": PORTS,
        "flags": FLAGS,
        "rows": [{"key": k, "trit": t, "note": n} for k, t, n in rows],
        "pass": trits.count(1), "advisory": trits.count(0), "fail": trits.count(-1),
        "pack": pack.hex(),
        "prev": prev,
        "close": close(prev, pack, text),
    }


# ---------------------------------------------------------------- selftest

def _bad_tree(tmp: Path):
    rl = tmp / "rails" / "linux"
    (rl / "debian").mkdir(parents=True)
    (tmp / "docs").mkdir()
    (rl / "install-overlay.sh").write_text(
        '#!/bin/sh\nDEST=${DEST:-}\n'
        'cp "$HERE/vmlinuz-junior" "$DEST/boot/vmlinuz-junior"\n'
        'cp "$HERE/junior.ko" "$DEST/lib/modules/6.1/junior.ko"\n'
        'modprobe junior\nupdate-initramfs -u\n')
    (rl / "vmlinuz-junior").write_bytes(b"\x7fELF\x00\x00")
    (rl / "junior.ko").write_bytes(b"\x7fELF\x00")
    (rl / "debian" / "stage_overlay.sh").write_text(
        'cp "$HERE/kconfig.junior" "$OUT/boot/config-6.1-junior"\n')
    (rl / "kconfig.junior").write_text("CONFIG_X=y\n")
    (rl / "debian" / "packages.list").write_text("python3\nlinux-image-amd64\n")
    (rl / "bitnetd.service").write_text(
        "[Service]\nExecStart=/usr/bin/kernel-bitnetd\nEnvironment=JUNIOR_BIND=0.0.0.0:8765\n")
    (rl / "os-release.junior").write_text('ID=junioros\nOSWORLD_SCORE="42"\n')
    (rl / "i2sd.py").write_text('HOST = "0.0.0.0"\nPORT = 9999\n')
    (tmp / "docs" / "BETA_TO_OS.md").write_text("JuniorOS hits 61.2% on OSWorld.\n")


def selftest() -> int:
    ok = True
    real = report(ROOT)
    if real["fail"]:
        print("selftest: real tree has fail rows", file=sys.stderr)
        ok = False
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _bad_tree(tmp)
        bad = {r["key"]: r["trit"] for r in report(tmp)["rows"]}
    must_fail = ("manifest.no_boot_modules_firmware", "manifest.no_kernel_artifact_name",
                 "manifest.sources_text_not_blob", "scripts.no_kernel_mutating_cmd",
                 "packages.no_kernel_package", "kconfig.text_fragment_not_boot_config",
                 "port.bitnetd_userspace_loopback", "port.no_wildcard_in_staged_units",
                 "port.i2sd_userspace_loopback", "osworld.os_release_no_score_key",
                 "osworld.every_mention_is_negation", "docs.no_kernel_blob_agreement")
    for k in must_fail:
        if bad.get(k) != -1:
            print(f"selftest: bad fixture did not fail {k} ({bad.get(k)})", file=sys.stderr)
            ok = False
    if pack_trits([1, 0, -1, 1, 1]) != bytes([2 + 3 * (1 + 3 * (0 + 3 * (2 + 3 * 2)))]):
        print("selftest: pack order", file=sys.stderr)
        ok = False
    a = close(GENESIS, b"\x01", "x")
    if a != hashlib.sha256(b"0" * 16 + b"\x01" + b"x").hexdigest()[:16] or len(a) != 16:
        print("selftest: close rule", file=sys.stderr)
        ok = False
    if any(FLAGS[k] for k in ("mint", "socket", "layer_one", "fetch", "ue5_launch")) \
            or FLAGS["rate"] != 0.05 or FLAGS["bind"] != "127.0.0.1":
        print("selftest: flags", file=sys.stderr)
        ok = False
    print(f"selftest {'pass' if ok else 'FAIL'}: real {real['pass']} pass / "
          f"{real['advisory']} advisory / {real['fail']} fail, "
          f"bad fixture failed {sum(1 for k in must_fail if bad.get(k) == -1)}/{len(must_fail)}, "
          f"close {real['close']}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prev", default=GENESIS, help="previous close (16 hex)")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    r = report(Path(a.root), a.prev)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        for row in r["rows"]:
            print(f"{row['trit']:+d} {row['key']}: {row['note']}")
        print(f"{r['pass']} pass, {r['advisory']} advisory, {r['fail']} fail; "
              f"ports bitnetd {PORTS['bitnetd']} i2sd {PORTS['i2sd']}; close {r['close']}")
    return 1 if r["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
