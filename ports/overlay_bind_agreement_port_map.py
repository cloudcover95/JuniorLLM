"""JuniorOS overlay bind agreement + user-unit parity port map (in-process, no sockets).

Advances JuniorLLM docs/BETA_TO_OS.md, item "JuniorOS overlay":
Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH.

New angle (genuinely uncovered piece): install-overlay.sh (C7) writes three files
from heredocs that no existing check reads: etc/profile.d/junioros.sh,
etc/junioros/path.env and the systemd --user unit usr/lib/systemd/user/bitnetd.service.
This port map extracts those heredocs as plain text and checks, in-process:
  * every bind declaration agrees on 127.0.0.1:8765: os-release.junior
    JUNIOR_BITNETD, the system unit, the user unit, the profile.d default,
    path.env and the installer default (never 0.0.0.0);
  * the user unit is hardened like the system unit (shared core directives
    must be present in both; system-only extras are listed as an advisory);
  * the juniorctl PATH pin dir agrees across profile.d, path.env and the
    installer copy target, and path.env names the user unit the installer writes;
  * the Home / OSai / FieldCore / Flagstaff / JuniorLLM port map, with the
    live-beta handshake order juniorctl health -> bitnetd 127.0.0.1:8765 ->
    i2sd 127.0.0.1:8767, routes to loopback only.

Ports are listed, never opened. No socket, no subprocess, no network, no fetch,
no model pull, no UE5. Reads only rails/linux text files. Does not import, wrap
or re-check ports/overlay_port_map.py or any other team file.
Stdlib only. Rate 0.05. Close rule: sha256(prev || pack || text)[:16], trits
packed as 2-bit i2s (0->0, 1->1, -1->2), MSB first, zero-padded.

Run: python ports/overlay_bind_agreement_port_map.py --selftest   (exit 0 on pass)
     python ports/overlay_bind_agreement_port_map.py              (real tree, JSON)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"

ROADMAP_ITEM = "BETA_TO_OS JuniorOS overlay"
LOOPBACK = "127.0.0.1"
BITNETD = f"{LOOPBACK}:8765"
I2SD = f"{LOOPBACK}:8767"
RATE = 0.05
PACK = "overlay-bind-agreement"
GENESIS = "0" * 16

FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}

PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
PORT_MAP = {
    "Home": ("bitnetd", "i2sd"),
    "OSai": ("bitnetd", "i2sd"),
    "FieldCore": ("bitnetd",),
    "Flagstaff": ("bitnetd",),
    "JuniorLLM": ("bitnetd", "i2sd"),
}
SERVICES = {"bitnetd": BITNETD, "i2sd": I2SD}
HANDSHAKE_ORDER = ("juniorctl health", f"bitnetd {BITNETD}", f"i2sd {I2SD}")

CORE_HARDENING = (
    "NoNewPrivileges=yes",
    "ProtectSystem=strict",
    "PrivateTmp=yes",
    "MemoryDenyWriteExecute=yes",
    "RestrictNamespaces=yes",
    "CapabilityBoundingSet=",
    "AmbientCapabilities=",
    "MemoryMax=2G",
    "TasksMax=256",
)

HEREDOCS = {
    "profile": "/etc/profile.d/junioros.sh",
    "user_unit": "/usr/lib/systemd/user/bitnetd.service",
    "path_env": "/etc/junioros/path.env",
}


def trit_pack(trits: list) -> bytes:
    code = {0: 0, 1: 1, -1: 2}
    bits = "".join(format(code[t], "02b") for t in trits)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def heredocs(installer: str) -> dict:
    out = {}
    for key, suffix in HEREDOCS.items():
        pat = re.compile(r'cat > "\$DEST' + re.escape(suffix) + r"\" <<'EOF'\n(.*?)\nEOF", re.S)
        m = pat.search(installer)
        out[key] = m.group(1) if m else ""
    return out


def unit_lines(text: str) -> set:
    return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")}


def kv(text: str) -> dict:
    d = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            d[k.strip()] = v.strip().strip('"')
    return d


def env_bind(unit: str) -> str:
    m = re.search(r"^Environment=JUNIOR_BIND=(\S+)$", unit, re.M)
    return m.group(1) if m else ""


def evaluate(os_release: str, sys_unit: str, installer: str) -> list:
    """Return rows (name, trit, detail). trit 1 pass, 0 advisory, -1 fail."""
    rows = []
    hd = heredocs(installer)
    rows.append(("heredocs_found", 1 if all(hd.values()) else -1,
                 {k: bool(v) for k, v in hd.items()}))

    prof_default = re.search(r'JUNIOR_BIND="\$\{JUNIOR_BIND:-([^}]+)\}"', hd["profile"])
    inst_default = re.search(r"BIND=\$\{JUNIOR_BIND:-([^}]+)\}", installer)
    binds = {
        "os_release.JUNIOR_BITNETD": kv(os_release).get("JUNIOR_BITNETD", ""),
        "system_unit": env_bind(sys_unit),
        "user_unit": env_bind(hd["user_unit"]),
        "profile_default": prof_default.group(1) if prof_default else "",
        "path_env": kv(hd["path_env"]).get("JUNIOR_BIND", ""),
        "installer_default": inst_default.group(1) if inst_default else "",
    }
    rows.append(("bind_agreement", 1 if set(binds.values()) == {BITNETD} else -1, binds))

    blob = "\n".join([os_release, sys_unit] + list(hd.values()))
    rows.append(("no_wildcard_bind", -1 if "0.0.0.0" in blob else 1, "0.0.0.0 absent"))

    su, uu = unit_lines(sys_unit), unit_lines(hd["user_unit"])
    for name, lines in (("system_unit", su), ("user_unit", uu)):
        ok = "Environment=JUNIOR_OFFLINE=1" in lines and "ExecStart=/usr/local/bin/junior-bitnetd" in lines
        rows.append((f"{name}_offline_exec", 1 if ok else -1, "JUNIOR_OFFLINE=1 + junior-bitnetd"))

    missing = [d for d in CORE_HARDENING if d not in su or d not in uu]
    rows.append(("core_hardening_parity", -1 if missing else 1, missing or "all core in both"))

    extras = sorted(ln for ln in su - uu
                    if "=" in ln and not ln.startswith(("Description=", "After=", "WantedBy=",
                                                        "User=", "Group=", "Environment=")))
    rows.append(("user_unit_extra_hardening", 0 if extras else 1,
                 {"system_only": extras} if extras else "none"))

    targets = ("WantedBy=multi-user.target" in su, "WantedBy=default.target" in uu)
    rows.append(("install_targets", 1 if all(targets) else -1,
                 "system multi-user.target, user default.target"))

    pe = kv(hd["path_env"])
    prof_bin = re.search(r'JUNIOROS_BIN="([^"]+)"', hd["profile"])
    pin_dirs = {
        "profile": prof_bin.group(1) if prof_bin else "",
        "path_env": pe.get("JUNIOROS_BIN", ""),
        "installer": "/usr/local/bin" if '"$DEST/usr/local/bin/juniorctl"' in installer else "",
    }
    rows.append(("juniorctl_path_pin", 1 if set(pin_dirs.values()) == {"/usr/local/bin"} else -1, pin_dirs))

    uref = pe.get("JUNIOR_USER_UNIT", "")
    rows.append(("path_env_user_unit_ref", 1 if uref and ("/" + uref) == HEREDOCS["user_unit"] else -1, uref))

    refuses = re.search(r"127\.0\.0\.1\|localhost\|::1\)", installer) and "refusing non-loopback" in installer
    rows.append(("installer_refuses_non_loopback", 1 if refuses else -1, "case guard present"))

    for port in PORTS:
        lanes = {s: SERVICES[s] for s in PORT_MAP[port]}
        ok = all(v.startswith(LOOPBACK + ":") for v in lanes.values()) and lanes.get("bitnetd") == BITNETD
        rows.append((f"port_map_{port}", 1 if ok else -1, lanes))

    rows.append(("handshake_order", 1, list(HANDSHAKE_ORDER)))
    rows.append(("flags_false", 1 if not any(FLAGS.values()) and RATE == 0.05 else -1,
                 {**FLAGS, "rate": RATE}))
    return rows


def chain(rows: list) -> list:
    prev, out = GENESIS, []
    for name, trit, detail in rows:
        text = json.dumps({"row": name, "trit": trit, "detail": detail}, sort_keys=True)
        prev = close(prev, trit_pack([trit]) + PACK.encode(), text)
        out.append({"row": name, "trit": trit, "detail": detail, "close": prev})
    return out


def report(rows: list) -> dict:
    chained = chain(rows)
    trits = [r["trit"] for r in chained]
    return {
        "item": ROADMAP_ITEM,
        "ports": list(PORTS),
        "pass": trits.count(1),
        "advisory": trits.count(0),
        "fail": trits.count(-1),
        "rows": len(trits),
        "pack_hex": trit_pack(trits).hex(),
        "tip": chained[-1]["close"],
        "ok": -1 not in trits,
        "detail": chained,
    }


def real_tree() -> dict:
    read = lambda n: (LINUX / n).read_text(encoding="utf-8")
    return report(evaluate(read("os-release.junior"), read("bitnetd.service"), read("install-overlay.sh")))


def selftest() -> int:
    assert trit_pack([1, -1, 0, 1]).hex() == "61"
    assert len(close(GENESIS, b"\x01", "x")) == 16
    good = real_tree()
    # Negative: a wildcard bind in the user unit must fail closed.
    ins = (LINUX / "install-overlay.sh").read_text(encoding="utf-8")
    bad_ins = ins.replace("Environment=JUNIOR_BIND=127.0.0.1:8765\nNoNewPrivileges",
                          "Environment=JUNIOR_BIND=0.0.0.0:8765\nNoNewPrivileges", 1)
    bad = report(evaluate((LINUX / "os-release.junior").read_text(encoding="utf-8"),
                          (LINUX / "bitnetd.service").read_text(encoding="utf-8"), bad_ins))
    assert bad_ins != ins and not bad["ok"], "negative case did not fail"
    print(f"{ROADMAP_ITEM}: {good['pass']}/{good['rows']} pass, {good['advisory']} advisory, "
          f"{good['fail']} fail, tip {good['tip']}, negative case fails closed")
    for r in good["detail"]:
        if r["trit"] != 1:
            print(f"  {'ADVISORY' if r['trit'] == 0 else 'FAIL'} {r['row']}: {json.dumps(r['detail'])}")
    return 0 if good["ok"] else 1


def main(argv: list) -> int:
    if "--selftest" in argv:
        return selftest()
    rep = real_tree()
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
