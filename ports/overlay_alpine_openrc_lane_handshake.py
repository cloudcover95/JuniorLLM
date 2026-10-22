"""JuniorOS overlay Alpine init-lane handshake (OpenRC vs systemd), in-process, no sockets.

Advances JuniorLLM docs/BETA_TO_OS.md, item "JuniorOS overlay":
Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH.

New angle (uncovered piece): os-release.junior declares ID_LIKE="debian alpine",
but the overlay ships only systemd units. Alpine boots OpenRC with busybox, has
no systemd, and nothing in rails/linux mentions OpenRC, so on the Alpine lane
bitnetd (127.0.0.1:8765) would never start and juniorctl health would report
it down. This file maps both init lanes from the handshake side, reading text
only:
  * D. Debian lane: system unit installed, ExecStart target installed, enable
    target named (WantedBy=multi-user.target);
  * A. Alpine lane: is an OpenRC service (etc/init.d/bitnetd, #!/sbin/openrc-run)
    written by install-overlay.sh? does packages.list name an apk set, or only
    Debian names (python3-venv, systemd) that apk cannot install? which systemd
    hardening directives have no OpenRC carrier?
  * S. busybox ash reach: bitnetd.sh and install-overlay.sh are #!/bin/sh and
    must stay free of bashisms;
  * B. bind: every bind on both lanes is 127.0.0.1:8765 (bitnetd) or
    127.0.0.1:8767 (i2sd fallback), never 0.0.0.0;
  * P. port map: Home / OSai / FieldCore / Flagstaff / JuniorLLM, each routed
    juniorctl health -> bitnetd 127.0.0.1:8765 -> i2sd 127.0.0.1:8767, per lane.
A proposed OpenRC script is emitted as text in the report (never written) and
is itself checked for loopback bind and a non-root command_user.

Trits: +1 holds, 0 advisory (a real gap, reported, run still exits 0), -1
broken (exit 1: wildcard bind, port drift, a bashism in a /bin/sh script that
busybox ash rejects, a flag set true). Ports are listed, never opened. No
socket, no subprocess, no network, no fetch, no model pull, no UE5, no
os.environ reads. Reads only rails/linux text files. Does not import, wrap or
re-check any team file under ports/ or scripts/.
Flags: mint false, socket false, layer_one false, fetch false, ue5_launch false.
Rate 0.05 (committed, never polled). Close rule sha256(prev || pack || text)[:16],
trits packed 2-bit i2s (0->0, 1->1, -1->2), MSB first, zero-padded.

Run: python ports/overlay_alpine_openrc_lane_handshake.py --selftest  (exit 0 on pass)
     python ports/overlay_alpine_openrc_lane_handshake.py             (real tree, JSON)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"

RATE = 0.05
MINT = SOCKET = LAYER_ONE = FETCH = UE5_LAUNCH = False
LOOPBACK = "127.0.0.1"
BITNETD = LOOPBACK + ":8765"
I2SD = LOOPBACK + ":8767"
WILDCARD = ".".join(["0"] * 4)
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
GENESIS = "0" * 16
SOURCES = {
    "os_release": "os-release.junior",
    "unit": "bitnetd.service",
    "installer": "install-overlay.sh",
    "wrapper": "bitnetd.sh",
    "packages": "debian/packages.list",
    "i2sd": "i2sd.py",
}
DEBIAN_ONLY_PKGS = {"python3-venv": "apk: python3 (venv is built in)",
                    "systemd": "apk: openrc (systemd not packaged)"}
# systemd hardening -> nearest OpenRC carrier (None = no carrier on OpenRC)
OPENRC_CARRIER = {
    "User": "command_user", "Group": "command_user",
    "Restart": "supervisor=supervise-daemon + respawn_delay",
    "Environment": "export in the init script",
    "MemoryMax": "rc_cgroup_settings memory.max",
    "TasksMax": "rc_cgroup_settings pids.max",
    "NoNewPrivileges": "capabilities / no_new_privs (OpenRC >= 0.45)",
    "CapabilityBoundingSet": "capabilities=^",
}
BASHISMS = [
    (r"\[\[", "[[ ]]"), (r"^\s*function\s+\w+", "function keyword"),
    (r"^\s*source\s", "source"), (r"\$'", "$'..' quoting"),
    (r"<<<", "here-string"), (r"\w+=\(", "array assignment"),
    (r"\$\{\w+\[", "array index"), (r"\$\{\w+//", "${v//} substitution"),
]

PROPOSED_OPENRC = """#!/sbin/openrc-run
description="Junior BitNet local inference (loopback)"
supervisor=supervise-daemon
command=/usr/local/bin/junior-bitnetd
command_user="nobody:nobody"
respawn_delay=2
export JUNIOR_OFFLINE=1
export JUNIOR_BIND=127.0.0.1:8765
rc_cgroup_settings="memory.max 2147483648
pids.max 256"
depend() {
  need localmount loopback
}
"""


def pack(trits) -> bytes:
    enc = {0: 0, 1: 1, -1: 2}
    acc = bits = 0
    out = bytearray()
    for t in trits:
        acc = (acc << 2) | enc[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def load(base: Path) -> dict:
    out = {}
    for k, rel in SOURCES.items():
        p = base / rel
        out[k] = p.read_text(encoding="utf-8") if p.is_file() else ""
    return out


def unit_keys(text: str) -> dict:
    keys: dict = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith(("#", ";", "[")) and "=" in ln:
            k, v = ln.split("=", 1)
            keys.setdefault(k.strip(), []).append(v.strip())
    return keys


def bashisms(text: str) -> list:
    hits = []
    body = re.sub(r"<<'?EOF'?\n.*?\nEOF\n", "", text, flags=re.S)  # heredoc bodies are data
    for ln in body.splitlines():
        s = ln.split("#", 1)[0] if not ln.lstrip().startswith("#!") else ""
        for pat, name in BASHISMS:
            if re.search(pat, s):
                hits.append(name)
    return sorted(set(hits))


def binds(text: str) -> list:
    return re.findall(r"(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}", text)


def rows_for(s: dict) -> list:
    rows = []

    def row(rid, trit, note):
        rows.append({"id": rid, "trit": trit, "note": note})

    like = re.search(r'^ID_LIKE="?([^"\n]*)"?', s["os_release"], re.M)
    like = like.group(1).split() if like else []
    row("os_release.id_like", 1 if {"debian", "alpine"} <= set(like) else 0,
        "ID_LIKE=" + " ".join(like))

    # D. Debian lane
    uk = unit_keys(s["unit"])
    inst = s["installer"]
    row("D.unit_installed", 1 if "etc/systemd/system/bitnetd.service" in inst else -1,
        "install-overlay.sh copies bitnetd.service to etc/systemd/system")
    exe = (uk.get("ExecStart") or [""])[0].split()[:1]
    row("D.exec_installed", 1 if exe and exe[0].lstrip("/") in inst else -1,
        "ExecStart %s is a copy target" % (exe[0] if exe else "?"))
    row("D.enable_target", 1 if "multi-user.target" in uk.get("WantedBy", []) else 0,
        "WantedBy=" + ",".join(uk.get("WantedBy", [])))

    # A. Alpine lane
    has_rc = "init.d/bitnetd" in inst and "openrc-run" in inst
    row("A.openrc_service", 1 if has_rc else 0,
        "OpenRC init.d/bitnetd written" if has_rc else
        "no etc/init.d/bitnetd: Alpine has no systemd, bitnetd never starts there")
    pk = [p.strip() for p in s["packages"].splitlines() if p.strip() and not p.startswith("#")]
    deb_only = [p for p in pk if p in DEBIAN_ONLY_PKGS]
    has_apk = "apk" in inst or "openrc" in pk
    row("A.package_set", 1 if has_apk or not deb_only else 0,
        "apk set present" if has_apk else
        "packages.list is Debian-only: " + "; ".join("%s -> %s" % (p, DEBIAN_ONLY_PKGS[p]) for p in deb_only))
    hard = [k for k in uk if k in ("NoNewPrivileges", "ProtectSystem", "ProtectHome", "PrivateTmp",
            "PrivateDevices", "MemoryDenyWriteExecute", "RestrictAddressFamilies",
            "RestrictNamespaces", "SystemCallFilter", "CapabilityBoundingSet",
            "MemoryMax", "TasksMax", "User", "Group", "Restart")]
    lost = sorted(k for k in hard if k not in OPENRC_CARRIER)
    row("A.hardening_carrier", 1 if has_rc and not lost else 0,
        "%d of %d hardening directives have an OpenRC carrier; no carrier: %s"
        % (len(hard) - len(lost), len(hard), ", ".join(lost) or "none"))

    # S. busybox ash reach
    for k in ("wrapper", "installer"):
        shebang = s[k].splitlines()[0] if s[k] else ""
        hits = bashisms(s[k])
        if not s[k]:
            row("S." + k, 0, SOURCES[k] + " missing")
        elif shebang.strip() != "#!/bin/sh":
            row("S." + k, 0, "shebang " + shebang)
        else:
            row("S." + k, -1 if hits else 1, "bashisms: " + (", ".join(hits) or "none"))

    # B. bind agreement across lanes
    all_text = s["unit"] + s["installer"] + s["wrapper"] + s["os_release"] + PROPOSED_OPENRC
    w = re.escape(WILDCARD)
    wild = re.findall(w + r":\d+|--host\s+" + w + r"|BIND=\S*" + w, all_text)
    row("B.no_wildcard", -1 if wild else 1,
        "wildcard bind: " + ", ".join(wild) if wild else "no wildcard bind on either lane"
        " (guard/comment mentions are not binds)")
    seen = sorted(set(binds(all_text)))
    drift = [b for b in seen if b not in (BITNETD, I2SD)]
    row("B.bitnetd_8765", 1 if BITNETD in seen and not drift else -1,
        "binds seen: " + ", ".join(seen) + ("; drift: " + ", ".join(drift) if drift else ""))
    i2 = re.search(r"^PORT\s*=\s*(\d+)", s["i2sd"], re.M)
    row("B.i2sd_8767", 1 if i2 and i2.group(1) == "8767" else -1,
        "i2sd.py PORT=" + (i2.group(1) if i2 else "?"))
    prc = PROPOSED_OPENRC
    row("B.proposal_loopback", 1 if ("JUNIOR_BIND=" + BITNETD) in prc and "root" not in prc else -1,
        "proposed init.d/bitnetd binds %s as nobody" % BITNETD)

    # flags
    row("F.flags", -1 if any((MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH)) else 1,
        "mint/socket/layer_one/fetch/ue5_launch false")
    return rows


def lanes(rows: list) -> list:
    t = {r["id"]: r["trit"] for r in rows}
    out = []
    for port in PORTS:
        for lane, key in (("debian/systemd", "D.unit_installed"), ("alpine/openrc", "A.openrc_service")):
            out.append({"port": port, "lane": lane,
                        "route": ["juniorctl health", "bitnetd " + BITNETD, "i2sd " + I2SD],
                        "starts": t[key] == 1})
    return out


def report(s: dict, prev: str = GENESIS) -> dict:
    rows = rows_for(s)
    h = prev
    for r in rows:
        r["close"] = h = close(h, pack([r["trit"]]), json.dumps(r, sort_keys=True))
    trits = [r["trit"] for r in rows]
    pm = lanes(rows)
    text = json.dumps({"rows": rows, "port_map": pm}, sort_keys=True)
    return {
        "item": 'JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" (Debian/Alpine init lanes)',
        "rate": RATE, "bind": LOOPBACK,
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
        "rows": rows, "port_map": pm,
        "pass": trits.count(1), "advisory": trits.count(0), "fail": trits.count(-1),
        "pack": pack(trits).hex(),
        "close": close(h, pack(trits), text),
        "proposed_openrc": PROPOSED_OPENRC,
    }


def selftest() -> int:
    ok = True

    def t(cond, msg):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok = ok and cond

    good = {
        "os_release": 'ID_LIKE="debian alpine"\nJUNIOR_BITNETD="127.0.0.1:8765"\n',
        "unit": "[Service]\nUser=nobody\nExecStart=/usr/local/bin/junior-bitnetd\n"
                "Environment=JUNIOR_BIND=127.0.0.1:8765\nNoNewPrivileges=yes\nMemoryMax=2G\n"
                "[Install]\nWantedBy=multi-user.target\n",
        "installer": "#!/bin/sh\nBIND=${JUNIOR_BIND:-127.0.0.1:8765}\n"
                     'cp a "$DEST/etc/systemd/system/bitnetd.service"\n'
                     'cp b "$DEST/usr/local/bin/junior-bitnetd"\n'
                     "cat > \"$DEST/etc/init.d/bitnetd\" <<'EOF'\n#!/sbin/openrc-run\nEOF\n",
        "wrapper": "#!/bin/sh\nBIND=${JUNIOR_BIND:-127.0.0.1:8765}\nhost=${BIND%:*}\n",
        "packages": "python3\nopenrc\n",
        "i2sd": "PORT = 8767\n",
    }
    r = report(good)
    t(r["fail"] == 0 and r["advisory"] == 0, "good tree: all rows hold (%s)" % r["pack"])
    t(len(r["port_map"]) == 10 and all(x["starts"] for x in r["port_map"]), "10 lanes start")
    t(r["close"] == report(good)["close"], "close is deterministic")
    t(r["close"] != report(good, prev="1" * 16)["close"], "close depends on prev")
    t(pack([1, -1, 0, 1]) == bytes([0b01100001]) and pack([-1]) == bytes([0x80]), "i2s pack MSB first")

    noalp = dict(good, installer=good["installer"].split("cat >")[0], packages="python3\npython3-venv\nsystemd\n")
    r2 = report(noalp)
    tt = {x["id"]: x["trit"] for x in r2["rows"]}
    t(tt["A.openrc_service"] == 0 and tt["A.package_set"] == 0 and r2["fail"] == 0,
      "no OpenRC / Debian-only packages -> advisory, not fail")
    t(not any(x["starts"] for x in r2["port_map"] if x["lane"] == "alpine/openrc"), "alpine lanes do not start")

    wild = dict(good, wrapper=good["wrapper"].replace("127.0.0.1:8765", WILDCARD + ":8765"))
    t({x["id"]: x["trit"] for x in report(wild)["rows"]}["B.no_wildcard"] == -1, "wildcard -> -1")
    drift = dict(good, unit=good["unit"].replace("8765", "9765"))
    t({x["id"]: x["trit"] for x in report(drift)["rows"]}["B.bitnetd_8765"] == -1, "port drift -> -1")
    i2 = dict(good, i2sd="PORT = 8765\n")
    t({x["id"]: x["trit"] for x in report(i2)["rows"]}["B.i2sd_8767"] == -1, "i2sd drift -> -1")
    bash = dict(good, wrapper=good["wrapper"] + "if [[ -n x ]]; then :; fi\n")
    t({x["id"]: x["trit"] for x in report(bash)["rows"]}["S.wrapper"] == -1, "bashism in /bin/sh -> -1")
    guard = dict(good, installer=good["installer"] + "# No %s.\nif grep -Fq '%s' f; then exit 3; fi\n" % (WILDCARD, WILDCARD))
    t({x["id"]: x["trit"] for x in report(guard)["rows"]}["B.no_wildcard"] == 1, "wildcard guard text is not a bind")
    t(bashisms("#!/bin/sh\ncat <<'EOF'\n[[ data ]]\nEOF\n") == [], "heredoc body is data, not script")

    real = report(load(LINUX))
    t(real["fail"] == 0, "real tree: no hard fail (pass %d, advisory %d)" % (real["pass"], real["advisory"]))
    print("close", real["close"])
    return 0 if ok else 1


def main(argv) -> int:
    if "--selftest" in argv:
        return selftest()
    r = report(load(LINUX))
    print(json.dumps(r, indent=2, sort_keys=True))
    return 1 if r["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
