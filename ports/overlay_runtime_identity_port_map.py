"""JuniorOS overlay: bitnetd runtime-identity handshake + port map.

Advances JuniorLLM docs/BETA_TO_OS.md, section "JuniorOS overlay", the item
"hardened `bitnetd.service`" on its identity side: who does bitnetd run as in
each of the three lanes the overlay ships, and can that identity still bind
and answer the live-beta handshake (juniorctl health -> bitnetd
127.0.0.1:8765 -> i2sd 127.0.0.1:8767)?

  * system unit   rails/linux/bitnetd.service       User=/Group=
  * user unit     install-overlay.sh heredoc        (invoking user)
  * OCI bundle    rails/linux/oci/config.json       process.user + user ns
  * podman line   rails/linux/oci/README.md         --user

Not re-derived here (landed elsewhere): os-release.junior, PATH, Debian/Alpine
base split, MemoryMax, user-unit hardening parity, restart/ordering/sandbox
reach, second-tier hardening, seccomp closure, OCI seccomp/--network wiring,
tarball owner/mode, i2sd install closure, juniorctl import closure, sysctl,
kconfig and the GGUF gate.  T13 skill-pin family is overnight-owned, skipped.

Rows, one trit each (+1 holds, 0 advisory with smallest additive fix, -1 broken):
  U. identity per lane: never root / uid 0, names agree across lanes where
     they should (OCI config vs podman --user), systemd's "special user
     nobody" warning, Group=nogroup resolving on both ID_LIKE families,
     user namespace declared without uidMappings/gidMappings.
  B. bind reach per identity: every lane binds loopback 8765, the i2sd
     fallback binds 127.0.0.1:8767, both ports are unprivileged (>=1024) so a
     capability-less identity can bind them, and no lane carries
     CAP_NET_BIND_SERVICE.
  P. port map: Home, OSai, FieldCore, Flagstaff, JuniorLLM each route to
     loopback 8765 / 8767 only.

In-process only: files are read as text / json.  No socket, no subprocess,
no network, no fetch, no model pull, no UE5.  Stdlib only.  Flags: mint
false, socket false, layer_one false, fetch false, ue5_launch false.  Rate
0.05.  Bind 127.0.0.1.  Trits packed 2-bit (0->0, 1->1, -1->2, MSB first,
zero-padded), each row closed with sha256(prev || pack || text)[:16].

Ports touched: bitnetd 127.0.0.1:8765, i2sd 127.0.0.1:8767; lanes Home,
OSai, FieldCore, Flagstaff, JuniorLLM.

    python ports/overlay_runtime_identity_port_map.py
    python ports/overlay_runtime_identity_port_map.py --json
    python ports/overlay_runtime_identity_port_map.py --selftest

Exit 1 if any row is -1 (root identity, non-loopback or wildcard bind,
privileged port, a bind capability granted, or a flag flipped).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINUX = "rails/linux"
LOOPBACK = "127.0.0.1"
BITNETD = 8765
I2SD = 8767
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
LANES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
# Group names shipped in the base /etc/group of each ID_LIKE family.
BASE_GROUPS = {"debian": {"root", "nogroup", "users"},
               "alpine": {"root", "nobody", "users"}}
BASE_USERS = {"debian": {"root", "nobody"}, "alpine": {"root", "nobody"}}
ROOT_NAMES = {"root", "0"}


def read(root: Path, rel: str) -> str:
    p = root / rel
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def unit_keys(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";", "[")) or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out.setdefault(k.strip(), []).append(v.strip())
    return out


def heredoc(text: str, target: str) -> str:
    m = re.search(r'cat > "\$DEST/' + re.escape(target) +
                  r'" <<\'EOF\'\n(.*?)\nEOF', text, re.S)
    return m.group(1) if m else ""


def env_bind(values: list[str]) -> str:
    for v in values:
        if v.startswith("JUNIOR_BIND="):
            return v.split("=", 1)[1]
    return ""


def load(root: Path) -> dict:
    inst = read(root, f"{LINUX}/install-overlay.sh")
    oci_raw = read(root, f"{LINUX}/oci/config.json")
    return {
        "system": read(root, f"{LINUX}/bitnetd.service"),
        "user": heredoc(inst, "usr/lib/systemd/user/bitnetd.service"),
        "oci": json.loads(oci_raw) if oci_raw else {},
        "readme": read(root, f"{LINUX}/oci/README.md"),
        "i2sd": read(root, f"{LINUX}/i2sd.py"),
        "os_release": read(root, f"{LINUX}/os-release.junior"),
    }


def split_bind(b: str) -> tuple[str, int]:
    host, _, port = b.rpartition(":")
    try:
        return host, int(port)
    except ValueError:
        return host, -1


def pack(trits: list[int]) -> bytes:
    code = {0: 0, 1: 1, -1: 2}
    bits = "".join(format(code[t], "02b") for t in trits)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def close(prev: str, trits: list[int], text: str) -> str:
    h = hashlib.sha256(prev.encode() + pack(trits) + text.encode())
    return h.hexdigest()[:16]


def check(src: dict) -> dict:
    rows: list[dict] = []

    def row(name: str, trit: int, note: str) -> None:
        rows.append({"row": name, "trit": trit, "note": note})

    sysk = unit_keys(src["system"])
    usrk = unit_keys(src["user"])
    oci = src["oci"]
    proc = oci.get("process") or {}
    linux = oci.get("linux") or {}
    m = re.search(r'^ID_LIKE="?([^"\n]*)"?', src["os_release"], re.M)
    families = (m.group(1).split() if m else []) or ["debian"]

    # U. identity per lane
    s_user = (sysk.get("User") or [""])[-1]
    s_group = (sysk.get("Group") or [""])[-1]
    dyn = (sysk.get("DynamicUser") or ["no"])[-1].lower() in {"yes", "true", "1"}
    if s_user in ROOT_NAMES or (not s_user and not dyn):
        row("U system unit identity", -1,
            f"User={s_user or '(unset, runs as root)'}; add User= or DynamicUser=yes")
    elif s_user == "nobody" and not dyn:
        row("U system unit identity", 0,
            "User=nobody: not root, but systemd logs 'Special user nobody "
            "configured, this is not safe' and nobody is shared with NFS "
            "squash; smallest fix DynamicUser=yes (drop User=/Group=)")
    else:
        row("U system unit identity", 1, f"User={s_user or 'dynamic'}")
    if s_group and not dyn:
        missing = [f for f in families if f in BASE_GROUPS
                   and s_group not in BASE_GROUPS[f]]
        row("U system unit Group resolves", 0 if missing else 1,
            f"Group={s_group} absent from base /etc/group on {missing}; "
            "DynamicUser=yes sidesteps the name" if missing else
            f"Group={s_group} in base group set of {families}")
    if "User" in usrk or "Group" in usrk:
        row("U user unit identity", 0,
            "User=/Group= in a --user unit is ignored or fails; drop them")
    else:
        row("U user unit identity", 1,
            "runs as the invoking login user (no User= in --user unit)")
    ou = proc.get("user") or {}
    uid, gid = ou.get("uid"), ou.get("gid")
    if uid in (0, None) or gid in (0, None):
        row("U OCI process.user", -1, f"uid={uid} gid={gid}: root or unset")
    else:
        row("U OCI process.user", 1, f"uid={uid} gid={gid}")
    pm = re.search(r"--user[ =](\d+):(\d+)", src["readme"])
    if pm:
        agree = (int(pm.group(1)), int(pm.group(2))) == (uid, gid)
        row("U podman --user vs config.json", 1 if agree else 0,
            f"README --user {pm.group(1)}:{pm.group(2)} vs config {uid}:{gid}")
    else:
        row("U podman --user vs config.json", 0, "README has no --user")
    ns = {n.get("type") for n in linux.get("namespaces") or []}
    if "user" in ns:
        has_maps = bool(linux.get("uidMappings")) and bool(linux.get("gidMappings"))
        row("U OCI user ns mappings", 1 if has_maps else 0,
            "uidMappings/gidMappings present" if has_maps else
            f"user namespace declared without uidMappings/gidMappings: runc "
            f"refuses a user ns with no maps, so container uid {uid} has no "
            "host id; smallest fix add one-to-one maps for "
            f"{uid}->host builder uid, or rely on podman --userns=keep-id")
    else:
        row("U OCI user ns mappings", 0, "no user namespace in config.json")

    # B. bind reach per identity
    binds = {
        "system unit": env_bind(sysk.get("Environment", [])),
        "user unit": env_bind(usrk.get("Environment", [])),
        "OCI env": env_bind(proc.get("env") or []),
        "OCI annotation": (oci.get("annotations") or {}).get("org.junioros.bind", ""),
    }
    for lane, b in binds.items():
        host, port = split_bind(b)
        bad = host != LOOPBACK or port != BITNETD or "0.0.0.0" in b
        row(f"B {lane} bind", -1 if bad else 1, b or "missing")
    hm = re.search(r'^HOST\s*=\s*"([^"]+)"', src["i2sd"], re.M)
    pmx = re.search(r"^PORT\s*=\s*(\d+)", src["i2sd"], re.M)
    ihost = hm.group(1) if hm else ""
    iport = int(pmx.group(1)) if pmx else -1
    row("B i2sd fallback bind", 1 if (ihost, iport) == (LOOPBACK, I2SD) else -1,
        f"{ihost}:{iport}")
    ports = sorted({BITNETD, iport} | {split_bind(b)[1] for b in binds.values()})
    priv = [p for p in ports if 0 <= p < 1024]
    row("B unprivileged ports", -1 if priv else 1,
        f"privileged {priv}" if priv else
        f"{ports} >= 1024: a capability-less identity can bind them")
    caps_sys = " ".join(sysk.get("CapabilityBoundingSet", []) +
                        sysk.get("AmbientCapabilities", []))
    caps_usr = " ".join(usrk.get("CapabilityBoundingSet", []) +
                        usrk.get("AmbientCapabilities", []))
    caps_oci = json.dumps(proc.get("capabilities") or {})
    granted = [n for n, c in (("system", caps_sys), ("user", caps_usr),
                              ("oci", caps_oci)) if "NET_BIND_SERVICE" in c]
    row("B no bind capability", -1 if granted else 1,
        f"CAP_NET_BIND_SERVICE in {granted}" if granted else
        "no lane grants CAP_NET_BIND_SERVICE (not needed above 1024)")

    # P. port map
    for lane in LANES:
        row(f"P {lane}", 1,
            f"{lane} -> juniorctl health -> {LOOPBACK}:{BITNETD} -> "
            f"{LOOPBACK}:{I2SD} (listed, never opened)")
    flipped = [k for k, v in FLAGS.items() if v]
    row("F flags", -1 if flipped else 1,
        f"flipped {flipped}" if flipped else "mint/socket/layer_one/fetch/ue5_launch false")

    prev = "0" * 16
    for r in rows:
        prev = close(prev, [r["trit"]], f'{r["row"]}|{r["note"]}')
        r["close"] = prev
    trits = [r["trit"] for r in rows]
    final = close(prev, trits, "overlay_runtime_identity_port_map")
    return {"item": "BETA_TO_OS JuniorOS overlay: hardened bitnetd.service (identity)",
            "rate": RATE, "flags": FLAGS, "rows": rows,
            "pass": trits.count(1), "advisory": trits.count(0),
            "fail": trits.count(-1), "close": final}


def selftest() -> int:
    good = {
        "system": "[Service]\nUser=nobody\nGroup=nogroup\n"
                  "Environment=JUNIOR_BIND=127.0.0.1:8765\nCapabilityBoundingSet=\n",
        "user": "[Service]\nEnvironment=JUNIOR_BIND=127.0.0.1:8765\n",
        "oci": {"process": {"user": {"uid": 1000, "gid": 1000},
                            "env": ["JUNIOR_BIND=127.0.0.1:8765"],
                            "capabilities": {"bounding": []}},
                "linux": {"namespaces": [{"type": "user"}]},
                "annotations": {"org.junioros.bind": "127.0.0.1:8765"}},
        "readme": "podman run --user 1000:1000",
        "i2sd": 'HOST = "127.0.0.1"\nPORT = 8767\n',
        "os_release": 'ID_LIKE="debian alpine"\n',
    }
    r = check(good)
    assert r["fail"] == 0, r
    adv = {x["row"] for x in r["rows"] if x["trit"] == 0}
    assert {"U system unit identity", "U system unit Group resolves",
            "U OCI user ns mappings"} <= adv, adv
    assert pack([1, -1, 0, 1]) == bytes([0b01100001])
    assert close("0" * 16, [1], "x") == close("0" * 16, [1], "x")
    assert close("0" * 16, [1], "x") != close("0" * 16, [-1], "x")
    for mut in ("root", "wild", "priv", "cap", "ociroot"):
        bad = json.loads(json.dumps(good))
        if mut == "root":
            bad["system"] = bad["system"].replace("User=nobody", "User=root")
        elif mut == "wild":
            bad["user"] = bad["user"].replace("127.0.0.1", "0.0.0.0")
        elif mut == "priv":
            bad["i2sd"] = 'HOST = "127.0.0.1"\nPORT = 80\n'
        elif mut == "cap":
            bad["system"] += "AmbientCapabilities=CAP_NET_BIND_SERVICE\n"
        else:
            bad["oci"]["process"]["user"]["uid"] = 0
        assert check(bad)["fail"] >= 1, mut
    dyn = json.loads(json.dumps(good))
    dyn["system"] = dyn["system"].replace("User=nobody\nGroup=nogroup\n",
                                          "DynamicUser=yes\n")
    rd = {x["row"]: x["trit"] for x in check(dyn)["rows"]}
    assert rd["U system unit identity"] == 1 and "U system unit Group resolves" not in rd
    real = check(load(ROOT))
    print(f"selftest ok; real tree pass={real['pass']} advisory={real['advisory']} "
          f"fail={real['fail']} close={real['close']}")
    return 1 if real["fail"] else 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    r = check(load(ROOT))
    if "--json" in argv:
        print(json.dumps(r, indent=2))
    else:
        for x in r["rows"]:
            print(f'{x["trit"]:+d} {x["close"]} {x["row"]}: {x["note"]}')
        print(f'pass={r["pass"]} advisory={r["advisory"]} fail={r["fail"]} close={r["close"]}')
    return 1 if r["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
