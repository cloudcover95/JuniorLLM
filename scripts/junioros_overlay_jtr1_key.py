"""JuniorOS overlay manifest JTR1 key. Stdlib only, offline, no socket.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay": Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH. I2_S 2B4T only
if GGUF already on disk. No kernel blob. (Quant Chief, pair A, slot
2026-10-17 14:05 MT.) This file is the I2_S / trit pack side of that item; the
quant-error bound side belongs to Winsor & Quant Error in a different path.

New capability: an offline overlay-manifest auditor. It reads text that is
already in hand (an os-release.junior file, a bitnetd.service unit, a PATH
string plus a list of overlay file paths) and scores every overlay requirement
as a trit, packs the trit vector as a JTR1 frame, and closes it into one I2_S
overlay key. It touches no network, runs no systemctl, imports no team module,
and does not wrap, test or re-derive any other team file (including
scripts/junioros_prod.py). If a GGUF path is given and is already local, only
its 8-byte header (magic + version) is read; nothing is ever downloaded.

Per-check trit:
  +1 requirement met (hardened / present / loopback)
   0 not stated (unit leaves a default) or optional piece absent (no GGUF)
  -1 explicitly weak or broken (wildcard bind, root user, docker.sock,
     NoNewPrivileges=no, kernel blob, juniorctl missing from PATH, bad GGUF)

JTR1 layout (not GGUF):
  magic  : 4 bytes b'JTR1'
  gamma  : 8 bytes big-endian IEEE-754 float64 (here: fraction of checks at +1)
  n      : 4 bytes big-endian uint32 (trit count)
  payload: ceil(2n/8) bytes, 2-bit I2_S codes MSB-first (0->0, 1->1, -1->2)

Close rule: sha256(prev || pack || text)[:16]; per check pack = the 1-trit I2_S
byte, text = canonical check JSON; the overlay key closes the full JTR1 frame
with the canonical report text. Rate 0.05 held (gamma budget step).
Compatible engines: BitNet-mlx, JuniorLLM, bitnetd 127.0.0.1:8765, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (bitnetd endpoint named, never dialed).
mint/socket/layer_one/fetch/ue5_launch false. No model pull, no GGUF download.

Usage:
  python3 junioros_overlay_jtr1_key.py --selftest
  python3 junioros_overlay_jtr1_key.py --os-release F --unit F \
      [--path-env "/usr/local/bin:/usr/bin"] [--files LISTFILE] [--gguf F] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
WILDCARD = ".".join(["0"] * 4)  # detected, never bound
BITNETD_PORT = 8765
BITNETD_ENDPOINT = "%s:%d" % (BIND, BITNETD_PORT)
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("BitNet-mlx", "JuniorLLM", "bitnetd 127.0.0.1:8765", "JuniorOS overlay")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay (Debian/Alpine + "
                "os-release.junior + hardened bitnetd.service + juniorctl on PATH)")
GENESIS = "0" * 16
MAGIC = b"JTR1"
BASES = ("debian", "alpine")

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def pack_i2s(trits) -> bytes:
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def unpack_i2s(data: bytes, n: int) -> list:
    if len(data) != (2 * n + 7) // 8:
        raise ValueError("payload length != ceil(2n/8)")
    out = []
    for i, b in enumerate(data):
        for s in (6, 4, 2, 0):
            idx = i * 4 + (6 - s) // 2
            code = (b >> s) & 3
            if idx >= n:
                if code:
                    raise ValueError("non-zero pad bits")
                continue
            if code == 3:
                raise ValueError("invalid I2_S code 3")
            out.append(_DEC[code])
    return out


def encode_jtr1(trits, gamma: float) -> bytes:
    trits = list(trits)
    return MAGIC + struct.pack(">dI", float(gamma), len(trits)) + pack_i2s(trits)


def decode_jtr1(frame: bytes):
    if frame[:4] != MAGIC:
        raise ValueError("bad JTR1 magic")
    gamma, n = struct.unpack(">dI", frame[4:16])
    return gamma, unpack_i2s(frame[16:], n)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def rate_step(x: float) -> float:
    return x * (1.0 + RATE)


# ---------------------------------------------------------------- parsers
def parse_os_release(text: str) -> dict:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def parse_unit(text: str) -> dict:
    """systemd unit -> {key: [values]} (last section wins; all values kept)."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;[" or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out.setdefault(k.strip(), []).append(v.strip())
    return out


def _last(unit: dict, key: str):
    vals = unit.get(key)
    return vals[-1] if vals else None


def _yes(v) -> bool:
    return v is not None and v.lower() in ("yes", "true", "1", "on")


def _no(v) -> bool:
    return v is not None and v.lower() in ("no", "false", "0", "off")


# ---------------------------------------------------------------- checks
def check_os_release(osr: dict) -> list:
    res = []
    ident = osr.get("ID", "").lower()
    like = set(osr.get("ID_LIKE", "").lower().split())
    res.append(("os_release.id_junior", 1 if ident == "junior" else -1, ident or "missing"))
    if like & set(BASES):
        res.append(("os_release.base", 1, " ".join(sorted(like & set(BASES)))))
    else:
        res.append(("os_release.base", -1, " ".join(sorted(like)) or "missing"))
    res.append(("os_release.version_id", 1 if osr.get("VERSION_ID") else 0,
                osr.get("VERSION_ID", "unset")))
    blob = any("kernel" in k.lower() and v for k, v in osr.items()
               if k.upper().startswith("JUNIOR_"))
    res.append(("os_release.no_kernel_blob", -1 if blob else 1,
                "kernel blob named" if blob else "overlay only"))
    return res


def check_unit(unit: dict) -> list:
    res = []
    execs = " ".join(unit.get("ExecStart", []))
    if not execs:
        res.append(("unit.exec_bind", -1, "no ExecStart"))
    elif WILDCARD in execs or "[::]" in execs:
        res.append(("unit.exec_bind", -1, "wildcard bind"))
    elif BITNETD_ENDPOINT in execs or ("127.0.0.1" in execs and str(BITNETD_PORT) in execs):
        res.append(("unit.exec_bind", 1, BITNETD_ENDPOINT))
    else:
        res.append(("unit.exec_bind", 0, "bind not stated"))
    whole = json.dumps(unit)
    res.append(("unit.no_docker_sock", -1 if "docker.sock" in whole else 1,
                "docker.sock" if "docker.sock" in whole else "clean"))
    user = _last(unit, "User")
    dyn = _last(unit, "DynamicUser")
    if user in ("root", "0") and not _yes(dyn):
        res.append(("unit.non_root", -1, "User=root"))
    elif _yes(dyn) or (user and user not in ("root", "0")):
        res.append(("unit.non_root", 1, "DynamicUser" if _yes(dyn) else "User=" + user))
    else:
        res.append(("unit.non_root", 0, "user not stated"))
    for key in ("NoNewPrivileges", "PrivateTmp", "PrivateDevices",
                "MemoryDenyWriteExecute", "LockPersonality", "RestrictRealtime"):
        v = _last(unit, key)
        res.append(("unit." + key, 1 if _yes(v) else (-1 if _no(v) else 0), v or "unset"))
    ps = _last(unit, "ProtectSystem")
    res.append(("unit.ProtectSystem", 1 if ps in ("strict", "full") else
                (-1 if _no(ps) else 0), ps or "unset"))
    ph = _last(unit, "ProtectHome")
    res.append(("unit.ProtectHome", 1 if (_yes(ph) or ph in ("read-only", "tmpfs")) else
                (-1 if _no(ph) else 0), ph or "unset"))
    caps = unit.get("CapabilityBoundingSet")
    if caps is None:
        res.append(("unit.cap_bound", 0, "unset"))
    else:
        last = caps[-1]
        res.append(("unit.cap_bound", 1 if last == "" else
                    (-1 if "CAP_SYS_ADMIN" in last or "CAP_NET_RAW" in last else 0),
                    last or "empty"))
    fam = _last(unit, "RestrictAddressFamilies")
    if fam is None:
        res.append(("unit.addr_families", 0, "unset"))
    else:
        bad = any(f in fam for f in ("AF_PACKET", "AF_NETLINK")) and not fam.startswith("~")
        res.append(("unit.addr_families", -1 if bad else 1, fam))
    deny = " ".join(unit.get("IPAddressDeny", []))
    allow = " ".join(unit.get("IPAddressAllow", []))
    if "any" in deny and allow and all(a in ("localhost", "127.0.0.1", "::1", "127.0.0.0/8")
                                       for a in allow.split()):
        res.append(("unit.ip_loopback_only", 1, "deny any, allow " + allow))
    elif WILDCARD in allow or "any" in allow:
        res.append(("unit.ip_loopback_only", -1, "allow " + allow))
    else:
        res.append(("unit.ip_loopback_only", 0, "unset"))
    return res


def check_juniorctl(path_env: str, files) -> list:
    dirs = [d.rstrip("/") or "/" for d in path_env.split(":") if d]
    hits = [d for d in dirs if (d + "/juniorctl") in files]
    return [("path.juniorctl", 1 if hits else -1, hits[0] + "/juniorctl" if hits else "not on PATH")]


def check_gguf(path) -> list:
    """Header read only, only if already local. Never downloads."""
    if not path:
        return [("gguf.i2s_2b4t_local", 0, "no GGUF given")]
    if not os.path.isfile(path):
        return [("gguf.i2s_2b4t_local", 0, "not on disk; not fetched")]
    with open(path, "rb") as fh:
        head = fh.read(8)
    if len(head) == 8 and head[:4] == b"GGUF":
        ver = struct.unpack("<I", head[4:8])[0]
        return [("gguf.i2s_2b4t_local", 1, "GGUF v%d header" % ver)]
    return [("gguf.i2s_2b4t_local", -1, "bad GGUF magic")]


# ---------------------------------------------------------------- key
def overlay_key(checks, prev: str = GENESIS) -> dict:
    chain = prev
    for name, trit, why in checks:
        text = json.dumps({"check": name, "trit": trit, "why": why},
                          sort_keys=True, separators=(",", ":"))
        chain = close_rule(chain, pack_i2s([trit]), text)
    trits = [t for _, t, _ in checks]
    plus = sum(1 for t in trits if t == 1)
    gamma = plus / len(trits) if trits else 0.0
    frame = encode_jtr1(trits, gamma)
    verdict = "ready" if all(t >= 0 for t in trits) else "blocked"
    report = {"item": ROADMAP_ITEM, "n": len(trits), "plus": plus,
              "minus": sum(1 for t in trits if t == -1), "gamma": round(gamma, 6),
              "budget_next": round(rate_step(gamma), 6), "rate": RATE,
              "verdict": verdict, "bind": BIND, "endpoint": BITNETD_ENDPOINT,
              "ports": list(PORTS), "engines": list(ENGINES)}
    text = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["chain"] = chain
    report["key"] = close_rule(chain, frame, text)
    report["frame_hex"] = frame.hex()
    return report


def audit(os_release_text, unit_text, path_env, files, gguf=None, prev=GENESIS):
    checks = (check_os_release(parse_os_release(os_release_text))
              + check_unit(parse_unit(unit_text))
              + check_juniorctl(path_env, set(files))
              + check_gguf(gguf))
    return checks, overlay_key(checks, prev)


# ---------------------------------------------------------------- selftest
_GOOD_OSR = """NAME="JuniorOS"
ID=junior
ID_LIKE="debian"
VERSION_ID=0.1-beta
PRETTY_NAME="JuniorOS overlay 0.1 beta"
"""

_GOOD_UNIT = """[Unit]
Description=BitNet daemon (JuniorOS overlay)
[Service]
ExecStart=/usr/local/bin/bitnetd --host 127.0.0.1 --port 8765
DynamicUser=yes
NoNewPrivileges=yes
PrivateTmp=yes
PrivateDevices=yes
MemoryDenyWriteExecute=yes
LockPersonality=yes
RestrictRealtime=yes
ProtectSystem=strict
ProtectHome=yes
CapabilityBoundingSet=
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
IPAddressDeny=any
IPAddressAllow=localhost
[Install]
WantedBy=multi-user.target
"""

_BAD_UNIT = """[Service]
ExecStart=/usr/local/bin/bitnetd --host %s --port 8765
User=root
NoNewPrivileges=no
ReadWritePaths=/var/run/docker.sock
CapabilityBoundingSet=CAP_SYS_ADMIN
RestrictAddressFamilies=AF_UNIX AF_PACKET
""" % WILDCARD


def selftest() -> int:
    assert RATE == 0.05 and BIND == "127.0.0.1"
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    # I2_S round trip incl. pad
    for v in ([], [1], [-1, 0, 1], [1, -1, 0, 0, -1]):
        assert unpack_i2s(pack_i2s(v), len(v)) == v
    try:
        unpack_i2s(b"\xc0", 1)
        raise AssertionError("code 3 accepted")
    except ValueError:
        pass
    # close rule exact
    exp = hashlib.sha256(GENESIS.encode() + b"\x40" + b"t").hexdigest()[:16]
    assert close_rule(GENESIS, b"\x40", "t") == exp
    files = ["/usr/local/bin/juniorctl", "/etc/os-release.junior"]
    pathenv = "/usr/local/sbin:/usr/local/bin:/usr/bin"
    checks, good = audit(_GOOD_OSR, _GOOD_UNIT, pathenv, files)
    assert good["verdict"] == "ready", good
    assert good["minus"] == 0 and good["n"] == len(checks) == 20
    gamma, trits = decode_jtr1(bytes.fromhex(good["frame_hex"]))
    assert trits == [t for _, t, _ in checks] and abs(gamma - good["gamma"]) < 1e-12
    assert abs(good["budget_next"] - round(good["gamma"] * 1.05, 6)) < 1e-9
    # key deterministic and prev-sensitive
    assert audit(_GOOD_OSR, _GOOD_UNIT, pathenv, files)[1]["key"] == good["key"]
    assert audit(_GOOD_OSR, _GOOD_UNIT, pathenv, files, prev="f" * 16)[1]["key"] != good["key"]
    # bad overlay blocks
    bad_osr = "ID=ubuntu\nJUNIOR_KERNEL=vmlinuz-junior\n"
    bchecks, bad = audit(bad_osr, _BAD_UNIT, "/usr/bin", files)
    assert bad["verdict"] == "blocked"
    bmap = {n: t for n, t, _ in bchecks}
    for k in ("os_release.id_junior", "os_release.base", "os_release.no_kernel_blob",
              "unit.exec_bind", "unit.no_docker_sock", "unit.non_root",
              "unit.NoNewPrivileges", "unit.cap_bound", "unit.addr_families",
              "path.juniorctl"):
        assert bmap[k] == -1, (k, bmap[k])
    assert bad["key"] != good["key"]
    # GGUF: absent path is 0 and never fetched; local header read only
    assert check_gguf("/nonexistent/ggml-model-i2_s.gguf")[0][1] == 0
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh:
        fh.write(b"GGUF" + struct.pack("<I", 3) + b"\x00" * 8)
        tmp = fh.name
    try:
        assert check_gguf(tmp)[0][1] == 1
    finally:
        os.unlink(tmp)
    print(json.dumps({"selftest": "pass", "key": good["key"], "bad_key": bad["key"],
                      "n": good["n"], "gamma": good["gamma"], "rate": RATE,
                      "ports": list(PORTS), "engines": list(ENGINES)}, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--os-release")
    ap.add_argument("--unit")
    ap.add_argument("--path-env", default="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin")
    ap.add_argument("--files", help="text file listing overlay paths, one per line")
    ap.add_argument("--gguf")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.os_release and a.unit):
        ap.error("--os-release and --unit are required (or --selftest)")
    with open(a.os_release) as fh:
        osr = fh.read()
    with open(a.unit) as fh:
        unit = fh.read()
    files = []
    if a.files:
        with open(a.files) as fh:
            files = [ln.strip() for ln in fh if ln.strip()]
    checks, rep = audit(osr, unit, a.path_env, files, a.gguf, a.prev)
    rep["checks"] = [{"check": n, "trit": t, "why": w} for n, t, w in checks]
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["verdict"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
