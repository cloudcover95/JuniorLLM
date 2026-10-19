"""juniorctl security gap key: parse-vs-substring, seccomp, OCI flags, claim checks (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 2 (`juniorctl security`)
against rails/linux/CONTAINER_SECURITY.md; docs/COMPILED_BACKLOG.md "container
security". Quant Chief pair A, slot 2026-10-19 02:05 MT: juniorctl security gaps
only. Trit Pack & I2_S side: the security-gap trit pack and its I2_S key. Winsor &
Quant Error covers the quant-error bound side of the same item in a different file.

Gaps in `juniorctl security` that this key covers (new capability; it does not
re-derive the health/ask/night contract, the exit-code trits, the
test_linux_security summary or the step 2/4/5/6 files):
  * substring vs systemd: security() passes a HARDENING key when its text appears
    anywhere in bitnetd.service, including a comment, another section, or a line
    that a later assignment overrides. Here the unit is parsed (sections,
    comments, last assignment wins; an empty assignment resets list keys) and
    each key gets an effective-value trit, plus `unit.substring_agrees`: -1 when
    the substring check would pass a key that systemd would not apply.
  * seccomp contents: security() only checks the profile file exists. Trits:
    default action denies; each dangerous syscall (ptrace, mount, bpf, kexec,
    module load, unshare/setns, keyring, perf, open_by_handle_at, ...) is not
    unconditionally allowed; clone allowed with no argument filter is 0 (it can
    create namespaces); an aarch64 arch entry exists (0 if missing: the Asahi
    host); duplicate names are 0.
  * OCI runtime flags: CONTAINER_SECURITY.md lists required docker/podman flags
    but nothing checks a run line. A captured `docker run` / `podman run` line is
    split with shlex and scored: --read-only, --cap-drop=ALL, no-new-privileges,
    seccomp=seccomp-bitnetd.json, --memory, --pids-limit, non-root --user,
    --network=none (pasta/slirp4netns only with 127.0.0.1 publishes); --privileged,
    a docker.sock / podman.sock mount, --cap-add of NET_RAW / SYS_ADMIN /
    SYS_PTRACE / SYS_MODULE, a wildcard publish, and :latest are -1; a tag
    without a digest is 0, @sha256: is +1.
  * hardcoded claims: security() returns privileged false, docker_socket false and
    bind 127.0.0.1:8765 as constants. A captured `juniorctl security` JSON is
    checked against the evidence above: claim matches evidence +1, contradicts
    -1, no evidence 0 (unverified). unit_ok is checked against the parsed unit.
  * sysctl-junior.conf: kptr_restrict 2, dmesg_restrict 1, yama.ptrace_scope >= 1
    (security() never reads it).
It never runs juniorctl, docker, podman, systemctl, a subprocess, a socket, or
the network; it reads only local files you point it at (or the built-in
selftest fixtures). It does not wrap, test, or re-derive any other team file.

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  JTR1 codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected)
  gamma = mean |trit| (share of trits that were actually decided)
I2_S block: code = trit + 1 (-1 -> 0, 0 -> 1, +1 -> 2), four per byte MSB first,
  padded with code 1 (zero) to a 128-trit block, as BitNet I2_S stores ternary.

Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16), pack = full JTR1 bytes, text = canonical JSON of the contract.
Rate 0.05: a green verdict needs zero -1 trits and at most 5% undecided (0)
trits; otherwise the verdict is "hold".
Compatible engines: JuniorLLM, BitNet-mlx, i2sd userspace, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage:
  python3 scripts/juniorctl_security_gap_i2s_key.py --selftest
  python3 scripts/juniorctl_security_gap_i2s_key.py [--unit P] [--seccomp P]
      [--oci P] [--claim P] [--sysctl P] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import struct
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
BITNETD = f"{BIND}:8765"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("JuniorLLM", "BitNet-mlx", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md Live beta step 2 `juniorctl security` gaps vs "
    "rails/linux/CONTAINER_SECURITY.md (COMPILED_BACKLOG container security)"
)
# Same keys rails/linux/juniorctl.py HARDENING checks by substring.
HARDENING = (
    ("NoNewPrivileges", "yes"),
    ("ProtectSystem", "strict"),
    ("MemoryDenyWriteExecute", "yes"),
    ("CapabilityBoundingSet", ""),
    ("Environment:JUNIOR_BIND", BITNETD),
)
DANGEROUS = (
    "ptrace", "mount", "umount2", "pivot_root", "chroot", "bpf", "kexec_load",
    "kexec_file_load", "init_module", "finit_module", "delete_module", "unshare",
    "setns", "keyctl", "add_key", "request_key", "perf_event_open",
    "open_by_handle_at", "process_vm_readv", "process_vm_writev", "userfaultfd",
    "reboot", "swapon", "swapoff",
)
DENY_ACTIONS = ("SCMP_ACT_ERRNO", "SCMP_ACT_KILL", "SCMP_ACT_KILL_PROCESS",
                "SCMP_ACT_KILL_THREAD", "SCMP_ACT_TRAP")
BAD_CAPS = ("NET_RAW", "SYS_ADMIN", "SYS_PTRACE", "SYS_MODULE")
WILDCARDS = (".".join(["0"] * 4), "::", "[::]", "*")  # deny-check only; never bound
SYSCTL_MIN = (("kernel.kptr_restrict", 2), ("kernel.dmesg_restrict", 1),
              ("kernel.yama.ptrace_scope", 1))
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}


# ---------------------------------------------------------------- unit
def parse_unit(text):
    """systemd-style parse: {section: {key: value}}, last assignment wins.

    Environment= is split into Environment:NAME entries. An empty assignment is
    kept as "" (for CapabilityBoundingSet= that is the hardened value)."""
    out = {}
    sec = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1]
            out.setdefault(sec, {})
            continue
        if sec is None or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "Environment":
            for part in shlex.split(v) if v else []:
                if "=" in part:
                    n, val = part.split("=", 1)
                    out[sec][f"Environment:{n}"] = val
            continue
        out[sec][k] = v
    return out


def unit_trits(text):
    if text is None:
        return [(f"unit.{k}", 0) for k, _ in HARDENING] + [("unit.substring_agrees", 0)]
    svc = parse_unit(text).get("Service", {})
    rows = []
    false_pass = False
    for k, want in HARDENING:
        ok = k in svc and svc[k] == want
        rows.append((f"unit.{k}", 1 if ok else -1))
        needle = (f"JUNIOR_BIND={want}" if k.startswith("Environment:") else f"{k}={want}")
        if needle in text and not ok:
            false_pass = True
    rows.append(("unit.substring_agrees", -1 if false_pass else 1))
    return rows, svc


# ---------------------------------------------------------------- seccomp
def seccomp_trits(text):
    if text is None:
        return [("seccomp.capture", 0)]
    try:
        prof = json.loads(text)
    except ValueError:
        return [("seccomp.json", -1)]
    rows = [("seccomp.default_deny", 1 if prof.get("defaultAction") in DENY_ACTIONS else -1)]
    allowed = {}
    names = []
    for rule in prof.get("syscalls", []):
        if rule.get("action") != "SCMP_ACT_ALLOW":
            continue
        filtered = bool(rule.get("args"))
        for n in rule.get("names", []):
            names.append(n)
            allowed[n] = allowed.get(n, False) or not filtered
    for n in DANGEROUS:
        rows.append((f"seccomp.deny:{n}", -1 if allowed.get(n) else 1))
    rows.append(("seccomp.clone_args", 0 if allowed.get("clone") else 1))
    arches = set()
    for a in prof.get("archMap", []):
        arches.add(a.get("architecture"))
        arches.update(a.get("subArchitectures", []))
    arches.update(prof.get("architectures", []))
    rows.append(("seccomp.arch_aarch64", 1 if "SCMP_ARCH_AARCH64" in arches else 0))
    rows.append(("seccomp.no_dup", 1 if len(names) == len(set(names)) else 0))
    return rows


# ---------------------------------------------------------------- OCI
def _opts(argv):
    """Return [(flag, value)] for long/short options after `run`, and the image."""
    try:
        i = argv.index("run") + 1
    except ValueError:
        return [], None
    takes = {"--security-opt", "--memory", "-m", "--pids-limit", "--user", "-u",
             "--network", "--net", "-v", "--volume", "--mount", "--cap-add",
             "--cap-drop", "-p", "--publish", "--name", "-e", "--env"}
    out = []
    image = None
    while i < len(argv):
        a = argv[i]
        if not a.startswith("-"):
            image = a
            break
        if "=" in a and a.startswith("--"):
            f, v = a.split("=", 1)
            out.append((f, v))
        elif a in takes and i + 1 < len(argv):
            out.append((a, argv[i + 1]))
            i += 1
        else:
            out.append((a, ""))
        i += 1
    return out, image


def _publish_host(spec):
    parts = spec.split(":")
    if len(parts) >= 3:
        return ":".join(parts[:-2]).strip("[]") or "*"
    return "*"  # "8765:8765" or "8765" publishes on every address


def oci_trits(text):
    if text is None:
        return [("oci.capture", 0)], None
    line = " ".join(l.rstrip("\\").strip() for l in text.splitlines() if l.strip())
    argv = shlex.split(line)
    opts, image = _opts(argv)
    if image is None:
        return [("oci.run_line", -1)], None
    get = lambda *fs: [v for f, v in opts if f in fs]  # noqa: E731
    secopts = get("--security-opt")
    vols = get("-v", "--volume", "--mount")
    caps_add = [c.upper().replace("CAP_", "") for v in get("--cap-add") for c in v.split(",")]
    caps_drop = [c.upper() for v in get("--cap-drop") for c in v.split(",")]
    user = (get("--user", "-u") or [""])[-1]
    net = (get("--network", "--net") or [""])[-1]
    pubs = [_publish_host(p) for p in get("-p", "--publish")]
    privileged = any(f == "--privileged" and v in ("", "true") for f, v in opts)
    sock = any(("docker.sock" in v or "podman.sock" in v) for v in vols)
    wild_pub = any(h in WILDCARDS for h in pubs)
    if net == "none":
        net_t = 1 if not pubs else -1
    elif net.split(":")[0] in ("pasta", "slirp4netns"):
        net_t = 1 if pubs and all(h == BIND for h in pubs) else -1
    else:
        net_t = -1
    if image.endswith(":latest") or (":" not in image.rsplit("/", 1)[-1] and "@" not in image):
        img_t = -1  # no tag means :latest
    elif "@sha256:" in image:
        img_t = 1
    else:
        img_t = 0
    rows = [
        ("oci.read_only", 1 if any(f == "--read-only" for f, _ in opts) else -1),
        ("oci.cap_drop_all", 1 if "ALL" in caps_drop else -1),
        ("oci.no_new_privs", 1 if any(s.startswith("no-new-privileges") for s in secopts) else -1),
        ("oci.seccomp", 1 if any(s[:8] in ("seccomp=", "seccomp:") and s.endswith("seccomp-bitnetd.json")
                                 for s in secopts) else -1),
        ("oci.memory", 1 if get("--memory", "-m") else -1),
        ("oci.pids_limit", 1 if get("--pids-limit") else -1),
        ("oci.user", 1 if user and user.split(":")[0] not in ("0", "root") else -1),
        ("oci.network", net_t),
        ("oci.not_privileged", -1 if privileged else 1),
        ("oci.no_engine_socket", -1 if sock else 1),
        ("oci.no_bad_cap_add", -1 if (set(caps_add) & set(BAD_CAPS) or "ALL" in caps_add) else 1),
        ("oci.no_wildcard_publish", -1 if wild_pub else 1),
        ("oci.image_pinned", img_t),
    ]
    evidence = {"privileged": privileged, "docker_socket": sock,
                "loopback_only": not wild_pub and net_t == 1}
    return rows, evidence


# ---------------------------------------------------------------- claims
def claim_trits(text, unit_rows, evidence):
    if text is None:
        return [("claim.capture", 0)]
    try:
        c = json.loads(text)
    except ValueError:
        return [("claim.json", -1)]
    rows = []
    unit_ok = None
    if unit_rows is not None:
        unit_ok = all(t == 1 for n, t in unit_rows if n != "unit.substring_agrees")
    for key, have in (("unit_ok", unit_ok),
                      ("privileged", None if evidence is None else evidence["privileged"]),
                      ("docker_socket", None if evidence is None else evidence["docker_socket"])):
        if key not in c:
            rows.append((f"claim.{key}", -1))
        elif have is None:
            rows.append((f"claim.{key}", 0))
        else:
            rows.append((f"claim.{key}", 1 if bool(c[key]) == have else -1))
    if c.get("bind") != BITNETD:
        rows.append(("claim.bind", -1))
    elif evidence is None:
        rows.append(("claim.bind", 0))
    else:
        rows.append(("claim.bind", 1 if evidence["loopback_only"] else -1))
    return rows


# ---------------------------------------------------------------- sysctl
def sysctl_trits(text):
    if text is None:
        return [(f"sysctl.{k}", 0) for k, _ in SYSCTL_MIN]
    vals = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;" or "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        vals[k] = v
    rows = []
    for k, lo in SYSCTL_MIN:
        v = vals.get(k)
        rows.append((f"sysctl.{k}", 1 if v is not None and v.isdigit() and int(v) >= lo else -1))
    return rows


# ---------------------------------------------------------------- packing
def pack_jtr1(trits):
    n = len(trits)
    gamma = (sum(abs(t) for t in trits) / n) if n else 0.0
    out = bytearray(MAGIC + struct.pack("<fI", gamma, n))
    for i in range(0, n, 4):
        b = 0
        chunk = trits[i:i + 4]
        for j in range(4):
            code = _JTR[chunk[j]] if j < len(chunk) else 0
            b |= code << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def unpack_jtr1(blob):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    trits = []
    for k in range(n):
        code = (blob[12 + k // 4] >> (6 - 2 * (k % 4))) & 3
        if code == 3:
            raise ValueError("code 0b11 rejected")
        trits.append(_JTR_DEC[code])
    return gamma, trits


def pack_i2s(trits):
    n = len(trits)
    padded = n + ((-n) % I2S_BLOCK) if n else I2S_BLOCK
    codes = [t + 1 for t in trits] + [1] * (padded - n)
    out = bytearray()
    for i in range(0, padded, 4):
        b = 0
        for j in range(4):
            b |= codes[i + j] << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def close_key(prev, pack, text):
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits):
    if not trits or any(t < 0 for t in trits):
        return "hold"
    undecided = sum(1 for t in trits if t == 0) / len(trits)
    return "green" if undecided <= RATE else "hold"


def build(unit, seccomp, oci, claim, sysctl, prev=GENESIS):
    if unit is None:
        u_rows = unit_trits(None)
        u_eval = None
    else:
        u_rows, _ = unit_trits(unit)
        u_eval = u_rows
    o_rows, evidence = oci_trits(oci)
    rows = (u_rows + seccomp_trits(seccomp) + o_rows
            + claim_trits(claim, u_eval, evidence) + sysctl_trits(sysctl))
    trits = [t for _, t in rows]
    pack = pack_jtr1(trits)
    i2s = pack_i2s(trits)
    contract = {
        "item": ROADMAP_ITEM,
        "rows": rows,
        "verdict": verdict(trits),
        "rate": RATE,
        "bitnetd": BITNETD,
        "i2s_sha": hashlib.sha256(i2s).hexdigest()[:16],
        "bind": BIND,
        "ports": list(PORTS),
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["key"] = close_key(prev, pack, text)
    contract["prev"] = prev
    contract["n"] = len(trits)
    contract["engines"] = list(ENGINES)
    return contract


# ---------------------------------------------------------------- selftest
_UNIT_OK = """[Unit]
Description=bitnetd
[Service]
User=nobody
Environment=JUNIOR_OFFLINE=1
Environment=JUNIOR_BIND=127.0.0.1:8765
NoNewPrivileges=yes
ProtectSystem=strict
MemoryDenyWriteExecute=yes
CapabilityBoundingSet=
"""
# Every HARDENING substring is present, but systemd would not apply them.
_UNIT_FAKE = """[Unit]
# NoNewPrivileges=yes
Description=ProtectSystem=strict
[Service]
Environment=JUNIOR_BIND=127.0.0.1:8765
MemoryDenyWriteExecute=yes
MemoryDenyWriteExecute=no
CapabilityBoundingSet=
CapabilityBoundingSet=CAP_SYS_ADMIN
"""
_SEC_OK = json.dumps({
    "defaultAction": "SCMP_ACT_ERRNO",
    "archMap": [{"architecture": "SCMP_ARCH_X86_64"}, {"architecture": "SCMP_ARCH_AARCH64"}],
    "syscalls": [{"names": ["read", "write", "futex"], "action": "SCMP_ACT_ALLOW"},
                 {"names": ["clone"], "action": "SCMP_ACT_ALLOW",
                  "args": [{"index": 0, "value": 2114060288, "op": "SCMP_CMP_MASKED_EQ"}]}],
})
_SEC_BAD = json.dumps({
    "defaultAction": "SCMP_ACT_ALLOW",
    "archMap": [{"architecture": "SCMP_ARCH_X86_64"}],
    "syscalls": [{"names": ["read", "read", "ptrace", "bpf", "clone"], "action": "SCMP_ACT_ALLOW"}],
})
_OCI_OK = ("podman run --read-only --cap-drop=ALL --security-opt=no-new-privileges \\\n"
           "  --security-opt seccomp=seccomp-bitnetd.json --memory=2g --pids-limit=256 \\\n"
           "  --user 1000:1000 --network=none ghcr.io/cloudcover95/bitnetd@sha256:" + "a" * 64 + "\n")
_OCI_BAD = ("docker run --privileged -v /var/run/docker.sock:/var/run/docker.sock "
            "--cap-add SYS_ADMIN -p 8765:8765 --user root bitnetd:latest\n")
_OCI_PASTA = ("podman run --read-only --cap-drop=ALL --security-opt=no-new-privileges "
              "--security-opt seccomp=seccomp-bitnetd.json -m 2g --pids-limit 256 -u 1000 "
              "--network=pasta -p 127.0.0.1:8765:8765 bitnetd:0.3\n")
_CLAIM = json.dumps({"unit_ok": True, "missing": [], "seccomp": True, "bind": BITNETD,
                     "privileged": False, "docker_socket": False,
                     "doc": "rails/linux/CONTAINER_SECURITY.md"})
_SYSCTL_OK = "kernel.kptr_restrict = 2\nkernel.dmesg_restrict = 1\nkernel.yama.ptrace_scope = 1\n"
_SYSCTL_BAD = "kernel.kptr_restrict = 1\n# kernel.dmesg_restrict = 1\nkernel.yama.ptrace_scope = 0\n"


def selftest():
    assert RATE == 0.05 and BIND == "127.0.0.1" and BITNETD == "127.0.0.1:8765"
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert set(PORTS) == {"Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"}
    # unit: parse vs substring
    ok_rows, svc = unit_trits(_UNIT_OK)
    assert [t for _, t in ok_rows] == [1, 1, 1, 1, 1, 1] and svc["CapabilityBoundingSet"] == ""
    fake_rows, _ = unit_trits(_UNIT_FAKE)
    assert [t for _, t in fake_rows] == [-1, -1, -1, -1, 1, -1]
    for k, want in HARDENING:  # the substring check juniorctl uses would pass it
        assert (f"JUNIOR_BIND={want}" if k.startswith("Env") else f"{k}={want}") in _UNIT_FAKE
    assert [t for _, t in unit_trits(None)] == [0] * 6
    # seccomp
    s_ok = seccomp_trits(_SEC_OK)
    assert all(t == 1 for _, t in s_ok) and len(s_ok) == 1 + len(DANGEROUS) + 3
    s_bad = dict(seccomp_trits(_SEC_BAD))
    assert s_bad["seccomp.default_deny"] == -1 and s_bad["seccomp.deny:ptrace"] == -1
    assert s_bad["seccomp.deny:bpf"] == -1 and s_bad["seccomp.deny:mount"] == 1
    assert s_bad["seccomp.clone_args"] == 0 and s_bad["seccomp.arch_aarch64"] == 0
    assert s_bad["seccomp.no_dup"] == 0
    assert seccomp_trits("{not json") == [("seccomp.json", -1)]
    # OCI
    o_ok, ev_ok = oci_trits(_OCI_OK)
    assert all(t == 1 for _, t in o_ok), o_ok
    assert ev_ok == {"privileged": False, "docker_socket": False, "loopback_only": True}
    o_bad, ev_bad = oci_trits(_OCI_BAD)
    d = dict(o_bad)
    assert d["oci.not_privileged"] == -1 and d["oci.no_engine_socket"] == -1
    assert d["oci.no_bad_cap_add"] == -1 and d["oci.no_wildcard_publish"] == -1
    assert d["oci.image_pinned"] == -1 and d["oci.user"] == -1 and d["oci.read_only"] == -1
    assert ev_bad["privileged"] and ev_bad["docker_socket"] and not ev_bad["loopback_only"]
    o_pasta, _ = oci_trits(_OCI_PASTA)
    dp = dict(o_pasta)
    assert dp["oci.network"] == 1 and dp["oci.image_pinned"] == 0
    assert all(t == 1 for n, t in o_pasta if n != "oci.image_pinned")
    assert oci_trits("echo hi\n") == ([("oci.run_line", -1)], None)
    assert oci_trits(None) == ([("oci.capture", 0)], None)
    # hardcoded claims vs evidence
    assert [t for _, t in claim_trits(_CLAIM, ok_rows, ev_ok)] == [1, 1, 1, 1]
    assert [t for _, t in claim_trits(_CLAIM, fake_rows, ev_bad)] == [-1, -1, -1, -1]
    assert [t for _, t in claim_trits(_CLAIM, None, None)] == [0, 0, 0, 0]
    assert claim_trits(None, None, None) == [("claim.capture", 0)]
    # sysctl
    assert [t for _, t in sysctl_trits(_SYSCTL_OK)] == [1, 1, 1]
    assert [t for _, t in sysctl_trits(_SYSCTL_BAD)] == [-1, -1, -1]
    # JTR1 round trip incl. partial byte
    for vec in ([1, 0, -1], [1, -1, 0, 1, -1], [], [0] * 9):
        g, back = unpack_jtr1(pack_jtr1(vec))
        assert back == vec
        if vec:
            assert abs(g - sum(abs(t) for t in vec) / len(vec)) < 1e-6
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 1.0, 1) + b"\xc0")
        raise AssertionError("0b11 accepted")
    except ValueError:
        pass
    # I2_S block
    blk = pack_i2s([-1, 0, 1, 1])
    assert len(blk) == I2S_BLOCK // 4 and blk[0] == 0b00011010 and blk[1] == 0b01010101
    # rate rule: 1/20 undecided is exactly 0.05 -> green; 2/20 -> hold
    assert verdict([1] * 19 + [0]) == "green"
    assert verdict([1] * 18 + [0, 0]) == "hold"
    assert verdict([1] * 19 + [-1]) == "hold"
    # close rule
    pk = pack_jtr1([1, 1])
    want = hashlib.sha256(GENESIS.encode() + pk + b"t").hexdigest()[:16]
    assert close_key(GENESIS, pk, "t") == want
    try:
        close_key("xyz", pk, "t")
        raise AssertionError("bad prev accepted")
    except ValueError:
        pass
    # full builds
    g1 = build(_UNIT_OK, _SEC_OK, _OCI_OK, _CLAIM, _SYSCTL_OK)
    g2 = build(_UNIT_OK, _SEC_OK, _OCI_OK, _CLAIM, _SYSCTL_OK)
    assert g1["verdict"] == "green" and g1["key"] == g2["key"]
    assert g1["n"] == 6 + (1 + len(DANGEROUS) + 3) + 13 + 4 + 3
    chained = build(_UNIT_OK, _SEC_OK, _OCI_OK, _CLAIM, _SYSCTL_OK, prev=g1["key"])
    assert chained["key"] != g1["key"] and chained["verdict"] == "green"
    red = build(_UNIT_FAKE, _SEC_BAD, _OCI_BAD, _CLAIM, _SYSCTL_BAD)
    assert red["verdict"] == "hold"
    no_oci = build(_UNIT_OK, _SEC_OK, None, _CLAIM, _SYSCTL_OK)
    assert no_oci["verdict"] == "hold"  # unverified claims are undecided
    empty = build(None, None, None, None, None)
    assert empty["verdict"] == "hold"
    return g1["key"]


def _read(path):
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--unit")
    ap.add_argument("--seccomp")
    ap.add_argument("--oci")
    ap.add_argument("--claim")
    ap.add_argument("--sysctl")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        key = selftest()
        print(json.dumps({"selftest": "pass", "key": key, "rate": RATE,
                          "ports": list(PORTS), "engines": list(ENGINES)}))
        return 0
    out = build(_read(a.unit), _read(a.seccomp), _read(a.oci), _read(a.claim),
                _read(a.sysctl), prev=a.prev)
    print(json.dumps(out, sort_keys=True))
    return 0 if out["verdict"] == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
