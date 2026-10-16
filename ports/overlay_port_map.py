"""JuniorOS overlay port map: in-process handshake check (no sockets).

Advances JuniorLLM docs/BETA_TO_OS.md, item "JuniorOS overlay":
Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH.

Reads the overlay sources under rails/linux/ as plain text and checks, in-process:
  * os-release.junior names JuniorOS, ID_LIKE debian/alpine, overlay variant,
    and pins JUNIOR_BITNETD to 127.0.0.1:8765 with JUNIOR_OFFLINE=1;
  * bitnetd.service binds 127.0.0.1:8765 and carries the hardening directives;
  * install-overlay.sh pins juniorctl to /usr/local/bin on PATH and refuses
    non-loopback binds;
  * the Home / OSai / FieldCore / Flagstaff / JuniorLLM port map routes only to
    loopback bitnetd (127.0.0.1:8765) and i2sd (127.0.0.1:8767), never 0.0.0.0.

No socket is opened, nothing is fetched, no model is pulled, nothing is launched.
Each check row is closed with sha256(prev || pack || text)[:16].
Stdlib only. Run: python ports/overlay_port_map.py  (exit 0 on pass).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"

LOOPBACK = "127.0.0.1"
BITNETD = (LOOPBACK, 8765)
I2SD = (LOOPBACK, 8767)
RATE = 0.05
PACK = "overlay-port-map"

FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}

# Node -> loopback service it hands off to. In-process map only.
PORT_MAP = {
    "Home": {"bitnetd": BITNETD, "i2sd": I2SD},
    "OSai": {"bitnetd": BITNETD, "i2sd": I2SD},
    "FieldCore": {"bitnetd": BITNETD},
    "Flagstaff": {"bitnetd": BITNETD},
    "JuniorLLM": {"bitnetd": BITNETD, "i2sd": I2SD},
}

HARDENING = (
    "NoNewPrivileges=yes",
    "ProtectSystem=strict",
    "ProtectHome=yes",
    "PrivateTmp=yes",
    "PrivateDevices=yes",
    "MemoryDenyWriteExecute=yes",
    "RestrictNamespaces=yes",
    "CapabilityBoundingSet=",
    "AmbientCapabilities=",
    "Environment=JUNIOR_OFFLINE=1",
)


def close(prev: str, pack: str, text: str) -> str:
    """Close rule: sha256(prev || pack || text)[:16]."""
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def _read(name: str) -> str:
    path = LINUX / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _kv(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def check_os_release(text: str) -> list:
    kv = _kv(text)
    like = kv.get("ID_LIKE", "").split()
    return [
        ("os_release.present", bool(text)),
        ("os_release.id", kv.get("ID") == "junioros"),
        ("os_release.id_like", "debian" in like and "alpine" in like),
        ("os_release.overlay", kv.get("VARIANT_ID") == "overlay"),
        ("os_release.bitnetd", kv.get("JUNIOR_BITNETD") == "%s:%d" % BITNETD),
        ("os_release.offline", kv.get("JUNIOR_OFFLINE") == "1"),
    ]


def check_unit(text: str) -> list:
    lines = {ln.strip() for ln in text.splitlines()}
    rows = [
        ("unit.present", bool(text)),
        ("unit.bind", "Environment=JUNIOR_BIND=%s:%d" % BITNETD in lines),
        ("unit.no_wildcard", "0.0.0.0" not in text),
        ("unit.no_docker_sock", "docker.sock" not in text),
    ]
    rows += [("unit." + d.split("=", 1)[0], d in lines) for d in HARDENING]
    return rows


def check_installer(text: str) -> list:
    return [
        ("install.present", bool(text)),
        ("install.juniorctl_on_path", '/usr/local/bin/juniorctl"' in text
         and 'JUNIOROS_BIN="/usr/local/bin"' in text),
        ("install.refuses_non_loopback", "refusing non-loopback" in text),
        ("install.default_bind", "127.0.0.1:8765" in text),
    ]


def check_port_map(port_map: dict) -> list:
    rows = []
    allowed = {BITNETD, I2SD}
    for node, svcs in port_map.items():
        ok = bool(svcs) and all(hp in allowed and hp[0] == LOOPBACK
                                for hp in svcs.values())
        rows.append(("map.%s" % node, ok))
    rows.append(("map.distinct_ports", BITNETD[1] != I2SD[1]))
    return rows


def run() -> dict:
    rows = (
        check_os_release(_read("os-release.junior"))
        + check_unit(_read("bitnetd.service"))
        + check_installer(_read("install-overlay.sh"))
        + check_port_map(PORT_MAP)
        + [("flags.all_false", not any(FLAGS.values())), ("rate", RATE == 0.05)]
    )
    prev = "0" * 16
    chain = []
    for name, ok in rows:
        text = "%s=%s" % (name, "pass" if ok else "fail")
        prev = close(prev, PACK, text)
        chain.append({"check": name, "ok": bool(ok), "close": prev})
    return {
        "item": "BETA_TO_OS: JuniorOS overlay",
        "pack": PACK,
        "rate": RATE,
        "flags": FLAGS,
        "ports": {"bitnetd": "%s:%d" % BITNETD, "i2sd": "%s:%d" % I2SD},
        "map": {n: {s: "%s:%d" % hp for s, hp in v.items()} for n, v in PORT_MAP.items()},
        "checks": chain,
        "tip": prev,
        "ok": all(r["ok"] for r in chain),
    }


def main() -> int:
    report = run()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
