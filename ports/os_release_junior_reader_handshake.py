"""os-release.junior reader/writer handshake + overlay confirm receipt (in-process, no sockets).

Advances JuniorLLM docs/BETA_TO_OS.md, item "JuniorOS overlay":
Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH.
Slot: the os-release.junior gaps plus an overlay confirm receipt (pair B,
Handshake & Ports side). PATH, Debian, MemoryMax and the user-unit hardening
parity are left to the files that already landed.

New angle (uncovered piece): earlier checks read os-release.junior's *fields*.
None asks whether anything at runtime ever *reads* the file the overlay
installs. This handshake pairs the writer with the readers, in-process:
  * writer: the `cp ... os-release.junior ...` targets in install-overlay.sh
    (etc/os-release.junior and usr/lib/os-release.junior, same source);
  * readers: every os-release path literal in the runtime rails/linux .py/.sh
    sources (backend._asahi reads /etc/os-release and /usr/lib/os-release);
  * reader_meets_writer: does any reader path equal an installed target?
    The freedesktop spec only reads /etc/os-release and /usr/lib/os-release,
    so a `.junior` file is invisible unless something opens it by name;
  * key consumers: for each JUNIOR_* key the file declares (JUNIOR_BITNETD,
    JUNIOR_OFFLINE, JUNIOR_CHARTER) is there a runtime reader of that key?
    JUNIOR_CHARTER must resolve to a file in the tree;
  * port identity: os-release names bitnetd 127.0.0.1:8765; the i2sd
    127.0.0.1:8767 fallback lane (bitnetd.sh execs i2sd.py) has no line;
  * the overlay confirm receipt: one JSON receipt with the writer targets,
    readers, key consumers, the Home / OSai / FieldCore / Flagstaff /
    JuniorLLM lanes (all routed juniorctl health -> bitnetd 127.0.0.1:8765
    -> i2sd 127.0.0.1:8767, loopback only) and the trit vector, closed with
    sha256(prev || pack || text)[:16].

Trits: 1 pass, 0 advisory (a real gap, reported, does not fail the run),
-1 fail (exit 1). Ports are listed, never opened. No socket, no subprocess,
no network, no fetch, no model pull, no UE5, no os.environ reads. Reads only
rails/linux text files and docs/BETA_TO_OS.md. Does not import, wrap or
re-check any team file under ports/ or scripts/, and never imports the rails.
Stdlib only. Rate 0.05 (committed beside the pack, never polled or set).
Close rule: sha256(prev || pack || text)[:16], trits packed as 2-bit i2s
(0->0, 1->1, -1->2), MSB first, zero-padded to a whole byte.

Run: python ports/os_release_junior_reader_handshake.py --selftest  (exit 0 on pass)
     python ports/os_release_junior_reader_handshake.py             (real tree, JSON)
     python ports/os_release_junior_reader_handshake.py --prev <16hex>
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"

ITEM = 'BETA_TO_OS "JuniorOS overlay": os-release.junior gaps + overlay confirm receipt'
LOOPBACK = "127.0.0.1"
BITNETD = (LOOPBACK, 8765)
I2SD = (LOOPBACK, 8767)
RATE = 0.05
GENESIS = "0" * 16
WILDCARD = ".".join(["0"] * 4)  # the any-address; only ever denied
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
LANES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
SPEC_PATHS = ("/etc/os-release", "/usr/lib/os-release")
KEY_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
CP_RE = re.compile(r'^\s*cp\s+"\$HERE/os-release\.junior"\s+"\$DEST/([^"]+)"\s*$')
PATH_LIT_RE = re.compile(r"""["'](/[A-Za-z0-9_./-]*os-release[A-Za-z0-9_.-]*)["']""")
RUNTIME_SKIP = re.compile(r"^(ctl_skillpin|ctl_cli_t)")


def kv(text: str) -> dict:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = KEY_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def writer_targets(installer: str) -> list:
    return sorted({"/" + m.group(1) for m in map(CP_RE.match, installer.splitlines()) if m})


def reader_paths(sources: dict) -> dict:
    """sources: {name: text}. Returns {name: sorted os-release path literals}."""
    found = {}
    for name, text in sorted(sources.items()):
        hits = sorted(set(PATH_LIT_RE.findall(text)))
        if hits:
            found[name] = hits
    return found


def key_readers(keys, sources: dict) -> dict:
    """A reader names the key without assigning it (KEY= is a declaration, e.g. Environment=)."""
    out = {}
    for key in keys:
        rx = re.compile(re.escape(key) + r"(?![=A-Z0-9_])")
        out[key] = sorted(n for n, t in sources.items() if rx.search(t))
    return out


def key_writers(keys, sources: dict) -> dict:
    return {k: sorted(n for n, t in sources.items() if (k + "=") in t) for k in keys}


def parse_port(value: str):
    host, _, port = value.rpartition(":")
    return (host, int(port)) if port.isdigit() else (value, -1)


def evaluate(os_release: str, installer: str, sources: dict, charter_exists: bool,
             i2sd_src: str, bitnetd_sh: str) -> list:
    """Return rows (name, trit, note). Pure: only the strings it is handed."""
    rows = []
    fields = kv(os_release)
    targets = writer_targets(installer)
    readers = reader_paths(sources)
    read_set = {p for hits in readers.values() for p in hits}

    def row(name, trit, note):
        rows.append((name, trit, note))

    row("os_release.parsed", 1 if fields.get("ID") == "junioros" else -1, "ID=junioros")
    row("os_release.no_wildcard", -1 if WILDCARD in os_release else 1, "no wildcard bind in identity")
    want = {"/etc/os-release.junior", "/usr/lib/os-release.junior"}
    row("writer.targets", 1 if want <= set(targets) else -1,
        "install-overlay.sh cp -> " + (", ".join(targets) or "none"))
    row("writer.single_source", 1 if len(targets) >= 2 else 0,
        "both targets copied from one $HERE/os-release.junior")
    row("readers.found", 1 if readers else 0,
        "; ".join(f"{n}: {', '.join(h)}" for n, h in readers.items()) or "no os-release reader")
    row("readers.spec_paths", 1 if read_set & set(SPEC_PATHS) else 0,
        "runtime reads the freedesktop paths " + ", ".join(sorted(read_set & set(SPEC_PATHS))))
    meet = sorted(read_set & set(targets))
    row("reader_meets_writer", 1 if meet else 0,
        ("reader opens " + ", ".join(meet)) if meet else
        "no runtime reader opens an installed os-release.junior; spec readers only see "
        "/etc/os-release and /usr/lib/os-release, which the overlay never writes, so the "
        "JuniorOS identity is invisible to backend probe and juniorctl health")

    jkeys = sorted(k for k in fields if k.startswith("JUNIOR_"))
    consumers = key_readers(jkeys, sources)
    setters = key_writers(jkeys, sources)
    for key in jkeys:
        who = consumers[key]
        if key == "JUNIOR_CHARTER":
            trit = 1 if charter_exists else -1
            note = f"{fields[key]} resolves in tree" if charter_exists else f"{fields[key]} missing"
        else:
            trit = 1 if who else 0
            set_by = (" (set as KEY= by " + ", ".join(setters[key]) + ")") if setters[key] else ""
            note = ("read by " + ", ".join(who)) if who else \
                "declared in os-release.junior, no runtime .py/.sh reader of this key" + set_by
        row(f"key.{key}", trit, note)

    claim = parse_port(fields.get("JUNIOR_BITNETD", ""))
    row("port.bitnetd_claim", 1 if claim == BITNETD else -1,
        f"JUNIOR_BITNETD={fields.get('JUNIOR_BITNETD', '')} want {BITNETD[0]}:{BITNETD[1]}")
    sh_ok = f"{LOOPBACK}:{BITNETD[1]}" in bitnetd_sh and "--host 127.0.0.1" in bitnetd_sh
    row("port.bitnetd_sh_agrees", 1 if sh_ok else -1, "bitnetd.sh default + exec bind 127.0.0.1:8765")
    i2sd_ok = f'HOST = "{I2SD[0]}"' in i2sd_src and f"PORT = {I2SD[1]}" in i2sd_src
    row("port.i2sd_loopback", 1 if i2sd_ok else -1, "i2sd.py HOST 127.0.0.1 PORT 8767")
    falls_back = "i2sd.py" in bitnetd_sh
    named = any(v.endswith(f":{I2SD[1]}") for v in fields.values())
    row("port.i2sd_identity", 1 if (named or not falls_back) else 0,
        "os-release.junior names the i2sd lane" if named else
        "bitnetd.sh falls back to i2sd.py on 127.0.0.1:8767 but os-release.junior has no i2sd line")
    row("guard.flags", 1 if not any(FLAGS.values()) else -1,
        "mint/socket/layer_one/fetch/ue5_launch all false")
    return rows


def pack(trits) -> bytes:
    code = {0: 0, 1: 1, -1: 2}
    bits = "".join(format(code[t], "02b") for t in trits)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)) if bits else b""


def close(prev: str, packed: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + packed + text.encode()).hexdigest()[:16]


def receipt(rows, targets, readers, prev: str = GENESIS) -> dict:
    trits = [t for _, t, _ in rows]
    body = {
        "item": ITEM,
        "writer_targets": targets,
        "readers": readers,
        "lanes": {lane: [f"{BITNETD[0]}:{BITNETD[1]}", f"{I2SD[0]}:{I2SD[1]}"] for lane in LANES},
        "handshake": ["juniorctl health", f"bitnetd {BITNETD[0]}:{BITNETD[1]}", f"i2sd {I2SD[0]}:{I2SD[1]}"],
        "rows": [{"name": n, "trit": t, "note": note} for n, t, note in rows],
        "rate": RATE,
        "flags": FLAGS,
        "bind": LOOPBACK,
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    packed = pack(trits)
    body["prev"] = prev
    body["pack"] = packed.hex()
    body["close"] = close(prev, packed, text)
    body["pass"] = trits.count(1)
    body["advisory"] = trits.count(0)
    body["fail"] = trits.count(-1)
    return body


def _runtime_sources() -> dict:
    out = {}
    for p in sorted(LINUX.iterdir()):
        if p.suffix in (".py", ".sh") and p.is_file() and not RUNTIME_SKIP.match(p.name):
            out[p.name] = p.read_text(encoding="utf-8", errors="ignore")
    return out


def live(prev: str = GENESIS) -> dict:
    os_release = (LINUX / "os-release.junior").read_text(encoding="utf-8")
    installer = (LINUX / "install-overlay.sh").read_text(encoding="utf-8")
    sources = _runtime_sources()
    charter = kv(os_release).get("JUNIOR_CHARTER", "")
    charter_ok = bool(charter) and (ROOT / charter).is_file()
    rows = evaluate(os_release, installer, sources, charter_ok,
                    sources.get("i2sd.py", ""), sources.get("bitnetd.sh", ""))
    return receipt(rows, writer_targets(installer), reader_paths(sources), prev)


GOOD_OSR = 'ID=junioros\nJUNIOR_BITNETD="127.0.0.1:8765"\nJUNIOR_OFFLINE="1"\nJUNIOR_CHARTER="docs/BETA_TO_OS.md"\n'
GOOD_INST = 'cp "$HERE/os-release.junior" "$DEST/etc/os-release.junior"\ncp "$HERE/os-release.junior" "$DEST/usr/lib/os-release.junior"\n'
GOOD_SH = 'BIND=${JUNIOR_BIND:-127.0.0.1:8765}\nexec x --host 127.0.0.1 --port 8765\nexec python3 "$HERE/i2sd.py"\n'
GOOD_I2SD = 'HOST = "127.0.0.1"\nPORT = 8767\n'


def selftest() -> int:
    ok = True

    def check(cond, what):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + what)
        ok = ok and bool(cond)

    check(pack([1, 0, -1, 1]) == bytes([0b01001001]), "pack vector [1,0,-1,1] -> 0x49")
    check(pack([1, 1, 1, 1, -1]) == bytes([0x55, 0x80]), "pack zero-pads to whole byte")
    c1 = close(GENESIS, b"\x49", "t")
    check(c1 == hashlib.sha256(b"0" * 16 + b"\x49" + b"t").hexdigest()[:16] and len(c1) == 16,
          "close = sha256(prev || pack || text)[:16]")
    check(writer_targets(GOOD_INST) == ["/etc/os-release.junior", "/usr/lib/os-release.junior"],
          "writer targets parsed from cp lines")

    srcs = {"backend.py": 'for p in ("/etc/os-release", "/usr/lib/os-release"):\n'}
    rows = dict((n, t) for n, t, _ in evaluate(GOOD_OSR, GOOD_INST, srcs, True, GOOD_I2SD, GOOD_SH))
    check(rows["reader_meets_writer"] == 0, "spec-only reader -> advisory, not fail")
    check(rows["key.JUNIOR_BITNETD"] == 0 and rows["key.JUNIOR_CHARTER"] == 1, "key consumers scored")
    kr = key_readers(["JUNIOR_OFFLINE"], {"u.sh": "Environment=JUNIOR_OFFLINE=1\n", "r.py": 'os.environ.get("JUNIOR_OFFLINE")\n'})
    check(kr["JUNIOR_OFFLINE"] == ["r.py"], "Environment=KEY= is a setter, not a reader")
    check(rows["port.i2sd_identity"] == 0, "i2sd fallback without identity line -> advisory")
    check(-1 not in rows.values(), "good fixture has no fail")

    srcs2 = dict(srcs, **{"probe.py": 'P = "/etc/os-release.junior"\nk = "JUNIOR_BITNETD"\n'})
    osr2 = GOOD_OSR + 'JUNIOR_I2SD="127.0.0.1:8767"\n'
    rows2 = dict((n, t) for n, t, _ in evaluate(osr2, GOOD_INST, srcs2, True, GOOD_I2SD, GOOD_SH))
    check(rows2["reader_meets_writer"] == 1 and rows2["key.JUNIOR_BITNETD"] == 1
          and rows2["port.i2sd_identity"] == 1, "closed gaps score 1")

    bad = 'ID=junioros\nJUNIOR_BITNETD="%s:8765"\nJUNIOR_CHARTER="docs/nope.md"\n' % WILDCARD
    rows3 = dict((n, t) for n, t, _ in evaluate(bad, "", {}, False, "", ""))
    check(rows3["os_release.no_wildcard"] == -1 and rows3["port.bitnetd_claim"] == -1
          and rows3["writer.targets"] == -1 and rows3["key.JUNIOR_CHARTER"] == -1,
          "wildcard bind, missing targets and missing charter fail closed")

    r1 = receipt(evaluate(GOOD_OSR, GOOD_INST, srcs, True, GOOD_I2SD, GOOD_SH), [], {})
    r2 = receipt(evaluate(GOOD_OSR, GOOD_INST, srcs, True, GOOD_I2SD, GOOD_SH), [], {})
    r3 = receipt(evaluate(GOOD_OSR, GOOD_INST, srcs, True, GOOD_I2SD, GOOD_SH), [], {}, prev=r1["close"])
    check(r1["close"] == r2["close"] and r3["close"] != r1["close"], "receipt deterministic, chains on prev")
    check(r1["rate"] == RATE and not any(r1["flags"].values()) and r1["bind"] == LOOPBACK,
          "rate 0.05, flags false, bind 127.0.0.1")

    rep = live()
    for r in rep["rows"]:
        tag = {1: "pass", 0: "advisory", -1: "FAIL"}[r["trit"]]
        print(f"  live {tag:8s} {r['name']}: {r['note']}")
    check(rep["fail"] == 0, f"live tree: {rep['pass']} pass, {rep['advisory']} advisory, {rep['fail']} fail")
    print(f"close {rep['close']}  pack {rep['pack']}  item {ITEM}")
    return 0 if ok else 1


def main(argv) -> int:
    if "--selftest" in argv:
        return selftest()
    prev = GENESIS
    if "--prev" in argv:
        i = argv.index("--prev")
        prev = argv[i + 1] if i + 1 < len(argv) else GENESIS
        if not re.fullmatch(r"[0-9a-f]{16}", prev):
            print("--prev must be 16 hex chars", file=sys.stderr)
            return 2
    rep = live(prev)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 1 if rep["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
