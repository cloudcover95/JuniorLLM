#!/usr/bin/env python3
"""Live beta i2sd fallback security handshake: 127.0.0.1:8767 vs the 8765 gate.

Advances JuniorLLM docs/BETA_TO_OS.md, item "Live beta", step 2
(`juniorctl security`) together with step 6 ("bitnetd on 127.0.0.1:8765
only"), from the i2sd side. Already landed and not redone here: dispatch and
ports (b3da3b5), exit codes (062bfc7), night (6257415), security exit trits
(fbfe7a1). Those name i2sd 8767 as "not gated"; this file answers the open
lead behind that: what actually answers when bitnetd.service falls back to
rails/linux/i2sd.py, and does any security gate see it?

How it checks, in-process with no sockets:
  * rails/linux/{i2sd.py, bitnetd.sh, bitnetd.service, juniorctl.py,
    install-overlay.sh, debian/stage_overlay.sh, iot.py} and
    rails/lint/bind_guard_lint.py are read as text and parsed with ast/regex.
  * The i2sd line protocol is driven in-process: Handler.handle() is called on
    an instance built with object.__new__, with io.BytesIO for rfile/wfile and
    a fake client_address. serve() is called only after its Server class is
    swapped for a stub that records the address and never binds. No
    serve_forever, no start_thread, no connect, no subprocess, no network.

Rows, one trit each (+1 holds, 0 advisory / gap, -1 broken):
  E. i2sd: bind constants, serve() loopback guard, peer gate, PING/PACK/
     UNPACK/ERR handshake, malformed PACK, uncapped readline, JUNIOR_BIND
     reach.
  G. Gate coverage: does juniorctl security()/health() name the 8767 lane
     that bitnetd.service actually serves on fallback?
  I. Installed overlay: where bitnetd.sh's fallback exec lands once it is
     /usr/local/bin/junior-bitnetd, and whether junior_bitnet resolves there.
  P. Port map: bind_guard_lint reserved ports, iot.py default endpoint, and
     one lane row each for Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Hard invariants (-1, exit 1): a non-loopback bind, serve() accepting
a non-loopback host, the peer gate answering a non-loopback client, a broken PING/PACK
round trip, bitnetd.sh losing its refuse-non-loopback guard, a flag set true.

Flags: mint false, socket false, layer_one false, fetch false,
ue5_launch false. Rate 0.05. Bind 127.0.0.1. Skill-pin family (T13) is
owned by the overnight build and skipped. Trits are 2-bit packed
(0->0, 1->1, -1->2, MSB first) and chained with
close = sha256(prev || pack || text)[:16]. Stdlib only.

Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM;
bitnetd 127.0.0.1:8765, i2sd 127.0.0.1:8767.

Run: python ports/live_beta_i2sd_fallback_security_handshake.py [--selftest] [--json]
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "rails" / "linux"
LOOPBACK = "127.0.0.1"
BITNETD = 8765
I2SD = 8767
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
GENESIS = "0" * 16
LANES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
LOOP_HOSTS = {"127.0.0.1", "localhost", "::1"}

SOURCES = {
    "i2sd": "rails/linux/i2sd.py",
    "sh": "rails/linux/bitnetd.sh",
    "unit": "rails/linux/bitnetd.service",
    "ctl": "rails/linux/juniorctl.py",
    "install": "rails/linux/install-overlay.sh",
    "stage": "rails/linux/debian/stage_overlay.sh",
    "iot": "rails/linux/iot.py",
    "lint": "rails/lint/bind_guard_lint.py",
}


# ---------------------------------------------------------------- close rule
def pack_trits(trits: list[int]) -> bytes:
    acc, n, out = 0, 0, bytearray()
    for t in trits:
        acc = (acc << 2) | (0 if t == 0 else (1 if t == 1 else 2))
        n += 2
        if n == 8:
            out.append(acc)
            acc, n = 0, 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- readers
def load(root: Path = ROOT) -> dict:
    out = {}
    for k, rel in SOURCES.items():
        p = root / rel
        out[k] = p.read_text(encoding="utf-8") if p.is_file() else ""
    return out


def _consts(src: str) -> dict:
    vals = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return vals
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    vals[t.id] = node.value.value
    return vals


def _func(src: str, name: str):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _func_text(src: str, name: str) -> str:
    f = _func(src, name)
    return ast.unparse(f) if f is not None else ""


def _readline_capped(src: str) -> bool:
    f = _func(src, "handle")
    if f is None:
        return False
    for node in ast.walk(f):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "readline"):
            return bool(node.args or node.keywords)
    return False


# ---------------------------------------------------------------- in-process handshake
def _talk(mod, line: bytes, peer: str = LOOPBACK) -> tuple[bytes, str]:
    h = object.__new__(mod.Handler)
    h.client_address = (peer, 40000)
    h.rfile = io.BytesIO(line)
    h.wfile = io.BytesIO()
    try:
        h.handle()
    except Exception as e:  # noqa: BLE001 - recorded as a row, not raised
        return h.wfile.getvalue(), type(e).__name__
    return h.wfile.getvalue(), ""


class _StubServer:
    calls: list = []

    def __init__(self, addr, handler):
        _StubServer.calls.append(addr)


def _serve_guard(mod, src: str) -> tuple[bool, bool, str]:
    """AST: serve() raises ValueError unless host is in a loopback-only set.
    Then serve() is called once for 127.0.0.1 with Server stubbed (never binds)."""
    f = _func(src, "serve")
    refused = False
    if f is not None:
        for node in ast.walk(f):
            if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.ops[0], ast.NotIn)
                    and isinstance(node.test.comparators[0], ast.Set)):
                hosts = {e.value for e in node.test.comparators[0].elts if isinstance(e, ast.Constant)}
                raises = any(isinstance(b, ast.Raise) for b in node.body)
                refused = raises and bool(hosts) and hosts <= LOOP_HOSTS
    real = mod.Server
    _StubServer.calls = []
    mod.Server = _StubServer
    try:
        mod.serve(LOOPBACK, I2SD)
    finally:
        mod.Server = real
    loop_ok = _StubServer.calls[-1:] == [(LOOPBACK, I2SD)]
    return refused, loop_ok, f"stub addrs {_StubServer.calls}"


def import_i2sd():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import rails.linux.i2sd as mod  # module import opens nothing
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return mod, ""


def handshake_rows(mod, err: str, src: str = "") -> list[tuple[str, int, str]]:
    if mod is None:
        return [("E.handshake", 0, f"i2sd not importable in-process ({err}); protocol not exercised")]
    rows = []
    refused, loop_ok, d = _serve_guard(mod, src or load()["i2sd"])
    rows.append(("E.serve_guard", 1 if refused and loop_ok else -1, f"non-loopback raise={refused} (ast); {d}"))
    out, exc = _talk(mod, b"PING\n", peer="10.0.0.5")
    rows.append(("E.peer_gate", 1 if out == b"" and not exc else -1, f"10.0.0.5 PING -> {out!r}"))
    out, exc = _talk(mod, b"PING\n")
    rows.append(("E.ping", 1 if out == b"PONG\n" else -1, f"PING -> {out!r}"))
    xs = [0.5, -0.4, 0.01, 1.0, -1.0]
    out, exc = _talk(mod, ("PACK " + ",".join(str(x) for x in xs) + "\n").encode())
    m = re.fullmatch(rb"OK (\d+) ([0-9.]+) ([0-9a-f]*)\n", out)
    if m and not exc:
        n, hexb = int(m.group(1)), m.group(3).decode()
        back, exc2 = _talk(mod, f"UNPACK {hexb} {n}\n".encode())
        trits = back.decode().strip()[3:].split(",") if back.startswith(b"OK ") else []
        ok = n == len(xs) and len(trits) == n and set(trits) <= {"-1", "0", "1"} and not exc2
        rows.append(("E.pack_unpack", 1 if ok else -1, f"PACK n={n} hex={hexb} -> UNPACK {trits}"))
    else:
        rows.append(("E.pack_unpack", -1, f"PACK -> {out!r} {exc}"))
    out, exc = _talk(mod, b"HELLO\n")
    rows.append(("E.err", 1 if out == b"ERR\n" else -1, f"HELLO -> {out!r}"))
    out, exc = _talk(mod, b"PACK x,y\n")
    rows.append(("E.malformed_pack", 1 if out.startswith(b"ERR") else 0,
                 f"PACK x,y -> {out!r} raised={exc or 'none'} (no ERR reply; socketserver drops the connection)"
                 if not out.startswith(b"ERR") else "PACK x,y -> ERR"))
    return rows


# ---------------------------------------------------------------- static rows
def static_rows(t: dict) -> list[tuple[str, int, str]]:
    rows = []
    c = _consts(t["i2sd"])
    host, port = c.get("HOST"), c.get("PORT")
    rows.append(("E.const", 1 if host == LOOPBACK and port == I2SD else -1, f"i2sd HOST={host} PORT={port}"))
    capped = _readline_capped(t["i2sd"])
    rows.append(("E.readline_cap", 1 if capped else 0,
                 "handle() readline has a limit" if capped else "handle() readline() has no length cap"))
    env = "JUNIOR_BIND" in t["i2sd"] or "environ" in t["i2sd"]
    unit_bind = re.search(r"^Environment=JUNIOR_BIND=(\S+)$", t["unit"], re.M)
    ub = unit_bind.group(1) if unit_bind else ""
    rows.append(("E.bind_reach", 1 if env else 0,
                 f"bitnetd.service JUNIOR_BIND={ub}; i2sd reads it" if env else
                 f"bitnetd.service JUNIOR_BIND={ub} never reaches i2sd.py; fallback listens {LOOPBACK}:{I2SD}"))

    sh = t["sh"]
    refuse = bool(re.search(r"refusing non-loopback.*exit 3", sh))
    cpp = f"--host {LOOPBACK} --port {BITNETD}" in sh
    fb = re.search(r'exec python3 "\$HERE/(i2sd\.py)"', sh)
    rows.append(("E.sh_guard", 1 if refuse and cpp else -1,
                 f"bitnetd.sh refuse-non-loopback={refuse} cpp {LOOPBACK}:{BITNETD}={cpp} fallback={bool(fb)}"))

    sec = _func_text(t["ctl"], "security")
    hl = _func_text(t["ctl"], "health")
    covers = str(I2SD) in sec or "i2sd" in sec
    rows.append(("G.security_covers_i2sd", 1 if covers else 0,
                 "juniorctl security() names i2sd 8767" if covers else
                 f"juniorctl security() reports bind {LOOPBACK}:{BITNETD} only; the {I2SD} fallback has no gate key"))
    hcov = str(I2SD) in hl or "i2sd" in hl
    rows.append(("G.health_names_lane", 1 if hcov or not fb else 0,
                 "juniorctl health() names the fallback lane" if hcov else
                 f"health() says bitnetd {LOOPBACK}:{BITNETD}; without GGUF+bitnet.cpp the unit serves {I2SD}"))
    bad_bind = [h for h in re.findall(r"\b(\d+\.\d+\.\d+\.\d+):\d+", sec + hl) if h != LOOPBACK]
    rows.append(("G.gate_bind", -1 if bad_bind else 1, f"gate binds non-loopback {bad_bind}" if bad_bind
                 else f"gate binds loopback only"))

    inst, stage = t["install"], t["stage"]
    sh_dest = re.search(r'cp "\$HERE/bitnetd\.sh" "\$DEST(/\S+?)"', inst)
    sh_dir = sh_dest.group(1).rsplit("/", 1)[0] if sh_dest else ""
    i2_dests = re.findall(r'"\$OUT(/[^"]+)/"', " ".join(ln for ln in stage.splitlines() if "i2sd.py" in ln))
    i2_dests += re.findall(r'"\$DEST(/[^"]+)"', " ".join(ln for ln in inst.splitlines() if "i2sd.py" in ln))
    lands = bool(sh_dir) and any(d.rstrip("/") == sh_dir or d == f"{sh_dir}/i2sd.py" for d in i2_dests)
    rows.append(("I.fallback_target", 1 if lands or not fb else 0,
                 f"installed fallback exec {sh_dir}/i2sd.py is staged" if lands else
                 f"installed junior-bitnetd execs {sh_dir or '?'}/i2sd.py; i2sd.py staged only at {i2_dests or 'nowhere'}"))
    pp = re.search(r'PYTHONPATH:-\$\(CDPATH= cd -- "\$HERE/(\.\./\.\.)"', sh)
    if sh_dir and pp:
        resolved = "/".join(sh_dir.split("/")[:-2]) or "/"
        has_pkg = any("junior_bitnet" in ln and "cp" in ln for ln in (inst + stage).splitlines())
        rows.append(("I.pythonpath", 1 if has_pkg else 0,
                     f"PYTHONPATH -> {resolved}; junior_bitnet staged" if has_pkg else
                     f"PYTHONPATH -> {resolved}; junior_bitnet not staged, i2sd import would raise ModuleNotFoundError"))
    else:
        rows.append(("I.pythonpath", 0, "bitnetd.sh PYTHONPATH default or install target not parsed"))

    m = re.search(r"RESERVED_PORTS\s*=\s*\(([^)]*)\)", t["lint"])
    reserved = {int(x) for x in re.findall(r"\d+", m.group(1))} if m else set()
    rows.append(("P.reserved", 1 if {BITNETD, I2SD} <= reserved else 0,
                 f"bind_guard_lint RESERVED_PORTS has {BITNETD},{I2SD}={({BITNETD, I2SD} <= reserved)}"))
    iot = _func(t["iot"], "pack_iot")
    dflt = {}
    if iot is not None:
        names = [a.arg for a in iot.args.args]
        for name, d in zip(names[-len(iot.args.defaults):], iot.args.defaults):
            if isinstance(d, ast.Constant):
                dflt[name] = d.value
    iot_ok = dflt.get("host") == LOOPBACK and dflt.get("port") == I2SD
    iot_bad = dflt.get("host") not in (None,) and dflt.get("host") not in LOOP_HOSTS
    rows.append(("P.iot_endpoint", -1 if iot_bad else (1 if iot_ok else 0),
                 f"iot.pack_iot default {dflt.get('host')}:{dflt.get('port')}"))
    for lane in LANES:
        ok = host == LOOPBACK and port == I2SD and refuse
        rows.append((f"P.lane.{lane}", 1 if ok else -1,
                     f"{lane} -> bitnetd {LOOPBACK}:{BITNETD} | fallback i2sd {host}:{port} (in-process map)"))
    rows.append(("P.flags", 1 if not any(FLAGS.values()) and RATE == 0.05 else -1,
                 f"flags all false={not any(FLAGS.values())} rate {RATE}"))
    return rows


def chain(rows: list[tuple[str, int, str]]) -> list[dict]:
    prev, out = GENESIS, []
    for name, trit, detail in rows:
        text = f"{name}|{trit}|{detail}"
        prev = close(prev, pack_trits([trit]), text)
        out.append({"row": name, "trit": trit, "detail": detail, "close": prev})
    return out


def run(root: Path = ROOT) -> list[dict]:
    mod, err = import_i2sd()
    return chain(static_rows(load(root)) + handshake_rows(mod, err))


# ---------------------------------------------------------------- selftest
GOOD = {
    "i2sd": ('HOST = "127.0.0.1"\nPORT = 8767\nimport os\nB = os.environ.get("JUNIOR_BIND")\n'
             "def handle(self):\n    self.rfile.readline(4096)\n"),
    "sh": ('BIND=${JUNIOR_BIND:-127.0.0.1:8765}\n  *) echo "refusing non-loopback $BIND" >&2; exit 3 ;;\n'
           "  exec x --host 127.0.0.1 --port 8765\n"
           'export PYTHONPATH=${PYTHONPATH:-$(CDPATH= cd -- "$HERE/../.." && pwd)}\n'
           'exec python3 "$HERE/i2sd.py"\n'),
    "unit": "Environment=JUNIOR_BIND=127.0.0.1:8765\n",
    "ctl": ("def health():\n    return {'bitnetd': '127.0.0.1:8765', 'i2sd': '127.0.0.1:8767'}\n"
            "def security():\n    return {'bind': '127.0.0.1:8765', 'i2sd_bind': '127.0.0.1:8767'}\n"),
    "install": ('cp "$HERE/bitnetd.sh" "$DEST/usr/local/bin/junior-bitnetd"\n'
                'cp "$HERE/i2sd.py" "$DEST/usr/local/bin/i2sd.py"\n'
                'cp -r "$ROOT/junior_bitnet" "$DEST/usr/junior_bitnet"\n'),
    "stage": "",
    "iot": "def pack_iot(xs, host='127.0.0.1', port=8767, timeout=0.2):\n    pass\n",
    "lint": "RESERVED_PORTS = (8765, 8767)\n",
}


def selftest() -> int:
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            fails.append(name)

    check("pack 1,-1,0", pack_trits([1, -1, 0]) == bytes([0b01100000]))
    check("pack 4", pack_trits([1, 1, 1, 1]) == bytes([0b01010101]))
    c = close(GENESIS, b"\x01", "x")
    check("close 16 hex", len(c) == 16 and int(c, 16) >= 0)
    check("close depends on prev", c != close("1" * 16, b"\x01", "x"))
    check("close rule", c == hashlib.sha256(GENESIS.encode() + b"\x01" + b"x").hexdigest()[:16])

    good = {r[0]: r[1] for r in static_rows(dict(GOOD))}
    check("good fixture all +1", all(v == 1 for v in good.values()))
    bad = dict(GOOD, i2sd=GOOD["i2sd"].replace("PORT = 8767", "PORT = 8768"))
    check("port drift -> -1", {r[0]: r[1] for r in static_rows(bad)}["E.const"] == -1)
    gap = dict(GOOD, ctl="def health():\n    return {'bitnetd': '127.0.0.1:8765'}\n"
                         "def security():\n    return {'bind': '127.0.0.1:8765'}\n",
               install='cp "$HERE/bitnetd.sh" "$DEST/usr/local/bin/junior-bitnetd"\n',
               stage='cp "$HERE/i2sd.py" "$HERE/backend.py" "$OUT/usr/lib/junioros/"\n',
               i2sd='HOST = "127.0.0.1"\nPORT = 8767\ndef handle(self):\n    self.rfile.readline()\n')
    g = {r[0]: r[1] for r in static_rows(gap)}
    for k in ("G.security_covers_i2sd", "G.health_names_lane", "I.fallback_target",
              "I.pythonpath", "E.readline_cap", "E.bind_reach"):
        check(f"gap {k} -> 0", g[k] == 0)
    check("gap keeps no -1", -1 not in g.values())
    nog = dict(GOOD, sh=GOOD["sh"].replace("exit 3", "exit 0"))
    check("lost refuse guard -> -1", {r[0]: r[1] for r in static_rows(nog)}["E.sh_guard"] == -1)
    iotw = dict(GOOD, iot=GOOD["iot"].replace("port=8767", "port=8768"))
    check("iot port drift -> 0", {r[0]: r[1] for r in static_rows(iotw)}["P.iot_endpoint"] == 0)

    ch = chain([("a", 1, "x"), ("b", 0, "y")])
    check("chain links", ch[1]["close"] == close(ch[0]["close"], pack_trits([0]), "b|0|y"))

    mod, err = import_i2sd()
    if mod is not None:
        hs = {r[0]: r[1] for r in handshake_rows(mod, err)}
        for k in ("E.serve_guard", "E.peer_gate", "E.ping", "E.pack_unpack", "E.err"):
            check(f"live {k} +1", hs.get(k) == 1)
        check("stub restored", mod.Server is not _StubServer)
    check("flags false", not any(FLAGS.values()) and RATE == 0.05)
    own = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    banned = {"create_connection", "connect", "bind", "serve_forever", "start_thread", "Popen", "urlopen"}
    hits = [n.func.attr for n in ast.walk(own) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr in banned]
    check("no socket/exec calls in this file", not hits)

    print(f"selftest {'pass' if not fails else 'FAIL'}: "
          f"{'all checks held' if not fails else ', '.join(fails)}"
          f"{'' if mod is not None else f' (handler not importable: {err})'}")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    rows = run()
    if "--json" in argv:
        print(json.dumps({"rows": rows, "flags": FLAGS, "rate": RATE, "bind": LOOPBACK,
                          "close": rows[-1]["close"] if rows else GENESIS}, indent=2))
    else:
        mark = {1: "+", 0: "0", -1: "-"}
        for r in rows:
            print(f"{mark[r['trit']]} {r['close']} {r['row']}: {r['detail']}")
        n = {t: sum(1 for r in rows if r["trit"] == t) for t in (1, 0, -1)}
        print(f"{n[1]} pass, {n[0]} advisory, {n[-1]} fail; close {rows[-1]['close'] if rows else GENESIS}")
    return 1 if any(r["trit"] == -1 for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
