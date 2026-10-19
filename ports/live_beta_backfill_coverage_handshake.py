#!/usr/bin/env python3
"""Live beta step 5: backfill_check coverage handshake + port map.

Advances docs/BETA_TO_OS.md, section "Live beta", step 5:
"`python scripts/backfill_check.py` exits 0".

The exit-code side of step 5 already landed elsewhere (exit-code trits and
an SPRT gate).  This file takes the coverage side: when step 5 exits 0, what
does that 0 actually attest?  scripts/backfill_check.py is a presence-only
gate over a fixed EXPECTED list.  This script, in-process:

  1. reads backfill_check.py with ast (EXPECTED list, 1-if-missing-else-0,
     stdlib-only imports, no network or subprocess names);
  2. loads it with importlib and calls main() with stdout captured, once on
     this clone (expect 0, "missing none") and once with ROOT pointed at an
     empty temp dir (expect 1, every path missing), so the gate is shown to
     gate, not just to pass;
  3. checks each EXPECTED path is a non-empty file (an empty file still
     passes a presence-only gate);
  4. maps every "done (path)" claim in docs/COMPILED_BACKLOG.md to the tree
     (exact or under adaptations/) and to EXPECTED (T-series skipped: the
     overnight build owns it);
  5. maps the Live beta entrypoints (steps 1-5) and the port sources
     (bitnetd 127.0.0.1:8765, i2sd 127.0.0.1:8767) to EXPECTED, reading the
     port literals statically.  Bind policy itself (step 6) is not re-checked.

Uncovered paths are advisories (0) with the smallest additive fix: add them
to a new EXPECTED-style list, never rewrite backfill_check.py.

Ports touched: bitnetd 127.0.0.1:8765 and i2sd 127.0.0.1:8767 (static port
map only).  Hand-off between Home, OSai, FieldCore, Flagstaff and JuniorLLM
is one trit pack (+1 pass, 0 advisory, -1 fail), base rate 0.05, closed with
sha256(prev || pack || text)[:16].  No sockets, no subprocess, no shell, no
download, no model pull.  Stdlib only.

    python ports/live_beta_backfill_coverage_handshake.py
    python ports/live_beta_backfill_coverage_handshake.py --json
    python ports/live_beta_backfill_coverage_handshake.py --selftest

Exit 1 if any row is -1 (backfill_check missing or not exiting 0 on this
clone, the empty-root flip not exiting 1, a backlog "done" path with nothing
on disk, a port source off 127.0.0.1, or a flag flipped).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKFILL = "scripts/backfill_check.py"
BACKLOG = "docs/COMPILED_BACKLOG.md"
GENESIS = "0" * 16
RATE = 0.05
PARTIES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")

FLAGS = {
    "bind": "127.0.0.1",
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
    "model_pull": False,
}

# Live beta entrypoints named in BETA_TO_OS steps 1-5 that live in this repo.
ENTRYPOINTS = {
    "step1": "rails/linux/juniorctl.py",
    "step2": "rails/linux/juniorctl.py",
    "step3a": "tests/test_junior_aie.py",
    "step3b": "tests/test_linux_security.py",
    "step5": BACKFILL,
}
OUT_OF_REPO = {"step4": "JuniorClimbs scripts/stonefield_stdlib_probe.py"}

# Port sources: path -> (service, expected port).
PORT_SOURCES = {
    "rails/linux/bitnetd.service": ("bitnetd", 8765),
    "rails/linux/bitnetd.sh": ("bitnetd", 8765),
    "rails/linux/i2sd.py": ("i2sd", 8767),
}

WILDCARD = ".".join(["0"] * 4)
BANNED_NAMES = {"socket", "subprocess", "urllib", "http", "requests", "os.system"}
DONE_RE = re.compile(r"\b([A-Z]\d+)\s+done\s*\(([^)]*)\)")
TOP_DIRS = ("rails/", "ports/", "adaptations/", "docs/", "scripts/", "tests/")
PATHISH = re.compile(r"[\w./-]+\.(?:py|md|json|sh|service)\b|[\w-]+/[\w./-]+")


def row(key: str, trit: int, note: str) -> dict:
    return {"key": key, "trit": trit, "note": note}


def parse_backfill(src: str) -> dict:
    tree = ast.parse(src)
    expected, imports, names = None, set(), set()
    gate = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "EXPECTED":
                    try:
                        expected = list(ast.literal_eval(node.value))
                    except ValueError:
                        expected = None
        elif isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.IfExp):
            b, o = node.value.body, node.value.orelse
            if (isinstance(b, ast.Constant) and b.value == 1
                    and isinstance(o, ast.Constant) and o.value == 0):
                gate = True
    return {"expected": expected, "imports": sorted(imports),
            "names": names, "gate": gate}


def run_backfill(path: Path, root_override: Path | None = None) -> tuple:
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_backfill_check_hp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if root_override is not None:
        mod.ROOT = root_override
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main()
    return rc, buf.getvalue().strip()


def resolve(root: Path, rel: str) -> str | None:
    for cand in (rel, "adaptations/" + rel):
        p = root / cand
        if p.is_file() or p.is_dir():
            return cand
    return None


def backlog_claims(text: str) -> list:
    out = []
    for item, inner in DONE_RE.findall(text):
        for m in PATHISH.findall(inner):
            m = m.rstrip(".,")
            if item.startswith("T"):
                continue  # T-series is owned by the overnight build; skipped
            if m.endswith((".py", ".md", ".json", ".sh")) or m.startswith(TOP_DIRS):
                out.append((item, m))
    return out


def load(root: Path) -> dict:
    s = {"root": str(root), "rows": [], "map": {}}
    rows = s["rows"]
    bf = root / BACKFILL
    if not bf.is_file():
        rows.append(row("backfill.present", -1, BACKFILL + " missing"))
        return s
    rows.append(row("backfill.present", 1, BACKFILL))
    info = parse_backfill(bf.read_text(encoding="utf-8"))
    exp = info["expected"]
    if not exp:
        rows.append(row("backfill.expected", -1, "EXPECTED list not literal"))
        return s
    rows.append(row("backfill.expected", 1, "%d paths" % len(exp)))
    bad = sorted(n for n in info["names"] | set(info["imports"]) if n in BANNED_NAMES)
    rows.append(row("backfill.stdlib", -1 if bad else 1,
                    "imports=" + ",".join(info["imports"]) + (" banned=" + ",".join(bad) if bad else "")))
    rows.append(row("backfill.gate_shape", 1 if info["gate"] else 0,
                    "return 1 if missing else 0" if info["gate"] else "gate return not found by ast"))

    rc, out = run_backfill(bf)
    rows.append(row("step5.exit_here", 1 if rc == 0 else -1, "rc=%s out=%r" % (rc, out)))
    with tempfile.TemporaryDirectory() as td:
        rc2, out2 = run_backfill(bf, Path(td))
    flip_ok = rc2 == 1 and all(p in out2 for p in exp)
    rows.append(row("step5.exit_empty_root", 1 if flip_ok else -1,
                    "rc=%s, %d/%d listed missing" % (rc2, sum(p in out2 for p in exp), len(exp))))

    empty = [p for p in exp if (root / p).is_file() and (root / p).stat().st_size == 0]
    rows.append(row("expected.non_empty", 0 if empty else 1,
                    "empty but passing: " + ",".join(empty) if empty else "all %d non-empty" % len(exp)))

    expset = set(exp)
    bl = root / BACKLOG
    claims = backlog_claims(bl.read_text(encoding="utf-8")) if bl.is_file() else []
    for item, rel in claims:
        hit = resolve(root, rel)
        key = "backlog.%s" % item
        s["map"][key] = {"claim": rel, "resolved": hit, "in_expected": hit in expset}
        if hit is None:
            rows.append(row(key, -1, "%s claims done, nothing at %s" % (item, rel)))
        elif hit in expset:
            rows.append(row(key, 1, "%s covered by backfill" % hit))
        else:
            drift = "" if hit == rel else " (backlog says %s)" % rel
            rows.append(row(key, 0, "%s exists, not in EXPECTED%s; fix: add to a new list" % (hit, drift)))

    for step, rel in ENTRYPOINTS.items():
        p = root / rel
        key = "entry.%s" % step
        s["map"][key] = {"path": rel, "present": p.is_file(), "in_expected": rel in expset}
        if not p.is_file():
            rows.append(row(key, 0, "%s not in this clone" % rel))
        elif rel in expset or rel == BACKFILL:
            rows.append(row(key, 1, "%s covered" % rel))
        else:
            rows.append(row(key, 0, "%s present, exit 0 does not attest it" % rel))
    for step, ref in OUT_OF_REPO.items():
        rows.append(row("entry.%s" % step, 0, "%s is out of repo; not attestable here" % ref))

    for rel, (svc, port) in PORT_SOURCES.items():
        p = root / rel
        key = "port.%s" % rel.rsplit("/", 1)[-1]
        if not p.is_file():
            rows.append(row(key, 0, "%s missing" % rel))
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        ports = sorted({int(x) for x in re.findall(r"\b(87\d\d)\b", text)})
        wild = WILDCARD in text
        s["map"][key] = {"service": svc, "port": port, "found": ports,
                         "bind": "127.0.0.1", "in_expected": rel in expset}
        if wild:
            rows.append(row(key, -1, "%s mentions the wildcard bind" % rel))
        elif port not in ports:
            rows.append(row(key, 0, "%s: %d not found statically" % (rel, port)))
        elif rel in expset:
            rows.append(row(key, 1, "%s 127.0.0.1:%d covered" % (svc, port)))
        else:
            rows.append(row(key, 0, "%s 127.0.0.1:%d source %s not in EXPECTED" % (svc, port, rel)))

    flipped = [k for k, v in FLAGS.items() if k != "bind" and v is not False]
    rows.append(row("flags", -1 if flipped or FLAGS["bind"] != "127.0.0.1" else 1,
                    "bind 127.0.0.1; mint/socket/layer_one/fetch/ue5_launch/model_pull false"))
    return s


def pack(rows: list) -> bytes:
    return bytes((r["trit"] + 1) for r in rows)


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def report(s: dict, prev: str = GENESIS) -> dict:
    rows = s["rows"]
    text = "\n".join("%+d %s %s" % (r["trit"], r["key"], r["note"]) for r in rows)
    pk = pack(rows)
    return {
        "item": "BETA_TO_OS Live beta step 5: python scripts/backfill_check.py exits 0 (coverage side)",
        "parties": list(PARTIES),
        "rate": RATE,
        "flags": FLAGS,
        "rows": rows,
        "map": s["map"],
        "pass": sum(r["trit"] == 1 for r in rows),
        "advisory": sum(r["trit"] == 0 for r in rows),
        "fail": sum(r["trit"] == -1 for r in rows),
        "pack": pk.hex(),
        "prev": prev,
        "close": close(prev, pk, text),
        "text": text,
    }


FAKE_BACKFILL = '''from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["docs/A.md", "docs/B.md"]
def main():
    missing = [p for p in EXPECTED if not (ROOT / p).is_file()]
    print("missing", missing or "none")
    return 1 if missing else 0
'''


def selftest() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        (r / "scripts").mkdir()
        (r / "docs").mkdir()
        (r / "scripts/backfill_check.py").write_text(FAKE_BACKFILL)
        (r / "docs/A.md").write_text("a")
        s = load(r)
        keys = {x["key"]: x["trit"] for x in s["rows"]}
        if keys.get("step5.exit_here") != -1:
            print("selftest: missing B.md should fail step5.exit_here", file=sys.stderr)
            ok = False
        (r / "docs/B.md").write_text("")
        s = load(r)
        keys = {x["key"]: x["trit"] for x in s["rows"]}
        if keys.get("step5.exit_here") != 1 or keys.get("step5.exit_empty_root") != 1:
            print("selftest: exit/flip rows wrong %r" % keys, file=sys.stderr)
            ok = False
        if keys.get("expected.non_empty") != 0:
            print("selftest: empty B.md should be advisory", file=sys.stderr)
            ok = False
        (r / "docs/COMPILED_BACKLOG.md").write_text("Z1 done (ports/nothere.py).\n")
        s = load(r)
        if {x["key"]: x["trit"] for x in s["rows"]}.get("backlog.Z1") != -1:
            print("selftest: missing done path should fail", file=sys.stderr)
            ok = False
    real = load(ROOT)
    a, b = report(real), report(real)
    if a["close"] != b["close"] or len(a["close"]) != 16:
        print("selftest: close not deterministic", file=sys.stderr)
        ok = False
    if a["close"] != close(GENESIS, bytes.fromhex(a["pack"]), a["text"]):
        print("selftest: close rule mismatch", file=sys.stderr)
        ok = False
    if len(bytes.fromhex(a["pack"])) != len(a["rows"]):
        print("selftest: pack length", file=sys.stderr)
        ok = False
    if any(v is not False for k, v in FLAGS.items() if k != "bind") or FLAGS["bind"] != "127.0.0.1":
        ok = False
    print("selftest", "pass" if ok else "fail")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--prev", default=GENESIS, help="previous close (16 hex)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    r = report(load(Path(a.root)), a.prev)
    if a.json:
        print(json.dumps({k: v for k, v in r.items() if k != "text"}, indent=2, sort_keys=True))
    else:
        print(r["item"])
        print(r["text"])
        print("pass %d advisory %d fail %d close %s rate %s" %
              (r["pass"], r["advisory"], r["fail"], r["close"], r["rate"]))
    return 1 if r["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
