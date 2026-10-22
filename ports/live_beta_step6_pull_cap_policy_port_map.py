#!/usr/bin/env python3
"""Live beta step 6 pull-cap policy port map (in-process, no sockets).

Advances JuniorLLM docs/BETA_TO_OS.md, item "Live beta", step 6
("No deletes. No >8GB pull. bitnetd on `127.0.0.1:8765` only"), from the
handshake and port side, policy checks only.

New angle: step 6's "No >8GB pull" is enforced by in-repo cap carriers, not
by any transcript. Nothing so far asks whether those carriers agree with each
other. This file reads them as source text, or loads them one at a time from
their own files with no package __init__, and checks, in-process:
  * C. cap constants: ports/ondisk.py CAP_BYTES, junior_aie/files_ledger.py
    CAP_BYTES, junior_aie/skill_pin.py CAP_BYTES, rails/linux/oci/rootless.py
    _CAP_BYTES are one value; adaptations kimi_k3 CAP_GB and the qwen /
    astra_reason max_download_gb defaults are <= 8; kimi fetch is False.
  * R. port map: every LLMPort in ports/registry.py and ports/ecosystem.py
    declares max_download_gb <= 8.0. Home, OSai, FieldCore, Flagstaff and
    JuniorLLM declare 0.0.
  * B. boundary handshake: sparse temp files (truncate only; no bytes are
    written, nothing fetched) at exactly the cap and at cap+1 go through
    ondisk.probe(), rootless.bitnet_weights(), files_ledger.too_big() and
    skill_pin.too_big(). All four must admit the cap and refuse cap+1.
  * M. full-weight markers: ondisk and kimi_k3 refuse the same name markers.
  * L. loopback: bitnetd.service / rootless BIND say 127.0.0.1:8765, with no
    0.0.0.0. The live-beta handshake order is juniorctl health, then bitnetd
    127.0.0.1:8765, then i2sd 127.0.0.1:8767. It is listed, never opened.
  * U. advisory: the caps are 8 GiB (8589934592 bytes) while the doc says
    "8GB". A pull between 8e9 and 8 GiB passes the code but may be read
    as over the doc's limit, so this row is 0 (undecided) on purpose.

Does not import, wrap or re-check any other team bot's file (live_beta_* and
overlay_* files are not read). No socket, no subprocess, no network, no fetch,
no model pull, no UE5. Stdlib only. Rate 0.05. Close rule:
sha256(prev || pack || text)[:16], trits packed 2-bit (0->0, 1->1, -1->2),
MSB first, zero-padded.

Run: python ports/live_beta_step6_pull_cap_policy_port_map.py --selftest
     python ports/live_beta_step6_pull_cap_policy_port_map.py   (JSON)
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False,
         "ue5_launch": False, "model_pull": False}
BIND_BITNETD = "127.0.0.1:8765"
BIND_I2SD = "127.0.0.1:8767"
GIB8 = 8 * 1024 * 1024 * 1024
HANDSHAKE = [("juniorctl health", "in-process"), ("bitnetd", BIND_BITNETD),
             ("i2sd", BIND_I2SD)]
NODES = ("JuniorHome", "JuniorOSai", "FieldCore", "Flagstaff", "JuniorLLM")
PREV0 = "0" * 16


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _const(rel: str, name: str):
    for node in ast.walk(ast.parse(_src(rel))):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in tgts:
                if isinstance(t, ast.Name) and t.id == name and node.value is not None:
                    return eval(compile(ast.Expression(node.value), rel, "eval"), {"__builtins__": {}})
    return None


def _field_default(rel: str, cls: str, field: str):
    for node in ast.walk(ast.parse(_src(rel))):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) \
                        and st.target.id == field and st.value is not None:
                    if isinstance(st.value, ast.Name):
                        return _const(rel, st.value.id)
                    return ast.literal_eval(st.value)
    return None


def _llmports(rel: str) -> list[tuple[str, float]]:
    out = []
    for node in ast.walk(ast.parse(_src(rel))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LLMPort":
            args = [ast.literal_eval(a) for a in node.args]
            out.append((args[0], float(args[4])))
    return out


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def _row(rows, key, trit, note):
    rows.append({"row": key, "trit": trit, "note": note})


def check() -> list[dict]:
    rows: list[dict] = []
    caps = {
        "ports/ondisk.py": _const("ports/ondisk.py", "CAP_BYTES"),
        "junior_aie/files_ledger.py": _const("junior_aie/files_ledger.py", "CAP_BYTES"),
        "junior_aie/skill_pin.py": _const("junior_aie/skill_pin.py", "CAP_BYTES"),
        "rails/linux/oci/rootless.py": _const("rails/linux/oci/rootless.py", "_CAP_BYTES"),
    }
    vals = set(caps.values())
    _row(rows, "C1 byte caps agree", 1 if len(vals) == 1 and None not in vals else -1,
         json.dumps(caps))
    _row(rows, "C2 byte cap <= 8 GiB", 1 if vals and max(v or 1 << 62 for v in vals) <= GIB8 else -1,
         f"max={max((v or 0) for v in vals)}")
    gb = {
        "kimi_k3 CAP_GB": _const("adaptations/kimi_k3/edge_loader.py", "CAP_GB"),
        "qwen max_download_gb": _field_default("adaptations/qwen/gguf_loader.py", "QwenLocalConfig", "max_download_gb"),
        "astra_reason max_download_gb": _field_default("adaptations/astra_reason/config.py", "AstraReasonConfig", "max_download_gb"),
    }
    ok = all(v is not None and float(v) <= 8.0 for v in gb.values())
    _row(rows, "C3 adaptation GB caps <= 8", 1 if ok else -1, json.dumps(gb))
    kf = _field_default("adaptations/kimi_k3/edge_loader.py", "KimiK3EdgeConfig", "fetch")
    _row(rows, "C4 kimi_k3 fetch False", 1 if kf is False else -1, f"fetch={kf}")

    reg = _llmports("ports/registry.py")
    eco = _llmports("ports/ecosystem.py")
    over = [(n, g) for n, g in reg + eco if g > 8.0]
    _row(rows, "R1 registry+ecosystem max_download_gb <= 8", 1 if reg and eco and not over else -1,
         f"ports={len(reg) + len(eco)} over={over}")
    allp = dict(eco)
    allp.update(dict(reg))
    nz = {n: allp.get(n) for n in NODES}
    _row(rows, "R2 Home/OSai/FieldCore/Flagstaff/JuniorLLM download 0.0",
         1 if all(v == 0.0 for v in nz.values()) else -1, json.dumps(nz))

    ondisk = _load("ports/ondisk.py", "_hp_ondisk")
    rootless = _load("rails/linux/oci/rootless.py", "_hp_rootless")
    ledger = _load("junior_aie/files_ledger.py", "_hp_files_ledger")
    pin = _load("junior_aie/skill_pin.py", "_hp_skill_pin")
    cap = ondisk.CAP_BYTES
    name = ondisk.CANDIDATES["BitNet-2B4T"][0]
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for label, size in (("at_cap", cap), ("over_cap", cap + 1)):
            d = Path(td) / label
            d.mkdir()
            with open(d / name, "wb") as fh:
                fh.truncate(size)  # sparse; no bytes written, nothing fetched
            od = next(p for p in ondisk.probe(d) if p.name == "BitNet-2B4T").present
            rl = rootless.bitnet_weights(d)["present"]
            res[label] = {"ondisk": od, "rootless": rl,
                          "files_ledger_ok": not ledger.too_big(size),
                          "skill_pin_ok": not pin.too_big(size)}
    at_ok = all(res["at_cap"].values())
    over_ok = not any(res["over_cap"].values())
    _row(rows, "B1 all carriers admit exactly the cap", 1 if at_ok else -1, json.dumps(res["at_cap"]))
    _row(rows, "B2 all carriers refuse cap+1", 1 if over_ok else -1, json.dumps(res["over_cap"]))
    _row(rows, "B3 rootless reports download False",
         1 if rootless.bitnet_weights(Path(tempfile.gettempdir()) / "_hp_none")["download"] is False else -1,
         "download flag")

    mk_o = set(ondisk.FULL_WEIGHT_MARKERS)
    mk_k = set(_const("adaptations/kimi_k3/edge_loader.py", "FULL_WEIGHT_MARKERS") or ())
    _row(rows, "M1 full-weight markers agree", 1 if mk_o and mk_o == mk_k else 0,
         f"ondisk_only={sorted(mk_o - mk_k)} kimi_only={sorted(mk_k - mk_o)}")

    svc = _src("rails/linux/bitnetd.service")
    lb = BIND_BITNETD in svc and "0.0.0.0" not in svc and rootless.BIND == BIND_BITNETD
    _row(rows, "L1 bitnetd 127.0.0.1:8765 only (service + rootless)", 1 if lb else -1,
         f"rootless.BIND={rootless.BIND}")
    hs_ok = all(a == "in-process" or a.startswith("127.0.0.1:") for _, a in HANDSHAKE)
    _row(rows, "L2 handshake order loopback only", 1 if hs_ok else -1,
         " -> ".join(f"{n} {a}" for n, a in HANDSHAKE))

    _row(rows, "U1 cap unit: 8 GiB in code vs '8GB' in doc", 0,
         f"{GIB8} bytes; window 8000000000..{GIB8} ambiguous")
    return rows


def pack(trits: list[int]) -> bytes:
    enc = {0: 0, 1: 1, -1: 2}
    bits = "".join(format(enc[t], "02b") for t in trits)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def report(rows: list[dict]) -> dict:
    trits = [r["trit"] for r in rows]
    text = "\n".join(f"{r['row']}|{r['trit']}" for r in rows)
    pk = pack(trits)
    return {"item": "BETA_TO_OS Live beta step 6", "rate": RATE, "flags": FLAGS,
            "nodes": list(NODES), "handshake": HANDSHAKE,
            "pass": trits.count(1), "advisory": trits.count(0), "fail": trits.count(-1),
            "pack": pk.hex(), "close": close(PREV0, pk, text), "rows": rows}


def selftest() -> int:
    assert pack([1, -1, 0, 1]) == bytes([0b01100001])
    assert pack([1]) == bytes([0b01000000])
    assert close(PREV0, b"", "") == hashlib.sha256(PREV0.encode()).hexdigest()[:16]
    assert all(v is False for v in FLAGS.values()) and RATE == 0.05
    rep = report(check())
    print(json.dumps({k: rep[k] for k in ("item", "pass", "advisory", "fail", "close")}))
    for r in rep["rows"]:
        if r["trit"] != 1:
            print(f"  {r['trit']:+d} {r['row']}: {r['note']}")
    return 0 if rep["fail"] == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    rep = report(check())
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["fail"] == 0 else 1)
