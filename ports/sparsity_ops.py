"""Sparsity next to Flagstaff + FieldCore. TP stays checklist."""
from __future__ import annotations

from junior_bitnet.math import absmean
from junior_bitnet.winsor import pack
from ports.fieldcore_spine import expand
from ports.flagstaff_balance import check
from ports.gguf_t3 import run as t3


def ops(note: str = "journal field note") -> dict:
    gate = check(note)
    xs = [float((ord(c) % 13) - 6) for c in (note or "x")]
    w = pack(xs)
    am, _ = absmean(xs)
    am_sp = sum(1 for t in am if t == 0) / max(1, len(am))
    fc = expand(n=32, k=8) if gate.get("ok") else {"skipped": True}
    return {
        "ok": gate.get("ok"),
        "votes": gate.get("votes"),
        "winsor_sparsity": w.get("sparsity"),
        "absmean_sparsity": round(am_sp, 4),
        "firing": round(1.0 - float(w.get("sparsity") or 0), 4),
        "fieldcore": {"n": fc.get("n"), "gamma": fc.get("gamma")},
        "brain": "jsonl-if-ok",
        "t3": t3("sparsity"),
        "tensor_split": False,
        "tp_needs": ["local-gguf", "llama-cli", "two-devices"],
        "download": False,
    }
