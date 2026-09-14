"""X=Junior winsor, Y=TQ1_0, Z=IQ1_S. Diagonal importance, not a model Hessian."""
from __future__ import annotations

from junior_bitnet.math import absmean
from junior_bitnet.winsor import pack as winsor_pack
from ports.flagstaff_balance import check


def _xs(note: str) -> list[float]:
    return [float((ord(c) % 13) - 6) for c in (note or "x")] or [0.2]


def importance(xs: list[float]) -> list[float]:
    """|x| / sum(|x|) — diagonal proxy, not H = d²L/dW²."""
    s = sum(abs(x) for x in xs) or 1.0
    return [abs(x) / s for x in xs]


def profit(xs: list[float], trit: list[int], g: float) -> float:
    l1 = sum(abs(x - t * g) for x, t in zip(xs, trit)) / max(1, len(xs))
    zeros = sum(1 for t in trit if t == 0) / max(1, len(trit))
    return round((1.0 / (1.0 + l1)) * (0.5 + 0.5 * (1.0 - zeros)), 4)


def xyz(note: str) -> dict:
    gate = check(note)
    xs = _xs(note)
    w = winsor_pack(xs)
    am, ag = absmean(xs)
    imp = importance(xs)
    return {
        "ok": gate.get("ok"),
        "votes": gate.get("votes"),
        "area": gate.get("area"),
        "port": gate.get("port"),
        "X_junior": {"gamma": w.get("gamma"), "sparsity": w.get("sparsity"), "profit": profit(xs, w.get("trit") or [], (w.get("gamma") or 0) + 1e-7)},
        "Y_tq1": {"role": "gguf-ternary-weights", "bpw": 1.69, "on_wire": False},
        "Z_iq1s": {"role": "codebook+imatrix", "bpw": 1.56, "on_wire": False, "needs_calib": True},
        "importance_diag": {"n": len(imp), "max": max(imp) if imp else 0, "hessian": False},
        "infer": True,
        "download": False,
    }
