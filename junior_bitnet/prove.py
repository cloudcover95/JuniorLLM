"""Hypothesis tests for *this* BitNet stack. Not the Microsoft paper."""
from __future__ import annotations

from adaptations.omega_cad.quant import absmean, i2s_pack
from junior_bitnet.compile_sheet import compile_sheet


def prove() -> dict:
    xs = [40.0, 20.0, 8.0, 0.2, -3.0]
    trits, scale = absmean(xs)
    h1 = all(t in (-1, 0, 1) for t in trits)
    h2 = scale > 0
    h3 = len(i2s_pack(trits)) >= 1
    blocked = compile_sheet("TITLE: x\nREV A")
    h4 = not blocked.ready and blocked.data["run_iq"] is False
    signed = compile_sheet(
        "TITLE: BRACKET\nREV A\nELEV\nHEIGHT: 8",
        fixes=[{"field": "height", "value": "8", "by": "nico"}],
    )
    h5 = signed.ready and signed.data["run_iq"] is True
    ok = h1 and h2 and h3 and h4 and h5
    return {
        "ok": ok,
        "hypotheses": {
            "trits_in_alphabet": h1,
            "absmean_scale_positive": h2,
            "i2s_packs": h3,
            "no_height_blocks_iq": h4,
            "signed_height_ready": h5,
        },
    }
