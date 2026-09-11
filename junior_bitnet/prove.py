"""Hypothesis tests for this stack across CAD, night, zk pack, BitLinear."""
from __future__ import annotations

from junior_bitnet.bitlinear import bitlinear
from junior_bitnet.compile_sheet import compile_sheet
from junior_bitnet.math import absmean, binarize, sparsity


def prove() -> dict:
    xs = [40.0, 20.0, 8.0, 0.2, -3.0]
    trits, scale = absmean(xs)
    bits = binarize(xs)
    h = {
        "trits_in_alphabet": all(t in (-1, 0, 1) for t in trits),
        "absmean_scale_positive": scale > 0,
        "absmean_has_zeros": 0 in trits,
        "binarize_no_zeros": 0 not in bits and all(b in (-1, 1) for b in bits),
        "sparsity_unit": 0.0 <= sparsity(trits) <= 1.0,
    }
    blocked = compile_sheet("TITLE: x\nREV A")
    signed = compile_sheet(
        "TITLE: BRACKET\nREV A\nELEV\nHEIGHT: 8",
        fixes=[{"field": "height", "value": "8", "by": "nico"}],
    )
    h["no_height_blocks_iq"] = (not blocked.ready) and (blocked.data["run_iq"] is False)
    h["signed_height_ready"] = signed.ready and signed.data["run_iq"] is True

    bl = bitlinear([0.5, -0.2, 0.1], [1.2, -0.4, 0.05])
    h["bitlinear_finite"] = abs(bl["y"]) < 1e6
    h["bitlinear_w_trits"] = all(t in (-1, 0, 1) for t in bl["wq"])

    from bitnet_night.dream_mesh import DIM, tick

    st = [1, -1, 0] * (DIM // 3) + [0] * (DIM % 3)
    nxt = tick(st, 0)
    h["night_closed_trits"] = all(v in (-1, 0, 1) for v in nxt) and len(nxt) == DIM

    from lattice_zk.pack import pack, unpack

    raw = [-1, 0, 1, 1, 0, -1]
    h["zk_pack_roundtrip"] = unpack(pack(raw), len(raw)) == raw

    try:
        from adaptations.astra_reason.rigid_iq import embed

        e = embed("field beta")
        h["iq_embed_trits"] = all(v in (-1, 0, 1) for v in e)
    except Exception:
        h["iq_embed_trits"] = False

    return {"ok": all(h.values()), "hypotheses": h, "sparsity": sparsity(trits), "bitlinear_y": bl["y"]}
