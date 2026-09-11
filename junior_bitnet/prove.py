"""Hypothesis tests: BitNet alphabet + JuniorTeqp + palace/ZK + catalog."""
from __future__ import annotations

from junior_bitnet.bitlinear import bitlinear
from junior_bitnet.compile_sheet import compile_sheet
from junior_bitnet.math import absmean, binarize, sparsity
from junior_bitnet.palace import Palace
from junior_bitnet.refprop import Library
from junior_bitnet.teqp import a_helmholtz, props, rho


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
    np_ = props(nxt)
    h["night_teqp_rho"] = 0.0 <= np_.rho <= 1.0
    h["teqp_A_finite"] = abs(a_helmholtz(nxt, 0.22)) < 1e6

    from lattice_zk.pack import pack, unpack

    raw = [-1, 0, 1, 1, 0, -1]
    h["zk_pack_roundtrip"] = unpack(pack(raw), len(raw)) == raw
    h["zk_teqp_rho"] = abs(rho(raw) - 4 / 6) < 1e-9

    try:
        from adaptations.astra_reason.rigid_iq import embed

        e = embed("field beta dry open V4")
        h["iq_embed_trits"] = all(v in (-1, 0, 1) for v in e)
        h["field_teqp_phase"] = props(e).phase in {"sparse", "dense", "mixed", "coexist"}
    except Exception:
        h["iq_embed_trits"] = False
        h["field_teqp_phase"] = False

    dense = [1, -1] * 16
    sparse = [0] * 28 + [1, -1, 1, -1]
    h["cs_dense_higher_rho"] = props(dense).rho > props(sparse).rho

    pal = Palace()
    pal.seal("night", nxt)
    pulled = pal.pull("night")
    pulled[0] = 9
    obs = pal.observe("night")
    h["palace_copy_isolated"] = pal.slots["night"].z[0] != 9
    h["palace_commit_survives_teqp"] = obs["verify"] and obs["commit_unchanged"]
    for _ in range(8):
        pal.observe("night")
    h["palace_repeat_pull_stable"] = pal.observe("night")["commit_unchanged"]

    lib = Library()
    h["catalog_named_fluids"] = set(lib.names()) >= {"NIGHT", "FIELD", "ABSMEAN", "SPARSE", "DENSE", "ZK"}
    h["catalog_props_si"] = float(lib.props_si("D", "DENSE")) > float(lib.props_si("D", "SPARSE"))
    h["catalog_still_sealed"] = all(lib.palace.observe(n.lower())["commit_unchanged"] for n in lib.names())

    return {
        "ok": all(h.values()),
        "hypotheses": h,
        "sparsity": sparsity(trits),
        "bitlinear_y": bl["y"],
        "night_props": np_.__dict__,
        "palace": {"backend": pal.backend, "observe": obs},
        "catalog": lib.names(),
    }
