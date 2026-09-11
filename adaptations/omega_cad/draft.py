"""JuniorBitNetDraft — ternary IQ over a drawing sidecar."""
from __future__ import annotations

from dataclasses import dataclass

from adaptations.omega_cad.fill import propose
from adaptations.omega_cad.interpolate import interpolate
from adaptations.omega_cad.spatial import guess_height
from junior_bitnet.compile_sheet import compile_sheet
from ports.registry import pick


@dataclass
class DraftCall:
    port: str
    rec: str
    rigidity: float
    height: float | None
    proposals: dict[str, str]
    allow_extrude: bool
    interp_source: str = "none"
    interp_hypothesis: bool = True
    agreement: float = 0.0
    actions: list[dict] | None = None


def interpret(sidecar: str, holes: list[str] | None = None, profile: str = "", misc: str = "", fixes: list[dict] | None = None) -> DraftCall:
    port = pick("cad drawing title block", 4.0)
    gate = compile_sheet(sidecar, profile, misc, fixes)
    height = guess_height(sidecar)
    inter = interpolate(profile or sidecar, misc, sidecar)
    if height is None:
        height = inter.height if gate.data["run_iq"] or inter.source == "explicit" else None
        if height is None:
            height = gate.height
    rec, rigidity = "review_needed", 0.0
    if gate.data["run_iq"]:
        from adaptations.astra_reason.rigid_iq import run_iq

        iq = run_iq(sidecar or "empty sheet", loops=4)
        rec, rigidity = iq.recommendation, iq.rigidity
    else:
        rec = "low_confidence"
    explicit = inter.source == "explicit" or any((fx.get("field") == "height" and fx.get("by")) for fx in (fixes or []))
    allow = gate.ready and rec != "low_confidence" and explicit and not inter.hypothesis
    props = propose(holes or [], sidecar)
    if not gate.ready:
        props["compile"] = "blocking actions — do not export"
    return DraftCall(
        port.name,
        rec,
        rigidity,
        height,
        props,
        allow,
        inter.source,
        inter.hypothesis,
        inter.agreement,
        [a.__dict__ for a in gate.actions],
    )
