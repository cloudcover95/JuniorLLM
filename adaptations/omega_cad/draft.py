"""JuniorBitNetDraft — ternary IQ over a drawing sidecar."""
from __future__ import annotations

from dataclasses import dataclass

from adaptations.astra_reason.rigid_iq import run_iq
from adaptations.omega_cad.fill import propose
from adaptations.omega_cad.interpolate import interpolate
from adaptations.omega_cad.spatial import guess_height
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


def interpret(sidecar: str, holes: list[str] | None = None, profile: str = "", misc: str = "") -> DraftCall:
    port = pick("cad drawing title block", 4.0)
    iq = run_iq(sidecar or "empty sheet", loops=4)
    height = guess_height(sidecar)
    inter = interpolate(profile or sidecar, misc, sidecar)
    if height is None:
        height = inter.height
    explicit = ("elev" in sidecar.lower()) or (inter.source == "explicit")
    allow = iq.recommendation != "low_confidence" and explicit and not inter.hypothesis
    props = propose(holes or [], sidecar)
    if inter.hypothesis and inter.height is not None:
        props["interp_height"] = f"{inter.height} from {inter.source}"
    return DraftCall(
        port.name,
        iq.recommendation,
        iq.rigidity,
        height,
        props,
        allow,
        inter.source,
        inter.hypothesis,
        inter.agreement,
    )
