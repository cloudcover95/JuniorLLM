"""JuniorBitNetDraft — ternary IQ over a drawing sidecar."""
from __future__ import annotations

from dataclasses import dataclass

from adaptations.astra_reason.rigid_iq import run_iq
from adaptations.omega_cad.fill import propose
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


def interpret(sidecar: str, holes: list[str] | None = None) -> DraftCall:
    port = pick("cad drawing title block", 4.0)
    iq = run_iq(sidecar or "empty sheet", loops=4)
    height = guess_height(sidecar)
    props = propose(holes or [], sidecar)
    allow = iq.recommendation != "low_confidence" and ("elev" in sidecar.lower() or height is not None)
    return DraftCall(port.name, iq.recommendation, iq.rigidity, height, props, allow)
