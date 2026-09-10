"""Lattice-SNARK security checklist for this repo. Not a proof of 128-bit PQ."""
from __future__ import annotations

from dataclasses import dataclass

from bitnet_pq.arg import PqProof, verify
from bitnet_pq.params import Params

CHECKS = (
    "lambda_bits==128",
    "challenge_width",
    "fiat_shamir_bind_trace",
    "air_step_closed_trits",
    "sis_endpoints",
    "secure_gate_off",
)


@dataclass
class Report:
    ok: bool
    failed: tuple[str, ...]
    checks: tuple[str, ...]
    secure_claimed: bool


def audit(proof: PqProof) -> Report:
    failed: list[str] = []
    p = proof.params
    if p.lambda_bits != 128:
        failed.append("lambda_bits==128")
    if len(proof.challenge) * 8 != p.lambda_bits:
        failed.append("challenge_width")
    if not verify(proof):
        failed.append("fiat_shamir_bind_trace")
        failed.append("air_step_closed_trits")
        failed.append("sis_endpoints")
    if p.secure:
        failed.append("secure_gate_off")
    return Report(not failed, tuple(failed), CHECKS, p.secure)
