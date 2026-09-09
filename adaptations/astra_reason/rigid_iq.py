"""Rigid IQ layers — looped BitNet ternary. Same weights, more passes.

More loops raise rigidity (disagreement must survive repeated neighbor votes).
This is the local stand-in for 'Astra-class' depth without extra parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

from bitnet_night.dream_mesh import DIM, drift, tick


def _clamp3(v: int) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def embed(text: str, dim: int = DIM) -> list[int]:
    from agent.guardrails import hash_text

    h = hash_text(text or "")
    out = []
    for ch in h:
        v = int(ch, 16)
        out.append(-1 if v < 5 else (0 if v < 10 else 1))
    return (out * (dim // max(1, len(out)) + 1))[:dim]


def iq_pass(state: list[int], loop_i: int) -> list[int]:
    """One rigid layer: neighbor vote, then force-sparsity on a rotating lane."""
    stepped = tick(state, loop_i)
    # rigidity: dimensions that flipped twice in spirit get clamped toward 0
    rigid = []
    for i, (a, b) in enumerate(zip(state, stepped)):
        if a != b and (i + loop_i) % 2 == 0:
            rigid.append(0)  # refuse noisy flip
        else:
            rigid.append(b)
    return [_clamp3(x) for x in rigid]


@dataclass
class IQTrace:
    loops: int
    start_drift: float
    end_drift: float
    rigidity: float  # 1 - mean |delta| across loops
    state: list[int]
    recommendation: str


def run_iq(text: str, loops: int = 4) -> IQTrace:
    loops = max(1, min(int(loops), 16))
    s0 = embed(text)
    s = list(s0)
    acc = 0.0
    for i in range(loops):
        nxt = iq_pass(s, i)
        acc += drift(s, nxt)
        s = nxt
    mean_d = acc / loops
    rigidity = max(0.0, min(1.0, 1.0 - mean_d))
    end_d = drift(s0, s)
    if rigidity >= 0.78 and end_d < 0.35:
        rec = "high_confidence"
    elif rigidity >= 0.55:
        rec = "review_needed"
    else:
        rec = "low_confidence"
    return IQTrace(
        loops=loops,
        start_drift=round(drift(s0, s0), 4),
        end_drift=round(end_d, 4),
        rigidity=round(rigidity, 4),
        state=s,
        recommendation=rec,
    )
