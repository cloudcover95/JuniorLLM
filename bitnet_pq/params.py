"""128-bit *target* parameters. secure flag is the product gate."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    lambda_bits: int = 128
    q: int = 8380417
    n: int = 32
    m: int = 64
    rounds: int = 16
    challenge_bytes: int = 16  # 128-bit FS challenge
    secure: bool = False
    scheme: str = "junior-bitnet-pq-arg-v0"
    note: str = "parameter width 128; not a proven PQ SNARK"
