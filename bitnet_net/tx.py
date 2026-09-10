from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Tx:
    frm: str
    to: str
    amount: int
    nonce: int

    def txid(self) -> str:
        raw = f"{self.frm}:{self.to}:{self.amount}:{self.nonce}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["txid"] = self.txid()
        return d
