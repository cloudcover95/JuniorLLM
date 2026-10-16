"""Operator rail: close-hash chain for JuniorLLM / JuniorOS slices.

Close rule: close = sha256(prev || pack || text)[:16]  (hex, 16 chars).
Trit pack rate: 0.05 (the repo carries a trit pack, see docs/TRITPACK.md).

Local-first and stdlib only. No network, no model pulls. If a service ever
fronts this, it binds to 127.0.0.1 only.

Named ports (quant / OS rail mesh): Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

HOST: str = "127.0.0.1"
TRIT_RATE: float = 0.05
GENESIS: str = "0" * 16
PORT_NAMES: tuple[str, ...] = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def _b(x: str | bytes) -> bytes:
    return x if isinstance(x, bytes) else x.encode("utf-8")


def close_hash(prev: str | bytes, pack: str | bytes, text: str | bytes) -> str:
    """Return sha256(prev || pack || text)[:16] as lowercase hex."""
    return hashlib.sha256(_b(prev) + _b(pack) + _b(text)).hexdigest()[:16]


def trit_pack(values: list[float], rate: float = TRIT_RATE) -> str:
    """Pack floats to a trit string: '+' if v > rate, '-' if v < -rate, else '0'."""
    if rate < 0:
        raise ValueError("rate must be non-negative")
    return "".join("+" if v > rate else "-" if v < -rate else "0" for v in values)


@dataclass
class CloseChain:
    """Append-only chain of close hashes, one per port slice."""

    head: str = GENESIS
    links: list[tuple[str, str, str]] = field(default_factory=list)  # (port, pack, close)

    def close(self, port: str, values: list[float], text: str) -> str:
        if port not in PORT_NAMES:
            raise KeyError(f"unknown port {port!r}; expected one of {PORT_NAMES}")
        pack = trit_pack(values)
        c = close_hash(self.head, pack, text)
        self.links.append((port, pack, c))
        self.head = c
        return c

    def verify(self, texts: list[str]) -> bool:
        """Recompute the chain from GENESIS using the stored packs and given texts."""
        if len(texts) != len(self.links):
            return False
        prev = GENESIS
        for (_, pack, c), t in zip(self.links, texts):
            if close_hash(prev, pack, t) != c:
                return False
            prev = c
        return prev == self.head


def _self_check() -> None:
    assert HOST == "127.0.0.1"
    assert trit_pack([0.06, -0.06, 0.05, -0.05, 0.0]) == "+-000"
    expect = hashlib.sha256(b"abc").hexdigest()[:16]
    assert close_hash("a", "b", "c") == expect and len(expect) == 16
    ch = CloseChain()
    texts = [f"{p} slice" for p in PORT_NAMES]
    for i, (p, t) in enumerate(zip(PORT_NAMES, texts)):
        ch.close(p, [0.1 * (i - 2), 0.02, -0.3], t)
    assert ch.verify(texts)
    assert not ch.verify(texts[:-1] + ["tampered"])
    try:
        ch.close("Elsewhere", [0.0], "x")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown port accepted")


if __name__ == "__main__":
    _self_check()
    print("close_chain self-check ok; rate", TRIT_RATE, "ports", ", ".join(PORT_NAMES))
