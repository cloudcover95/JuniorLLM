"""2-bit trit pack + trit Hamming vs bit Hamming."""
from __future__ import annotations

# t+1: -1→00, 0→01, 1→10. 11 unused.


def pack_trits(trits: list[int]) -> str:
    bits = 0
    for t in trits:
        bits = (bits << 2) | (max(-1, min(1, t)) + 1)
    return format(bits, "x")


def trit_ham(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))


def bit_ham(ha: str, hb: str) -> int:
    def bits(h: str) -> str:
        try:
            raw = bytes.fromhex(h)
        except ValueError:
            raw = (h or "").encode()
        return "".join(f"{b:08b}" for b in raw)

    x, y = bits(ha), bits(hb)
    n = min(len(x), len(y))
    return sum(p != q for p, q in zip(x[:n], y[:n])) + abs(len(x) - len(y))


def demo() -> dict:
    a = [1, 1, 0, -1]
    b = [1, -1, 0, -1]  # one sign flip
    ha, hb = pack_trits(a), pack_trits(b)
    return {
        "a": a,
        "b": b,
        "i2s_a": ha,
        "i2s_b": hb,
        "trit_ham": trit_ham(a, b),
        "bit_ham": bit_ham(ha, hb),
        "unused_pattern": "11",
        "collision_resistant": False,
    }
