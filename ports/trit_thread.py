"""Gaia/OSai numeric thread: pack raw vs center vs delta + SHA3."""
from __future__ import annotations

import hashlib
import time

from ports.trit_neighbor import pack_trits, trit_ham
from ports.trit_series import center, delta


def _winsor_trits(xs: list[float]) -> list[int]:
    if not xs:
        return []
    a = sorted(abs(x) for x in xs)
    tau = a[max(0, int(0.95 * (len(a) - 1)))] or 1.0
    clipped = [min(max(x, -tau), tau) for x in xs]
    g = sum(abs(x) for x in clipped) / len(clipped) or 1.0
    out = []
    for x in clipped:
        q = round(x / g)
        out.append(1 if q > 0 else (-1 if q < 0 else 0))
    return out


def thread(xs: list[float]) -> dict:
    t0 = time.perf_counter()
    raw = _winsor_trits(xs)
    cen = _winsor_trits(center(xs))
    dlt = _winsor_trits(delta(xs))
    ms = (time.perf_counter() - t0) * 1000.0
    blob = ",".join(str(x) for x in xs).encode()
    return {
        "n": len(xs),
        "raw_plus": sum(1 for t in raw if t == 1),
        "cen_plus": sum(1 for t in cen if t == 1),
        "cen_minus": sum(1 for t in cen if t == -1),
        "i2s_raw": pack_trits(raw),
        "i2s_cen": pack_trits(cen),
        "i2s_delta": pack_trits(dlt),
        "sha3_256": hashlib.sha3_256(blob).hexdigest(),
        "ms": round(ms, 4),
        "svd": False,
        "qubit": False,
    }


def ham_h0(hexes: list[str], radius: int) -> dict:
    """H0 only: components under trit/bit-hex Hamming ≤ radius. n small."""
    n = len(hexes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    from ports.trit_neighbor import bit_ham

    for i in range(n):
        for j in range(i + 1, n):
            if bit_ham(hexes[i], hexes[j]) <= radius:
                union(i, j)
    roots = {find(i) for i in range(n)}
    return {"n": n, "radius": radius, "h0": len(roots), "rips": False, "svd": False}
