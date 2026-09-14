"""1.58 internals. Alphabet, sparsity, L1 vs absmean. Stdlib."""
from __future__ import annotations

from junior_bitnet.math import absmean
from junior_bitnet.trit_bench import run as bench
from junior_bitnet.winsor import pack


def research(n: int = 256) -> dict:
    b = bench(n)
    xs = [0.01 * i - 1.2 for i in range(n)]
    q = pack(xs, 95.0)
    am, g = absmean(xs)
    alphabet = set(q["trit"]) <= {-1, 0, 1}
    return {
        "wire": "winsor-p95",
        "alphabet_ok": alphabet,
        "sparsity": q["sparsity"],
        "gamma": q["gamma"],
        "absmean_gamma": g,
        "bench": b,
        "terraform": "flagstaff vote, not hashicorp",
        "mlx": False,
    }
