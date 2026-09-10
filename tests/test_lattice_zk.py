from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lattice_zk.bench import main as bench
from lattice_zk.mldsa_toy import keygen, sign, verify as sig_ok
from lattice_zk.ternary_arg import proof_bytes, prove, verify
from lattice_zk.toy_vm import run


class LatticeZkTests(unittest.TestCase):
    def test_arg(self):
        z = [1, -1, 0, 1] * 8
        p = prove(z)
        self.assertTrue(verify(p))
        self.assertLess(proof_bytes(p), 100)
        z2 = list(z)
        z2[0] = -1 if z2[0] != -1 else 1
        self.assertFalse(verify(prove(z2)) and prove(z2).c0 == p.c0 and prove(z2).packed == p.packed)

    def test_vm_and_sig(self):
        z = [1, 0, -1] * 10 + [0, 0]
        _zf, root, n = run(z, [("MAC", 0), ("SPARSE", 1)])
        self.assertEqual(n, 2)
        self.assertEqual(len(root), 32)
        k = keygen()
        self.assertFalse(k.secure)
        self.assertTrue(sig_ok(k, b"x", sign(k, b"x")))

    def test_bench_honest(self):
        out = bench()
        self.assertFalse(out["comparable"])
        self.assertLess(out["ours"]["proof_bytes"], out["jolt_reference_not_measured_here"]["proof_bytes_claimed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
