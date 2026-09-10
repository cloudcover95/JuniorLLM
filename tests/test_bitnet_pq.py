from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bitnet_pq.arg import prove, verify
from bitnet_pq.chain import Chain
from bitnet_pq.params import Params
from bitnet_pq.scripts.cli import bench, tick
from lattice_zk.zkvm.isa import MAC, Instr


class PqTests(unittest.TestCase):
    def test_params_128(self):
        p = Params()
        self.assertEqual(p.lambda_bits, 128)
        self.assertFalse(p.secure)
        self.assertEqual(p.challenge_bytes * 8, 128)

    def test_arg(self):
        pr = prove([1, 0, -1, 1] * 8, [Instr(MAC, 1, 2)])
        self.assertTrue(verify(pr))
        self.assertEqual(len(pr.challenge) * 8, 128)

    def test_chain(self):
        out = tick(2)
        self.assertGreaterEqual(out["blocks"], 3)
        self.assertFalse(out["tip"]["secure"])

    def test_bench(self):
        b = bench()
        self.assertTrue(b["ok"])
        self.assertEqual(b["lambda_bits"], 128)
        self.assertFalse(b["secure"])

    def test_secure_gate(self):
        c = Chain(Params(secure=True))
        c.genesis("x", [1, 0, -1, 1] * 8)
        with self.assertRaises(RuntimeError):
            c.tick("x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
