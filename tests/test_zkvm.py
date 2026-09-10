from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lattice_zk.zkvm.bench import main as bench
from lattice_zk.zkvm.isa import FLIP, MAC, Instr
from lattice_zk.zkvm.prover import prove_program, verify_program


class ZkvmTests(unittest.TestCase):
    def test_sound_happy(self):
        z = [1, -1, 0, 1] * 8
        p = prove_program(z, [Instr(MAC, 0, 1), Instr(FLIP, 2)])
        self.assertTrue(verify_program(p))
        self.assertEqual(p.cycles, 2)

    def test_rejects_tamper(self):
        z = [1, 0, -1, 1] * 8
        p = prove_program(z, [Instr(MAC, 0, 1)])
        bad = prove_program(z, [Instr(FLIP, 0)])
        # swap last commit onto first proof
        p.last_c0 = bad.last_c0
        p.last_packed = bad.last_packed
        self.assertFalse(verify_program(p))

    def test_bench(self):
        out = bench(16)
        self.assertTrue(out["ok"])
        self.assertFalse(out["comparable_to_jolt_rv"])
        self.assertGreater(out["cycles"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
