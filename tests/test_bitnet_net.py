from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bitnet_net.node import Node


class NetTests(unittest.TestCase):
    def test_mint_send(self):
        with tempfile.TemporaryDirectory() as td:
            n = Node(Path(td))
            n.mint("nico", 10)
            n.seal()
            n.transfer("nico", "ada", 3)
            n.seal()
            self.assertEqual(n.balances["nico"], 7)
            self.assertEqual(n.balances["ada"], 3)
            self.assertEqual(n.gossip()["format"], "junior-gossip-v1")
            n2 = Node(Path(td))
            self.assertEqual(n2.balances["ada"], 3)

    def test_insufficient(self):
        with tempfile.TemporaryDirectory() as td:
            n = Node(Path(td))
            with self.assertRaises(ValueError):
                n.transfer("ghost", "ada", 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
