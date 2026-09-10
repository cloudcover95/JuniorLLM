from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptations.kimi_k3.ondisk_bind import notes, resolve_checkpoint
from adaptations.kimi_k3.edge_loader import KimiK3EdgeLoader
from ports.ondisk import ready


class KimiK3OnDiskBindTests(unittest.TestCase):
    def test_missing_falls_to_fieldcore(self):
        with tempfile.TemporaryDirectory() as td:
            path, backend = resolve_checkpoint(Path(td))
            self.assertIsNone(path)
            self.assertEqual(backend, "JuniorBitNetFieldCore")
            n = notes(Path(td))
            self.assertFalse(n["present"])
            self.assertFalse(n["fetch"])
            self.assertEqual(n["cap_gb"], 8)
            self.assertFalse(n["full_kimi_1_5tb"])
            self.assertFalse(ready("JuniorKimiK3-edge", Path(td)))

    def test_prune_present(self):
        with tempfile.TemporaryDirectory() as td:
            hit = Path(td) / "kimi-k3-edge-pruned-8gb.mlx"
            hit.write_bytes(b"")
            path, backend = resolve_checkpoint(Path(td))
            self.assertEqual(path, str(hit))
            self.assertEqual(backend, "JuniorKimiK3-edge")
            self.assertTrue(ready("JuniorKimiK3-edge", Path(td)))

    def test_gguf_prune_present(self):
        with tempfile.TemporaryDirectory() as td:
            hit = Path(td) / "kimi-k3-edge-pruned.gguf"
            hit.write_bytes(b"")
            path, backend = resolve_checkpoint(Path(td))
            self.assertEqual(path, str(hit))
            self.assertEqual(backend, "JuniorKimiK3-edge")

    def test_gemma_fallback_when_kimi_absent(self):
        with tempfile.TemporaryDirectory() as td:
            hit = Path(td) / "gemma4-4b-q4_k_m.gguf"
            hit.write_bytes(b"")
            path, backend = resolve_checkpoint(Path(td))
            self.assertEqual(path, str(hit))
            self.assertEqual(backend, "JuniorGemma4-4B")
            n = notes(Path(td))
            self.assertFalse(n["present"])
            self.assertEqual(n["backend"], "JuniorGemma4-4B")

    def test_kimi_wins_over_gemma(self):
        with tempfile.TemporaryDirectory() as td:
            k = Path(td) / "kimi-k3-edge-pruned-8gb.mlx"
            g = Path(td) / "gemma4-4b-q4_k_m.gguf"
            k.write_bytes(b"")
            g.write_bytes(b"")
            path, backend = resolve_checkpoint(Path(td))
            self.assertEqual(path, str(k))
            self.assertEqual(backend, "JuniorKimiK3-edge")

    def test_loader_does_not_invent_weights(self):
        with tempfile.TemporaryDirectory() as td:
            loader = KimiK3EdgeLoader()
            ok = loader.load(checkpoint=str(Path(td) / "missing.mlx"))
            self.assertFalse(ok)
            self.assertFalse(loader.ready)

    def test_loader_refuses_full_1_5tb_name(self):
        with tempfile.TemporaryDirectory() as td:
            full = Path(td) / "kimi-k3-full-1.5tb.bin"
            full.write_bytes(b"")
            loader = KimiK3EdgeLoader()
            ok = loader.load(checkpoint=str(full))
            self.assertFalse(ok)
            self.assertFalse(loader.ready)


if __name__ == "__main__":
    unittest.main(verbosity=2)
