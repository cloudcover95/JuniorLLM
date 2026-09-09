from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptations.gemma4.ondisk_bind import notes, resolve_checkpoint
from adaptations.gemma4.bitnet_mlx_loader import Gemma4BitNetLoader


class GemmaOnDiskBindTests(unittest.TestCase):
    def test_missing_falls_to_fieldcore(self):
        with tempfile.TemporaryDirectory() as td:
            path, backend = resolve_checkpoint(Path(td))
            self.assertIsNone(path)
            self.assertEqual(backend, "JuniorBitNetFieldCore")
            n = notes(Path(td))
            self.assertFalse(n["present"])
            self.assertFalse(n["fetch"])

    def test_gguf_present(self):
        with tempfile.TemporaryDirectory() as td:
            hit = Path(td) / "gemma4-4b-q4_k_m.gguf"
            hit.write_bytes(b"")
            path, backend = resolve_checkpoint(Path(td))
            self.assertEqual(path, str(hit))
            self.assertEqual(backend, "JuniorGemma4-4B")

    def test_mlx_present(self):
        with tempfile.TemporaryDirectory() as td:
            hit = Path(td) / "gemma4-4b-bitnet-1.58.mlx"
            hit.write_bytes(b"")
            path, backend = resolve_checkpoint(Path(td))
            self.assertEqual(path, str(hit))
            self.assertEqual(backend, "JuniorGemma4-4B")

    def test_loader_does_not_invent_weights(self):
        with tempfile.TemporaryDirectory() as td:
            loader = Gemma4BitNetLoader()
            ok = loader.load(checkpoint=str(Path(td) / "missing.gguf"))
            self.assertFalse(ok)
            self.assertFalse(loader.ready)


if __name__ == "__main__":
    unittest.main(verbosity=2)
