"""
JuniorKimiK3-edge loader notes
==============================
Pruned edge install only. Never pull the original ~1.5 TB Kimi K3
weights. Resolve exclusively via ports.ondisk. Absent prune falls
back to JuniorGemma4-4B if already on disk, else FieldCore.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("juniorllm.kimi_k3")

CAP_GB = 8.0
FULL_WEIGHT_MARKERS = (
    "1.5tb",
    "1.45tb",
    "kimi-k3-full",
    "moonshot-kimi-k3",
)


@dataclass
class KimiK3EdgeConfig:
    model_id: str = "JuniorKimiK3-edge"
    quant: str = "pruned-ternary"
    filename: str = "kimi-k3-edge-pruned-8gb.mlx"
    max_download_gb: float = CAP_GB
    backend: str = "mlx"
    fetch: bool = False


class KimiK3EdgeLoader:
    """Load pruned Kimi-edge weights only if already on disk. Never downloads."""

    def __init__(self, config: Optional[KimiK3EdgeConfig] = None):
        self.config = config or KimiK3EdgeConfig()
        self._model = None
        self._is_ready = False
        self._backend_port = "JuniorBitNetFieldCore"

    def _looks_full(self, path: Path) -> bool:
        blob = path.name.lower()
        if any(m in blob for m in FULL_WEIGHT_MARKERS):
            return True
        try:
            if path.exists() and path.stat().st_size > CAP_GB * 1024 * 1024 * 1024:
                return True
        except OSError:
            return True
        return False

    def resolve(self, source_path: Optional[str] = None) -> str | None:
        from adaptations.kimi_k3.ondisk_bind import resolve_checkpoint

        if source_path:
            p = Path(source_path).expanduser()
            if self._looks_full(p):
                logger.info("Refusing full Kimi weights (no 1.5TB fetch): %s", p)
                self._backend_port = "JuniorBitNetFieldCore"
                return None
            if p.exists():
                logger.info("Kimi-edge prune on disk: %s", p)
                return str(p)
            logger.info("Requested prune missing (no fetch): %s", p)
            return None
        path, backend = resolve_checkpoint()
        self._backend_port = backend
        if path is None:
            logger.info("Kimi-edge prune absent; stay on %s", backend)
        return path

    def load(self, checkpoint: Optional[str] = None) -> bool:
        path = self.resolve(checkpoint)
        if not path:
            self._model = {
                "path": None,
                "config": self.config,
                "backend": self._backend_port,
                "present": False,
                "fetch": False,
            }
            self._is_ready = False
            return False
        self._model = {
            "path": path,
            "config": self.config,
            "backend": "mlx",
            "present": True,
            "fetch": False,
        }
        self._backend_port = "JuniorKimiK3-edge"
        self._is_ready = True
        logger.info("JuniorKimiK3-edge loader ready from %s", path)
        return True

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        if not self._is_ready:
            self.load()
        if not self._is_ready:
            return (
                f"[JuniorKimiK3-edge] Prune not on disk — "
                f"staying on {self._backend_port}. Never pull 1.5TB."
            )
        return (
            f"[JuniorKimiK3-edge pruned-ternary / MLX] "
            f"Processed prompt ({len(prompt)} chars). "
            f"Edge prune only. Cap 8GB. No fetch."
        )

    @property
    def ready(self) -> bool:
        return self._is_ready


_loader: Optional[KimiK3EdgeLoader] = None


def get_kimi_edge_loader() -> KimiK3EdgeLoader:
    global _loader
    if _loader is None:
        _loader = KimiK3EdgeLoader()
    return _loader
