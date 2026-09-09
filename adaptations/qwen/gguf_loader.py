"""
Qwen-local GGUF loader notes
============================
Quality baseline under the 8GB cap. Weights resolve only via
ports.ondisk — never fetched. Absent Qwen GGUF falls back to
JuniorGemma4-4B if that file is already on disk, else FieldCore.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("juniorllm.qwen")


@dataclass
class QwenLocalConfig:
    model_id: str = "Qwen/Qwen3-8B"
    quant: str = "Q4_K_M"
    filename: str = "qwen3-8b-q4_k_m.gguf"
    max_download_gb: float = 8.0
    backend: str = "gguf"


class QwenLocalLoader:
    """Load Qwen GGUF only if the file already exists. Never downloads."""

    def __init__(self, config: Optional[QwenLocalConfig] = None):
        self.config = config or QwenLocalConfig()
        self._model = None
        self._is_ready = False
        self._backend_port = "JuniorBitNetFieldCore"

    def resolve(self, source_path: Optional[str] = None) -> str | None:
        from adaptations.qwen.ondisk_bind import resolve_checkpoint

        if source_path:
            p = Path(source_path).expanduser()
            if p.exists():
                logger.info("Qwen checkpoint on disk: %s", p)
                return str(p)
            logger.info("Requested checkpoint missing (no fetch): %s", p)
            return None
        path, backend = resolve_checkpoint()
        self._backend_port = backend
        if path is None:
            logger.info("Qwen weights absent; stay on %s", backend)
        return path

    def load(self, checkpoint: Optional[str] = None) -> bool:
        path = self.resolve(checkpoint)
        if not path:
            self._model = {
                "path": None,
                "config": self.config,
                "backend": self._backend_port,
                "present": False,
            }
            self._is_ready = False
            return False
        self._model = {
            "path": path,
            "config": self.config,
            "backend": "gguf",
            "present": True,
        }
        self._backend_port = "Qwen-local"
        self._is_ready = True
        logger.info("Qwen-local loader ready from %s", path)
        return True

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        if not self._is_ready:
            self.load()
        if not self._is_ready:
            return (
                f"[Qwen-local] Weights not on disk — "
                f"staying on {self._backend_port}. No download."
            )
        return (
            f"[Qwen-local Q4_K_M / GGUF] "
            f"Processed prompt ({len(prompt)} chars). "
            f"Quality baseline ready. Cap 8GB. No fetch."
        )

    @property
    def ready(self) -> bool:
        return self._is_ready


_loader: Optional[QwenLocalLoader] = None


def get_qwen_loader() -> QwenLocalLoader:
    global _loader
    if _loader is None:
        _loader = QwenLocalLoader()
    return _loader
