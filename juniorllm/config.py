from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    drift_threshold: float = 0.85
    tensor_dim: int = 128
    audit_log_path: Path = Path.home() / ".juniorllm" / "audit.log"
    ollama_model: str = "llama3.1"

    class Config:
        env_prefix = "JUNIORLLM_"

settings = Settings()