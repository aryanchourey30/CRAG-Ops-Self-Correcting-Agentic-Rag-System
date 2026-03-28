"""Application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"
STORAGE_DIR = BASE_DIR / "storage"
LOG_DIR = BASE_DIR / "logs"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = "CRAG-Ops"
    api_prefix: str = "/api"
    openai_api_key: str = Field(default="")
    openai_model: str = "gpt-4.1-mini"
    chroma_path: Path = STORAGE_DIR / "chroma"
    upload_path: Path = STORAGE_DIR / "uploads"
    log_path: Path = LOG_DIR
    retrieval_top_k: int = 5
    frontend_origin: str = "http://localhost:5173"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    default_mode: Literal["pdf", "web"] = "web"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    settings = Settings()
    settings.openai_api_key = settings.openai_api_key.strip()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.log_path.mkdir(parents=True, exist_ok=True)
    (settings.log_path / "traces").mkdir(parents=True, exist_ok=True)
    return settings
