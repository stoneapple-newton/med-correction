from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDTERM_", env_file=".env", extra="ignore")

    artifact_path: Path = Path("data/artifacts/dictionary.json")
    source_terms_path: Path = Path("data/source_terms.csv")
    audit_db_path: Path = Path("data/audit/medterm.sqlite3")
    audio_dir: Path = Path("data/audio")
    fasttext_model_path: Path | None = None
    whisper_model: str | None = None
    review_threshold: float = 0.55
    suspect_threshold: float = 0.30
    force_review_margin: float = 0.12
    max_span_tokens: int = 5
    default_top_k: int = 5
    target_sensitivity: float = 0.95
    target_selectivity: float = 0.95


@lru_cache
def get_settings() -> Settings:
    return Settings()
