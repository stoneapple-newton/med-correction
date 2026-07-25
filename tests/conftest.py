from __future__ import annotations

from pathlib import Path

import pytest

from medterm.audit import AuditStore
from medterm.config import Settings
from medterm.language import LanguageIdentifier
from medterm.matcher import MedicalTermMatcher
from medterm.terminology import DictionaryIndex, build_artifact


@pytest.fixture
def source_path() -> Path:
    return Path(__file__).parents[1] / "data" / "source_terms.csv"


@pytest.fixture
def index(tmp_path: Path, source_path: Path) -> DictionaryIndex:
    artifact_path = tmp_path / "dictionary.json"
    artifact = build_artifact(source_path, artifact_path, version="test_v1")
    return DictionaryIndex(artifact)


@pytest.fixture
def matcher(index: DictionaryIndex) -> MedicalTermMatcher:
    return MedicalTermMatcher(index, LanguageIdentifier(), suspect_threshold=0.30)


@pytest.fixture
def settings(tmp_path: Path, source_path: Path) -> Settings:
    return Settings(
        artifact_path=tmp_path / "dictionary.json",
        source_terms_path=source_path,
        audit_db_path=tmp_path / "audit.sqlite3",
        audio_dir=tmp_path / "audio",
    )


@pytest.fixture
def audit(tmp_path: Path) -> AuditStore:
    return AuditStore(tmp_path / "audit.sqlite3")
