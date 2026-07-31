from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from medterm.bulk import sha256_file
from medterm.medrag_batch import batch_paths, build_tts_batch, export_sqlite_batch, reconcile_batch
from medterm.terminology import load_artifact
from medterm.tts import SynthesisResult


class FakeSynthesizer:
    calls = 0

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "tts_model_id": "fake-kokoro-fixed",
            "tts_voice": "af_heart",
            "tts_language_code": "a",
            "tts_device": "cpu",
            "tts_speed": 1.0,
            "tts_sample_rate": 24_000,
        }

    def synthesize(
        self, text: str, output_path: Path, *, phonemes: str | None = None
    ) -> SynthesisResult:
        self.calls += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF-" + text.encode())
        return SynthesisResult(output_path, 24_000, phonemes)


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE source (source_id TEXT PRIMARY KEY);
        CREATE TABLE source_release (
          release_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, version TEXT NOT NULL
        );
        CREATE TABLE concept (
          concept_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL,
          review_status TEXT NOT NULL, status TEXT NOT NULL
        );
        CREATE TABLE term (
          term_id INTEGER PRIMARY KEY, concept_id TEXT NOT NULL,
          term TEXT NOT NULL, normalized_term TEXT NOT NULL,
          language_tag TEXT NOT NULL, source_release_id TEXT NOT NULL,
          preferred INTEGER NOT NULL, status TEXT NOT NULL
        );
        INSERT INTO source VALUES ('mesh');
        INSERT INTO source_release VALUES ('mesh-2026', 'mesh', '2026');
        INSERT INTO concept VALUES ('mesh:2', 'substance', 'unreviewed', 'active');
        INSERT INTO concept VALUES ('mesh:1', 'biomedical_concept', 'clinical-reviewed', 'active');
        INSERT INTO term VALUES (1, 'mesh:2', 'Zeta', 'zeta', 'en', 'mesh-2026', 1, 'active');
        INSERT INTO term VALUES (2, 'mesh:1', 'Alpha', 'alpha', 'en', 'mesh-2026', 1, 'active');
        INSERT INTO term VALUES (3, 'mesh:1', 'Alpha alias', 'alpha alias', 'en', 'mesh-2026', 0, 'active');
        """
    )
    connection.commit()
    connection.close()


def test_export_is_deterministic_auditable_and_batch_id_is_immutable(tmp_path: Path) -> None:
    database = tmp_path / "terms.sqlite"
    _database(database)
    checksum = sha256_file(database)
    build_root = tmp_path / "build"

    manifest, artifact = export_sqlite_batch(
        database,
        build_root,
        release_id="local-v1",
        batch_id="batch-000001",
        source_language="en",
        offset=0,
        limit=2,
        expected_sqlite_sha256=checksum,
    )
    paths = batch_paths(build_root, "en", "batch-000001")
    source = json.loads(paths["source"].read_text(encoding="utf-8"))

    assert [item["concept_id"] for item in source] == ["mesh:1", "mesh:2"]
    assert source[0]["aliases"] == ["Alpha alias"]
    assert source[0]["concept_type"] == "biomedical_concept"
    assert source[0]["risk_tier"] == "unclassified"
    assert source[0]["lasa_status"] == "unclassified"
    assert source[0]["provenance_detail"]["source_language"] == "en"
    assert manifest.auto_commit_enabled is False
    assert len(artifact.entries) == 2
    assert load_artifact(paths["dictionary"]).artifact_version == "local-v1-en-batch-000001"

    resumed, _ = export_sqlite_batch(
        database,
        build_root,
        release_id="local-v1",
        batch_id="batch-000001",
        source_language="en",
        offset=0,
        limit=2,
        expected_sqlite_sha256=checksum,
    )
    assert resumed.source_export_sha256 == manifest.source_export_sha256

    with pytest.raises(ValueError, match="immutable"):
        export_sqlite_batch(
            database,
            build_root,
            release_id="local-v1",
            batch_id="batch-000001",
            source_language="en",
            offset=1,
            limit=1,
            expected_sqlite_sha256=checksum,
        )


def test_export_stops_on_registered_database_checksum_change(tmp_path: Path) -> None:
    database = tmp_path / "terms.sqlite"
    _database(database)

    with pytest.raises(ValueError, match="register a new release"):
        export_sqlite_batch(
            database,
            tmp_path / "build",
            release_id="local-v1",
            batch_id="batch-000001",
            source_language="en",
            offset=0,
            limit=1,
            expected_sqlite_sha256="0" * 64,
        )


def test_tts_resume_validates_audio_and_reconciliation(tmp_path: Path) -> None:
    database = tmp_path / "terms.sqlite"
    _database(database)
    build_root = tmp_path / "build"
    export_sqlite_batch(
        database,
        build_root,
        release_id="local-v1",
        batch_id="batch-000001",
        source_language="en",
        offset=0,
        limit=2,
        expected_sqlite_sha256=sha256_file(database),
    )
    paths = batch_paths(build_root, "en", "batch-000001")
    synthesizer = FakeSynthesizer()
    checkpoint = build_tts_batch(
        paths["dictionary"],
        paths["audio"],
        paths["references"],
        paths["failures"],
        paths["tts_checkpoint"],
        synthesizer,
        language="en",
        checkpoint_every=1,
    )
    calls = synthesizer.calls
    resumed = build_tts_batch(
        paths["dictionary"],
        paths["audio"],
        paths["references"],
        paths["failures"],
        paths["tts_checkpoint"],
        synthesizer,
        language="en",
        checkpoint_every=1,
    )
    reconciliation = reconcile_batch(
        paths["references"], paths["failures"], paths["tts_checkpoint"]
    )

    assert checkpoint.status == "complete"
    assert resumed.active_references == 2
    assert synthesizer.calls == calls
    assert reconciliation["unexplained_inventory_difference"] == 0
    assert reconciliation["auto_commit_enabled"] is False


def test_tts_resume_regenerates_a_committed_checksum_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "terms.sqlite"
    _database(database)
    build_root = tmp_path / "build"
    export_sqlite_batch(
        database,
        build_root,
        release_id="local-v1",
        batch_id="batch-000001",
        source_language="en",
        offset=0,
        limit=1,
        expected_sqlite_sha256=sha256_file(database),
    )
    paths = batch_paths(build_root, "en", "batch-000001")
    synthesizer = FakeSynthesizer()
    build_tts_batch(
        paths["dictionary"],
        paths["audio"],
        paths["references"],
        paths["failures"],
        paths["tts_checkpoint"],
        synthesizer,
        language="en",
    )
    manifest_record = json.loads(paths["references"].read_text(encoding="utf-8"))
    audio_path = paths["references"].parent / manifest_record["audio_path"]
    audio_path.write_bytes(b"corrupt")
    calls = synthesizer.calls

    build_tts_batch(
        paths["dictionary"],
        paths["audio"],
        paths["references"],
        paths["failures"],
        paths["tts_checkpoint"],
        synthesizer,
        language="en",
    )

    assert synthesizer.calls == calls + 1
    assert audio_path.read_bytes().startswith(b"RIFF-")
