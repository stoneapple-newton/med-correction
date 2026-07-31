from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from medterm.audio_embeddings import ReferencePronunciation, load_reference_manifest
from medterm.bulk import atomic_write_text, canonical_hash, canonical_json, sha256_file
from medterm.terminology import DictionaryArtifact, SourceTerm, build_artifact, load_artifact
from medterm.tts import TermSynthesizer, build_synthetic_references, synthetic_reference_identity

SELECTION_ORDER = ["concept_id", "normalized_term", "term_id"]
LICENSE_STATE = "source_catalog_open_local_processing_unreviewed"
LANGUAGE_MAP = {"en": "en", "zh-Hans": "zh"}

SELECTION_SQL = """
SELECT
  t.term_id,
  t.concept_id,
  t.term,
  t.language_tag,
  t.source_release_id,
  c.entity_type,
  c.review_status,
  s.source_id,
  r.version
FROM term AS t
JOIN concept AS c ON c.concept_id = t.concept_id
JOIN source_release AS r ON r.release_id = t.source_release_id
JOIN source AS s ON s.source_id = r.source_id
WHERE t.preferred = 1
  AND t.status = 'active'
  AND c.status = 'active'
  AND t.language_tag = ?
ORDER BY t.concept_id, t.normalized_term, t.term_id
LIMIT ? OFFSET ?
"""

ALIAS_SQL = """
SELECT term
FROM term
WHERE concept_id = ?
  AND language_tag = ?
  AND term <> ?
  AND status = 'active'
ORDER BY preferred DESC, normalized_term, term_id
"""


class MedragSelectionManifest(BaseModel):
    schema_version: int = 1
    release_id: str
    batch_id: str
    sqlite_path: str
    sqlite_sha256: str
    source_export_path: str
    source_export_sha256: str
    source_language: str
    artifact_language: str
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)
    selected_count: int = Field(ge=0)
    ordering_fields: list[str]
    selection_stage_hash: str
    build_classification: str = "development_local_only"
    licensing_status: str = LICENSE_STATE
    distribution_approved: bool = False
    decision: str = "review"
    auto_commit_enabled: bool = False
    created_at: datetime


class TtsFailure(BaseModel):
    concept_id: str
    term: str
    exception_class: str
    reason: str


class TtsCheckpoint(BaseModel):
    schema_version: int = 1
    stage_hash: str
    artifact_sha256: str
    planned_references: int
    active_references: int = 0
    terminal_failures: int = 0
    processed_references: int = 0
    status: str = "running"
    tts_provenance: dict[str, Any]
    decision: str = "review"
    auto_commit_enabled: bool = False
    updated_at: datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def batch_paths(build_root: Path, language: str, batch_id: str) -> dict[str, Path]:
    terminology = build_root / "terminology" / language
    tts = build_root / "tts" / language
    vectors = build_root / "vectors" / language / batch_id
    return {
        "selection": terminology / f"{batch_id}.selection.json",
        "source": terminology / f"{batch_id}.source.json",
        "dictionary": terminology / f"{batch_id}.dictionary.json",
        "references": tts / f"{batch_id}.active_references.jsonl",
        "failures": tts / f"{batch_id}.failures.jsonl",
        "tts_checkpoint": tts / f"{batch_id}.checkpoint.json",
        "audio": tts / f"{batch_id}.audio",
        "vectors": vectors,
        "index": build_root / "indexes" / f"kokoro_vectors_{language}_{batch_id}.npz",
        "reconciliation": (
            build_root
            / "indexes"
            / f"kokoro_vectors_{language}_{batch_id}.reconciliation.json"
        ),
    }


def export_sqlite_batch(
    sqlite_path: Path,
    build_root: Path,
    *,
    release_id: str,
    batch_id: str,
    source_language: str,
    offset: int,
    limit: int,
    expected_sqlite_sha256: str,
) -> tuple[MedragSelectionManifest, DictionaryArtifact]:
    if source_language not in LANGUAGE_MAP:
        raise ValueError(f"Unsupported release language: {source_language!r}.")
    if offset < 0 or limit < 1:
        raise ValueError("offset must be non-negative and limit must be positive.")
    sqlite_path = sqlite_path.resolve()
    actual_hash = sha256_file(sqlite_path)
    expected_hash = expected_sqlite_sha256.casefold()
    if actual_hash != expected_hash:
        raise ValueError(
            "SQLite checksum differs from the registered release; register a new release identity."
        )
    artifact_language = LANGUAGE_MAP[source_language]
    paths = batch_paths(build_root, artifact_language, batch_id)
    selection_inputs = {
        "schema_version": 1,
        "release_id": release_id,
        "batch_id": batch_id,
        "sqlite_sha256": actual_hash,
        "source_language": source_language,
        "artifact_language": artifact_language,
        "offset": offset,
        "limit": limit,
        "ordering_fields": SELECTION_ORDER,
        "licensing_status": LICENSE_STATE,
        "auto_commit_enabled": False,
    }
    stage_hash = canonical_hash(selection_inputs)
    if paths["selection"].is_file():
        existing = MedragSelectionManifest.model_validate_json(
            paths["selection"].read_text(encoding="utf-8")
        )
        if existing.selection_stage_hash != stage_hash:
            raise ValueError("Existing batch ID has different immutable selection settings.")
        if (
            not paths["source"].is_file()
            or sha256_file(paths["source"]) != existing.source_export_sha256
        ):
            raise ValueError("Existing source export is missing or has a checksum mismatch.")
        artifact = _load_or_build_dictionary(
            paths["source"], paths["dictionary"], f"{release_id}-{artifact_language}-{batch_id}"
        )
        return existing, artifact

    uri = f"file:{sqlite_path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(SELECTION_SQL, (source_language, limit, offset)).fetchall()
        source_terms = []
        for ordinal, row in enumerate(rows, start=offset + 1):
            (
                term_id,
                concept_id,
                term,
                language_tag,
                source_release_id,
                entity_type,
                review_status,
                source_id,
                release_version,
            ) = row
            aliases = [
                item[0]
                for item in connection.execute(
                    ALIAS_SQL, (concept_id, language_tag, term)
                ).fetchall()
            ]
            source_terms.append(
                SourceTerm(
                    concept_id=concept_id,
                    term=term,
                    aliases=list(dict.fromkeys(aliases)),
                    language=artifact_language,
                    risk_tier="unclassified",
                    provenance=source_id,
                    source_vocabulary=source_id,
                    source_release=release_version,
                    licensing_status=LICENSE_STATE,
                    review_status=review_status,
                    concept_type=entity_type,
                    source_record_type="preferred",
                    lasa_status="unclassified",
                    provenance_detail={
                        "source_language": language_tag,
                        "source_release_id": source_release_id,
                        "source_term_id": term_id,
                        "source_sqlite_sha256": actual_hash,
                        "selection_ordinal": ordinal,
                    },
                )
            )
    finally:
        connection.close()
    if not source_terms:
        raise ValueError("The requested deterministic SQLite slice is empty.")

    source_payload = [item.model_dump(mode="json") for item in source_terms]
    atomic_write_text(
        paths["source"], json.dumps(source_payload, indent=2, ensure_ascii=False) + "\n"
    )
    source_hash = sha256_file(paths["source"])
    manifest = MedragSelectionManifest(
        release_id=release_id,
        batch_id=batch_id,
        sqlite_path=str(sqlite_path),
        sqlite_sha256=actual_hash,
        source_export_path=str(paths["source"]),
        source_export_sha256=source_hash,
        source_language=source_language,
        artifact_language=artifact_language,
        offset=offset,
        limit=limit,
        selected_count=len(source_terms),
        ordering_fields=SELECTION_ORDER,
        selection_stage_hash=stage_hash,
        created_at=utc_now(),
    )
    artifact = _load_or_build_dictionary(
        paths["source"], paths["dictionary"], f"{release_id}-{artifact_language}-{batch_id}"
    )
    atomic_write_text(paths["selection"], manifest.model_dump_json(indent=2) + "\n")
    return manifest, artifact


def _load_or_build_dictionary(
    source_path: Path, dictionary_path: Path, version: str
) -> DictionaryArtifact:
    source_hash = sha256_file(source_path)
    if dictionary_path.is_file():
        artifact = load_artifact(dictionary_path)
        if artifact.source_sha256 != source_hash or artifact.artifact_version != version:
            raise ValueError("Existing dictionary does not match the immutable batch source.")
        return artifact
    temporary = dictionary_path.with_name(f".{dictionary_path.stem}.{os.getpid()}.tmp.json")
    try:
        artifact = build_artifact(source_path, temporary, version)
        temporary.replace(dictionary_path)
        return artifact
    finally:
        temporary.unlink(missing_ok=True)


def build_tts_batch(
    artifact_path: Path,
    audio_dir: Path,
    manifest_path: Path,
    failures_path: Path,
    checkpoint_path: Path,
    synthesizer: TermSynthesizer,
    *,
    language: str,
    checkpoint_every: int = 500,
) -> TtsCheckpoint:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive.")
    artifact = load_artifact(artifact_path)
    entries = [item for item in artifact.entries if item.language == language]
    if not entries:
        raise ValueError(f"Dictionary has no {language!r} entries.")
    artifact_hash = sha256_file(artifact_path)
    stage_hash = canonical_hash(
        {
            "schema_version": 1,
            "artifact_sha256": artifact_hash,
            "language": language,
            "tts": synthesizer.provenance,
        }
    )
    if checkpoint_path.is_file():
        prior_checkpoint = TtsCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if prior_checkpoint.stage_hash != stage_hash:
            raise ValueError("Existing TTS checkpoint has a different stage hash.")

    existing, invalid = _validated_existing_references(
        manifest_path, artifact, synthesizer, language
    )
    failures: list[TtsFailure] = []
    references: list[ReferencePronunciation] = []
    for index, entry in enumerate(entries, start=1):
        expected_id = _expected_reference_id(artifact, entry, synthesizer, language)
        reference = existing.get(expected_id)
        if reference is not None:
            references.append(reference)
        else:
            one_entry = artifact.model_copy(update={"entries": [entry]})
            try:
                references.extend(
                    build_synthetic_references(
                        one_entry,
                        synthesizer,
                        audio_dir,
                        language=language,
                        overwrite=expected_id in invalid,
                    )
                )
            except Exception as exc:  # isolated terminal term failure is persisted
                failures.append(
                    TtsFailure(
                        concept_id=entry.concept_id,
                        term=entry.term,
                        exception_class=type(exc).__name__,
                        reason=str(exc).replace("\r", " ").replace("\n", " ")[:500],
                    )
                )
        if index % checkpoint_every == 0:
            _write_tts_state(
                references,
                failures,
                manifest_path,
                failures_path,
                checkpoint_path,
                stage_hash,
                artifact_hash,
                len(entries),
                synthesizer.provenance,
                "running",
            )
    status = "complete" if not failures else "complete_with_failures"
    return _write_tts_state(
        references,
        failures,
        manifest_path,
        failures_path,
        checkpoint_path,
        stage_hash,
        artifact_hash,
        len(entries),
        synthesizer.provenance,
        status,
    )


def _expected_reference_id(
    artifact: DictionaryArtifact, entry: Any, synthesizer: TermSynthesizer, language: str
) -> str:
    provenance = synthesizer.provenance
    reference_id, _ = synthetic_reference_identity(
        concept_id=entry.concept_id,
        term=entry.term,
        language=language,
        curated_phonemes=entry.pronunciations[0] if entry.pronunciations else None,
        terminology_release=artifact.artifact_version,
        tts_model_revision=str(
            provenance.get("tts_model_revision")
            or provenance.get("tts_model_id")
            or "unresolved"
        ),
        g2p_revision=str(provenance.get("g2p_revision") or "kokoro-bundled-unresolved"),
        voice=str(provenance.get("tts_voice") or "unresolved"),
        speed=float(provenance.get("tts_speed") or 1.0),
        sample_rate=int(provenance.get("tts_sample_rate") or 24_000),
    )
    return reference_id


def _validated_existing_references(
    manifest_path: Path,
    artifact: DictionaryArtifact,
    synthesizer: TermSynthesizer,
    language: str,
) -> tuple[dict[str, ReferencePronunciation], set[str]]:
    if not manifest_path.is_file():
        return {}, set()
    expected = {
        _expected_reference_id(artifact, entry, synthesizer, language): entry
        for entry in artifact.entries
        if entry.language == language
    }
    valid: dict[str, ReferencePronunciation] = {}
    invalid: set[str] = set()
    for reference in load_reference_manifest(manifest_path):
        entry = expected.get(reference.reference_id)
        if (
            entry is None
            or reference.concept_id != entry.concept_id
            or reference.term != entry.term
        ):
            continue
        if not reference.audio_path.is_file():
            invalid.add(reference.reference_id)
            continue
        checksum = sha256_file(reference.audio_path)
        if checksum != reference.audio_sha256 or checksum != reference.rendition_id:
            invalid.add(reference.reference_id)
            continue
        if reference.reference_id in valid:
            raise ValueError("Existing TTS manifest contains duplicate reference IDs.")
        valid[reference.reference_id] = reference
    return valid, invalid


def _write_reference_manifest_atomic(
    references: list[ReferencePronunciation], manifest_path: Path
) -> None:
    lines = []
    for reference in references:
        payload = reference.model_dump(mode="json")
        try:
            payload["audio_path"] = Path(
                os.path.relpath(reference.audio_path, manifest_path.parent)
            ).as_posix()
        except ValueError:
            payload["audio_path"] = reference.audio_path.as_posix()
        lines.append(canonical_json(payload))
    atomic_write_text(manifest_path, "".join(f"{line}\n" for line in lines))


def _write_tts_state(
    references: list[ReferencePronunciation],
    failures: list[TtsFailure],
    manifest_path: Path,
    failures_path: Path,
    checkpoint_path: Path,
    stage_hash: str,
    artifact_hash: str,
    planned: int,
    provenance: dict[str, Any],
    status: str,
) -> TtsCheckpoint:
    if len({item.reference_id for item in references}) != len(references):
        raise ValueError("Generated TTS references contain duplicate reference IDs.")
    _write_reference_manifest_atomic(references, manifest_path)
    atomic_write_text(
        failures_path,
        "".join(f"{canonical_json(item.model_dump(mode='json'))}\n" for item in failures),
    )
    checkpoint = TtsCheckpoint(
        stage_hash=stage_hash,
        artifact_sha256=artifact_hash,
        planned_references=planned,
        active_references=len(references),
        terminal_failures=len(failures),
        processed_references=len(references) + len(failures),
        status=status,
        tts_provenance=provenance,
        updated_at=utc_now(),
    )
    if checkpoint.processed_references > planned or (
        status != "running" and checkpoint.processed_references != planned
    ):
        raise ValueError("TTS inventory does not reconcile with the planned references.")
    atomic_write_text(checkpoint_path, checkpoint.model_dump_json(indent=2) + "\n")
    return checkpoint


def reconcile_batch(
    manifest_path: Path,
    failures_path: Path,
    checkpoint_path: Path,
    index_summary_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint = TtsCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    references = load_reference_manifest(manifest_path)
    failures = [
        TtsFailure.model_validate_json(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference_ids: set[str] = set()
    for reference in references:
        if reference.reference_id in reference_ids:
            raise ValueError("Duplicate logical reference ID during reconciliation.")
        reference_ids.add(reference.reference_id)
        if not reference.audio_path.is_file():
            raise ValueError(f"Missing reference audio: {reference.reference_id}")
        checksum = sha256_file(reference.audio_path)
        if checksum != reference.audio_sha256 or checksum != reference.rendition_id:
            raise ValueError(f"Reference audio checksum mismatch: {reference.reference_id}")
    planned_difference = checkpoint.planned_references - len(references) - len(failures)
    indexed_rows = None
    vector_difference = None
    index_sha256 = None
    if index_summary_path is not None:
        summary = json.loads(index_summary_path.read_text(encoding="utf-8"))
        indexed_rows = int(summary["row_count"])
        vector_difference = len(references) - indexed_rows - int(summary["failed_renditions"])
        index_path = Path(summary["index_path"])
        if not index_path.is_absolute() and not index_path.is_file():
            index_path = index_summary_path.parent / index_path.name
        index_sha256 = sha256_file(index_path)
        if index_sha256 != summary["index_sha256"]:
            raise ValueError("Finalized index checksum mismatch.")
    result = {
        "schema_version": 1,
        "planned_references": checkpoint.planned_references,
        "active_references": len(references),
        "terminal_tts_failures": len(failures),
        "indexed_rows": indexed_rows,
        "explicit_vector_failures": 0 if indexed_rows is not None else None,
        "planned_inventory_difference": planned_difference,
        "vector_inventory_difference": vector_difference,
        "unexplained_inventory_difference": planned_difference
        + (vector_difference or 0),
        "index_sha256": index_sha256,
        "decision": "review",
        "auto_commit_enabled": False,
    }
    if result["unexplained_inventory_difference"] != 0:
        raise ValueError("Batch reconciliation has an unexplained inventory difference.")
    return result
