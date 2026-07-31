from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from medterm.normalization import normalize_term
from medterm.pronunciation import PronunciationEngine

RXNCONSO_FIELDS = (
    "RXCUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "RXAUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
    "EMPTY",
)
RXNORM_TERM_TYPES = frozenset({"IN", "PIN", "MIN", "SCD", "SBD", "BN", "PSN", "SY"})
ALLOWED_LICENSE_STATES = frozenset(
    {"approved_for_local_processing", "approved_for_artifact_distribution"}
)
IMPORT_SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(UTC)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SourceManifest(BaseModel):
    schema_version: int = 1
    release_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_type: Literal["rxnorm-rrf"]
    source_release: str = Field(min_length=1)
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_status: str
    license_approval_id: str = Field(min_length=1)
    license_owner: str = Field(min_length=1)
    license_evidence: str = Field(min_length=1)
    license_approval_date: str = Field(min_length=1)
    required_languages: list[str] = Field(default_factory=lambda: ["en"])
    created_at: datetime
    contains_patient_data: bool = False

    @field_validator("license_status")
    @classmethod
    def validate_license(cls, value: str) -> str:
        if value not in ALLOWED_LICENSE_STATES:
            raise ValueError("Source license is not approved for local processing.")
        return value

    @field_validator("required_languages")
    @classmethod
    def validate_languages(cls, value: list[str]) -> list[str]:
        normalized = sorted({item.strip().casefold() for item in value if item.strip()})
        if normalized != ["en"]:
            raise ValueError("The initial RxNorm importer supports required_languages=['en'] only.")
        return normalized


class ImportCheckpoint(BaseModel):
    schema_version: int = IMPORT_SCHEMA_VERSION
    release_id: str
    stage_hash: str
    source_sha256: str
    source_rows_processed: int = 0
    status: Literal["running", "parsed", "complete"] = "running"
    attempt: int = 1
    updated_at: datetime
    counts: dict[str, int] = Field(
        default_factory=lambda: {
            "read": 0,
            "accepted_rows": 0,
            "rejected": 0,
            "suppressed": 0,
            "filtered": 0,
            "duplicate": 0,
            "emitted_records": 0,
        }
    )
    history: list[dict[str, Any]] = Field(default_factory=list)


class ImportSummary(BaseModel):
    schema_version: int = IMPORT_SCHEMA_VERSION
    release_id: str
    stage_hash: str
    source_sha256: str
    partial: bool
    max_concepts: int | None
    counts: dict[str, int]
    shard_checksums: dict[str, str]
    staging_size_bytes: int
    completed_at: datetime
    auto_commit_enabled: bool = False


class ImportedTerminologyRecord(BaseModel):
    concept_id: str = Field(min_length=1)
    term: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    language: str = Field(min_length=2)
    concept_type: str = "medication"
    risk_tier: str = "unclassified"
    lasa_status: str = "unclassified"
    source_vocabulary: str
    source_release: str
    source_record_type: str
    licensing_status: str
    review_status: str = "unreviewed"
    provenance: dict[str, Any]
    pronunciations: list[str] = Field(default_factory=list)
    source_row_hash: str


def register_source(
    source: Path,
    build_root: Path,
    *,
    release_id: str,
    source_release: str,
    license_status: str,
    license_approval_id: str,
    license_owner: str,
    license_evidence: str,
    license_approval_date: str,
) -> SourceManifest:
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest = SourceManifest(
        release_id=release_id,
        source_name="RxNorm",
        source_type="rxnorm-rrf",
        source_release=source_release,
        source_path=str(source.resolve()),
        source_sha256=sha256_file(source),
        license_status=license_status,
        license_approval_id=license_approval_id,
        license_owner=license_owner,
        license_evidence=license_evidence,
        license_approval_date=license_approval_date,
        required_languages=["en"],
        created_at=utc_now(),
    )
    manifest_path = build_root / release_id / "source" / "source_manifest.json"
    if manifest_path.is_file():
        existing = SourceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if existing.source_sha256 != manifest.source_sha256:
            raise ValueError(
                "Release directory is already registered to a different source checksum."
            )
        return existing
    atomic_write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load_source_manifest(path: Path) -> SourceManifest:
    manifest = SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.contains_patient_data:
        raise ValueError("Terminology builds must not contain patient data.")
    source = Path(manifest.source_path)
    if not source.is_file() or sha256_file(source) != manifest.source_sha256:
        raise ValueError("Registered source file is missing or its checksum has changed.")
    return manifest


def _connect_staging(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_rows (
          ordinal INTEGER PRIMARY KEY, concept_id TEXT NOT NULL, term TEXT NOT NULL,
          is_preferred INTEGER NOT NULL, tty TEXT NOT NULL, sab TEXT NOT NULL,
          source_row_hash TEXT NOT NULL UNIQUE
        );
        CREATE INDEX IF NOT EXISTS idx_source_rows_concept
          ON source_rows(concept_id, is_preferred DESC, ordinal);
        CREATE TABLE IF NOT EXISTS rejects (
          ordinal INTEGER PRIMARY KEY, reason TEXT NOT NULL, source_row_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS row_outcomes (
          ordinal INTEGER PRIMARY KEY, outcome TEXT NOT NULL
        );
        """
    )
    return connection


def _checkpoint(path: Path, checkpoint: ImportCheckpoint, status: str) -> None:
    now = utc_now()
    checkpoint.status = status  # type: ignore[assignment]
    checkpoint.updated_at = now
    checkpoint.history.append(
        {
            "status": status,
            "source_rows_processed": checkpoint.source_rows_processed,
            "timestamp": now.isoformat(),
        }
    )
    atomic_write_text(path, checkpoint.model_dump_json(indent=2) + "\n")


def import_rxnorm_release(
    manifest_path: Path,
    output_dir: Path,
    *,
    concept_batch_size: int = 25_000,
    parse_checkpoint_rows: int = 100_000,
    max_concepts: int | None = None,
    resume: bool = False,
) -> ImportSummary:
    if concept_batch_size < 1 or parse_checkpoint_rows < 1:
        raise ValueError("Batch sizes must be positive.")
    if max_concepts is not None and max_concepts < 1:
        raise ValueError("max_concepts must be positive.")
    manifest = load_source_manifest(manifest_path)
    stage_inputs = {
        "schema_version": IMPORT_SCHEMA_VERSION,
        "source_sha256": manifest.source_sha256,
        "source_release": manifest.source_release,
        "license_approval_id": manifest.license_approval_id,
        "filters": {
            "language": "ENG",
            "sab": "RXNORM",
            "suppress": "N",
            "term_types": sorted(RXNORM_TERM_TYPES),
        },
        "sort": ["language", "concept_id", "normalized_term", "source_row_hash"],
        "concept_batch_size": concept_batch_size,
        "max_concepts": max_concepts,
    }
    stage_hash = canonical_hash(stage_inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    staging_path = output_dir / "staging.sqlite3"
    if checkpoint_path.is_file():
        checkpoint = ImportCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if checkpoint.stage_hash != stage_hash:
            raise ValueError("Existing import checkpoint has a different stage hash.")
        if checkpoint.status == "complete" and resume:
            return ImportSummary.model_validate_json(
                (output_dir / "import_summary.json").read_text(encoding="utf-8")
            )
        if not resume:
            raise ValueError(
                "Import output already exists; use --resume or a new release directory."
            )
        checkpoint.attempt += 1
    else:
        checkpoint = ImportCheckpoint(
            release_id=manifest.release_id,
            stage_hash=stage_hash,
            source_sha256=manifest.source_sha256,
            updated_at=utc_now(),
        )
    _checkpoint(checkpoint_path, checkpoint, "running")
    counts = checkpoint.counts.copy()
    connection = _connect_staging(staging_path)
    source = Path(manifest.source_path)
    try:
        with source.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="|")
            for ordinal, values in enumerate(reader, start=1):
                if ordinal <= checkpoint.source_rows_processed:
                    continue
                counts["read"] += 1
                row_hash = hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()
                if len(values) < len(RXNCONSO_FIELDS):
                    connection.execute(
                        "INSERT OR IGNORE INTO rejects VALUES (?, ?, ?)",
                        (ordinal, "malformed_column_count", row_hash),
                    )
                    counts["rejected"] += 1
                    outcome = "rejected"
                else:
                    row = dict(zip(RXNCONSO_FIELDS, values, strict=False))
                    if row["SUPPRESS"] != "N":
                        counts["suppressed"] += 1
                        outcome = "suppressed"
                    elif (
                        row["LAT"] != "ENG"
                        or row["SAB"] != "RXNORM"
                        or row["TTY"] not in RXNORM_TERM_TYPES
                    ):
                        counts["filtered"] += 1
                        outcome = "filtered"
                    elif not row["RXCUI"].strip() or not row["STR"].strip():
                        connection.execute(
                            "INSERT OR IGNORE INTO rejects VALUES (?, ?, ?)",
                            (ordinal, "missing_concept_id_or_term", row_hash),
                        )
                        counts["rejected"] += 1
                        outcome = "rejected"
                    else:
                        cursor = connection.execute(
                            "INSERT OR IGNORE INTO source_rows VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                ordinal,
                                row["RXCUI"],
                                row["STR"],
                                row["ISPREF"] == "Y",
                                row["TTY"],
                                row["SAB"],
                                row_hash,
                            ),
                        )
                        if cursor.rowcount:
                            counts["accepted_rows"] += 1
                            outcome = "accepted"
                        else:
                            counts["duplicate"] += 1
                            outcome = "duplicate"
                connection.execute(
                    "INSERT OR REPLACE INTO row_outcomes VALUES (?, ?)",
                    (ordinal, outcome),
                )
                checkpoint.source_rows_processed = ordinal
                if ordinal % parse_checkpoint_rows == 0:
                    connection.commit()
                    checkpoint.counts = counts.copy()
                    _checkpoint(checkpoint_path, checkpoint, "running")
        connection.commit()
        checkpoint.counts = counts.copy()
        _checkpoint(checkpoint_path, checkpoint, "parsed")
        _write_rejects(connection, output_dir / "rejected.jsonl")
        shard_checksums, emitted = _write_concept_shards(
            connection, manifest, output_dir / "batches", concept_batch_size, max_concepts
        )
        counts["emitted_records"] = emitted
        outcome_counts = dict(
            connection.execute(
                "SELECT outcome, COUNT(*) FROM row_outcomes GROUP BY outcome"
            ).fetchall()
        )
        counts["read"] = sum(outcome_counts.values())
        for name in ("rejected", "suppressed", "filtered", "duplicate"):
            counts[name] = int(outcome_counts.get(name, 0))
        counts["accepted_rows"] = int(
            connection.execute("SELECT COUNT(*) FROM source_rows").fetchone()[0]
        )
        checkpoint.counts = counts.copy()
        summary = ImportSummary(
            release_id=manifest.release_id,
            stage_hash=stage_hash,
            source_sha256=manifest.source_sha256,
            partial=max_concepts is not None,
            max_concepts=max_concepts,
            counts=counts,
            shard_checksums=shard_checksums,
            staging_size_bytes=staging_path.stat().st_size,
            completed_at=utc_now(),
        )
        atomic_write_text(
            output_dir / "import_summary.json", summary.model_dump_json(indent=2) + "\n"
        )
        _checkpoint(checkpoint_path, checkpoint, "complete")
        return summary
    finally:
        connection.close()


def _write_rejects(connection: sqlite3.Connection, path: Path) -> None:
    lines = (
        canonical_json({"source_row_ordinal": row[0], "reason": row[1], "source_row_hash": row[2]})
        for row in connection.execute(
            "SELECT ordinal, reason, source_row_hash FROM rejects ORDER BY ordinal"
        )
    )
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def _preferred_rows(
    connection: sqlite3.Connection, max_concepts: int | None
) -> Iterator[tuple[Any, ...]]:
    limit = "" if max_concepts is None else " LIMIT ?"
    params: tuple[int, ...] = () if max_concepts is None else (max_concepts,)
    query = (
        """
      SELECT s.concept_id, s.term, s.tty, s.sab, s.source_row_hash, s.ordinal
      FROM source_rows s
      WHERE s.ordinal = (
        SELECT s2.ordinal FROM source_rows s2 WHERE s2.concept_id = s.concept_id
        ORDER BY s2.is_preferred DESC, s2.ordinal ASC LIMIT 1
      )
      ORDER BY s.concept_id ASC, lower(s.term) ASC, s.source_row_hash ASC
    """
        + limit
    )
    yield from connection.execute(query, params)


def _write_concept_shards(
    connection: sqlite3.Connection,
    manifest: SourceManifest,
    batches_dir: Path,
    batch_size: int,
    max_concepts: int | None,
) -> tuple[dict[str, str], int]:
    batches_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    buffer: list[str] = []
    emitted = 0
    shard_number = 1
    for concept_id, term, tty, sab, row_hash, ordinal in _preferred_rows(connection, max_concepts):
        aliases = [
            row[0]
            for row in connection.execute(
                "SELECT term FROM source_rows WHERE concept_id=? AND term<>? ORDER BY ordinal",
                (concept_id, term),
            )
        ]
        record = ImportedTerminologyRecord(
            concept_id=f"RxCUI:{concept_id}",
            term=term,
            aliases=list(dict.fromkeys(aliases)),
            language="en",
            source_vocabulary="RxNorm",
            source_release=manifest.source_release,
            source_record_type=tty,
            licensing_status=manifest.license_status,
            provenance={
                "SAB": sab,
                "TTY": tty,
                "source_row_ordinal": ordinal,
                "source_sha256": manifest.source_sha256,
                "license_approval_id": manifest.license_approval_id,
            },
            source_row_hash=row_hash,
        )
        buffer.append(canonical_json(record.model_dump(mode="json")))
        emitted += 1
        if len(buffer) == batch_size:
            _flush_shard(batches_dir, shard_number, buffer, checksums)
            shard_number += 1
            buffer.clear()
    if buffer:
        _flush_shard(batches_dir, shard_number, buffer, checksums)
    return checksums, emitted


def _flush_shard(path: Path, number: int, lines: list[str], checksums: dict[str, str]) -> None:
    name = f"batch-{number:06d}.jsonl"
    target = path / name
    atomic_write_text(target, "".join(f"{line}\n" for line in lines))
    checksums[name] = sha256_file(target)


def finalize_dictionary(
    import_dir: Path, terminology_dir: Path, *, version: str | None = None
) -> dict[str, Any]:
    summary = ImportSummary.model_validate_json(
        (import_dir / "import_summary.json").read_text(encoding="utf-8")
    )
    if not (import_dir / "checkpoint.json").is_file():
        raise ValueError("Import checkpoint is missing.")
    checkpoint = ImportCheckpoint.model_validate_json(
        (import_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    if checkpoint.status != "complete":
        raise ValueError("Import is incomplete.")
    terminology_dir.mkdir(parents=True, exist_ok=True)
    uniqueness = sqlite3.connect(terminology_dir / "validation.sqlite3")
    uniqueness.execute(
        "CREATE TABLE IF NOT EXISTS identities "
        "(concept_id TEXT PRIMARY KEY, language TEXT NOT NULL)"
    )
    inventory: dict[str, dict[str, int]] = {"languages": {}, "concept_types": {}, "risk_tiers": {}}
    artifact_version = version or f"{summary.release_id}-{summary.stage_hash[:12]}"
    dictionary_path = terminology_dir / "dictionary.json"
    temporary = dictionary_path.with_name(f".{dictionary_path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            header = {
                "artifact_version": artifact_version,
                "created_at": utc_now().isoformat(),
                "source_sha256": summary.source_sha256,
            }
            output.write(
                "{"
                + ",".join(f"{json.dumps(k)}:{canonical_json(v)}" for k, v in header.items())
                + ',"entries":['
            )
            first = True
            engine = PronunciationEngine({})
            for shard in sorted((import_dir / "batches").glob("batch-*.jsonl")):
                expected = summary.shard_checksums.get(shard.name)
                if expected != sha256_file(shard):
                    raise ValueError(f"Import shard checksum mismatch: {shard.name}")
                for line_number, line in enumerate(shard.open(encoding="utf-8"), start=1):
                    try:
                        record = ImportedTerminologyRecord.model_validate_json(line)
                        uniqueness.execute(
                            "INSERT INTO identities VALUES (?, ?)",
                            (record.concept_id, record.language),
                        )
                    except (ValueError, sqlite3.IntegrityError) as exc:
                        raise ValueError(
                            f"Invalid or duplicate record in {shard.name}:{line_number}"
                        ) from exc
                    aliases = list(dict.fromkeys([record.term, *record.aliases]))
                    normalized_aliases = [normalize_term(item, record.language) for item in aliases]
                    pronunciation_forms = [
                        form.value
                        for item in aliases
                        for form in engine.generate(item, record.language)
                    ]
                    entry = {
                        "concept_id": record.concept_id,
                        "term": record.term,
                        "aliases": record.aliases,
                        "language": record.language,
                        "risk_tier": record.risk_tier,
                        "prior": 0.5,
                        "provenance": record.source_vocabulary,
                        "source_vocabulary": record.source_vocabulary,
                        "source_release": record.source_release,
                        "licensing_status": record.licensing_status,
                        "review_status": record.review_status,
                        "concept_type": record.concept_type,
                        "source_record_type": record.source_record_type,
                        "lasa_status": record.lasa_status,
                        "provenance_detail": record.provenance,
                        "pronunciations": record.pronunciations,
                        "normalized_term": normalize_term(record.term, record.language),
                        "normalized_aliases": list(dict.fromkeys(normalized_aliases)),
                        "pronunciation_forms": list(dict.fromkeys(pronunciation_forms)),
                    }
                    if not first:
                        output.write(",")
                    output.write(canonical_json(entry))
                    first = False
                    count += 1
                    for dimension, value in (
                        ("languages", record.language),
                        ("concept_types", record.concept_type),
                        ("risk_tiers", record.risk_tier),
                    ):
                        inventory[dimension][value] = inventory[dimension].get(value, 0) + 1
            output.write("]}\n")
            output.flush()
            os.fsync(output.fileno())
        uniqueness.commit()
        temporary.replace(dictionary_path)
    finally:
        uniqueness.close()
        temporary.unlink(missing_ok=True)
    if count != summary.counts["emitted_records"]:
        raise ValueError("Final dictionary count does not reconcile with imported records.")
    result = {
        "schema_version": 1,
        "artifact_version": artifact_version,
        "record_count": count,
        "inventory": inventory,
        "dictionary_sha256": sha256_file(dictionary_path),
        "source_sha256": summary.source_sha256,
        "shard_checksums": summary.shard_checksums,
        "partial": summary.partial,
        "auto_commit_enabled": False,
    }
    atomic_write_text(
        terminology_dir / "inventory.json", json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
