from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from medterm.audio_embeddings import (
    AudioEmbeddingIndex,
    AudioEncoder,
    ReferencePronunciation,
    load_reference_manifest,
)
from medterm.bulk import atomic_write_text, canonical_hash, sha256_file


class VectorShardRecord(BaseModel):
    shard_id: str
    input_reference_ids: list[str]
    input_checksum: str
    output_path: str
    output_checksum: str
    count: int
    dimension: int
    status: str = "complete"
    completed_at: datetime


class VectorCheckpoint(BaseModel):
    schema_version: int = 1
    stage_hash: str
    manifest_sha256: str
    encoder_provenance: dict[str, Any]
    shards: list[VectorShardRecord] = Field(default_factory=list)
    auto_commit_enabled: bool = False


def _resolve_encoder(encoder: AudioEncoder) -> None:
    loader = getattr(encoder, "_load_model", None)
    if callable(loader):
        loader()


def _encode_many(encoder: AudioEncoder, references: list[ReferencePronunciation]) -> np.ndarray:
    method = getattr(encoder, "encode_many", None)
    if callable(method):
        matrix = np.asarray(method([item.audio_path for item in references]), dtype=np.float32)
    else:
        matrix = np.stack([encoder.encode(item.audio_path) for item in references])
    if matrix.ndim != 2 or matrix.shape[0] != len(references):
        raise ValueError("Encoder batch output does not align with the input references.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Vector shard contains non-finite values.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Vector shard contains a zero vector.")
    return matrix


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_vector_shards(
    manifest_path: Path,
    output_dir: Path,
    encoder: AudioEncoder,
    *,
    micro_batch_size: int = 32,
    shard_size: int = 5_000,
    resume: bool = False,
) -> VectorCheckpoint:
    if micro_batch_size < 1 or shard_size < 1:
        raise ValueError("Batch and shard sizes must be positive.")
    references = load_reference_manifest(manifest_path)
    if not references:
        raise ValueError("Active reference manifest is empty.")
    identities = [item.reference_id for item in references]
    if len(identities) != len(set(identities)):
        raise ValueError("Active reference manifest contains duplicate reference IDs.")
    _resolve_encoder(encoder)
    provenance = encoder.provenance
    manifest_sha256 = sha256_file(manifest_path)
    stage_hash = canonical_hash(
        {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "encoder": provenance,
            "micro_batch_size": micro_batch_size,
            "shard_size": shard_size,
            "dtype": "float32",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = VectorCheckpoint(
        stage_hash=stage_hash,
        manifest_sha256=manifest_sha256,
        encoder_provenance=provenance,
    )
    completed: dict[str, VectorShardRecord] = {}
    if checkpoint_path.is_file():
        existing = VectorCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
        if existing.stage_hash != stage_hash:
            raise ValueError("Existing vector checkpoint has a different stage hash.")
        if not resume:
            raise ValueError("Vector output already exists; use --resume.")
        completed = {item.shard_id: item for item in existing.shards}

    all_records: list[VectorShardRecord] = []
    for shard_start in range(0, len(references), shard_size):
        shard_number = shard_start // shard_size + 1
        shard_id = f"batch-{shard_number:06d}"
        shard_references = references[shard_start : shard_start + shard_size]
        input_checksum = canonical_hash([item.reference_id for item in shard_references])
        target = output_dir / "shards" / f"{shard_id}.npz"
        prior = completed.get(shard_id)
        if (
            prior is not None
            and prior.input_checksum == input_checksum
            and target.is_file()
            and sha256_file(target) == prior.output_checksum
        ):
            all_records.append(prior)
            continue
        matrices = []
        for start in range(0, len(shard_references), micro_batch_size):
            micro_batch = shard_references[start : start + micro_batch_size]
            try:
                matrices.append(_encode_many(encoder, micro_batch))
            except RuntimeError as exc:
                if "out of memory" not in str(exc).casefold() or len(micro_batch) == 1:
                    raise
                midpoint = max(1, len(micro_batch) // 2)
                matrices.append(_encode_many(encoder, micro_batch[:midpoint]))
                matrices.append(_encode_many(encoder, micro_batch[midpoint:]))
        matrix = np.concatenate(matrices)
        metadata = [
            item.model_dump(mode="json", exclude={"audio_path"}) for item in shard_references
        ]
        _write_npz_atomic(
            target,
            vectors=matrix,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            provenance=np.asarray(json.dumps(provenance, ensure_ascii=False)),
            stage_hash=np.asarray(stage_hash),
            schema_version=np.asarray(1, dtype=np.int64),
        )
        record = VectorShardRecord(
            shard_id=shard_id,
            input_reference_ids=[item.reference_id for item in shard_references],
            input_checksum=input_checksum,
            output_path=str(target.resolve()),
            output_checksum=sha256_file(target),
            count=matrix.shape[0],
            dimension=matrix.shape[1],
            completed_at=datetime.now(UTC),
        )
        all_records.append(record)
        checkpoint.shards = all_records
        atomic_write_text(checkpoint_path, checkpoint.model_dump_json(indent=2) + "\n")
    checkpoint.shards = all_records
    atomic_write_text(checkpoint_path, checkpoint.model_dump_json(indent=2) + "\n")
    return checkpoint


def finalize_vector_index(vector_dir: Path, output_path: Path) -> dict[str, Any]:
    checkpoint = VectorCheckpoint.model_validate_json(
        (vector_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    dimensions: set[int] = set()
    seen: set[str] = set()
    shard_checksums: dict[str, str] = {}
    for record in checkpoint.shards:
        path = Path(record.output_path)
        if not path.is_absolute():
            path = vector_dir / path
        if sha256_file(path) != record.output_checksum:
            raise ValueError(f"Vector shard checksum mismatch: {record.shard_id}")
        with np.load(path, allow_pickle=False) as data:
            if str(data["stage_hash"]) != checkpoint.stage_hash:
                raise ValueError(f"Vector shard stage hash mismatch: {record.shard_id}")
            matrix = np.asarray(data["vectors"], dtype=np.float32)
            items = json.loads(str(data["metadata"]))
        if matrix.ndim != 2 or matrix.shape[0] != len(items):
            raise ValueError(f"Invalid vector shard shape: {record.shard_id}")
        if not np.all(np.isfinite(matrix)) or np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError(f"Invalid vectors in shard: {record.shard_id}")
        for item in items:
            reference_id = item["reference_id"]
            if reference_id in seen:
                raise ValueError(f"Duplicate vector reference: {reference_id}")
            seen.add(reference_id)
        dimensions.add(matrix.shape[1])
        vectors.append(matrix)
        metadata.extend(items)
        shard_checksums[record.shard_id] = record.output_checksum
    if len(dimensions) != 1 or not vectors:
        raise ValueError("Vector shards are empty or have incompatible dimensions.")
    dimension = next(iter(dimensions))
    provenance = {**checkpoint.encoder_provenance, "output_dimension": dimension}
    index = AudioEmbeddingIndex(np.concatenate(vectors), metadata, provenance)
    index.save(output_path)
    summary = {
        "schema_version": 1,
        "index_path": str(output_path),
        "index_sha256": sha256_file(output_path),
        "row_count": len(metadata),
        "dimension": dimension,
        "shard_checksums": shard_checksums,
        "failed_renditions": 0,
        "unexplained_inventory_difference": 0,
        "decision": "review",
        "auto_commit_enabled": False,
    }
    atomic_write_text(
        output_path.with_suffix(".summary.json"),
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary
