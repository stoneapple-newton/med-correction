from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from medterm.audio_embeddings import AudioEmbeddingIndex
from medterm.vector_batch import build_vector_shards, finalize_vector_index


class BatchEncoder:
    calls = 0

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "embedding_model": "fake-batch-v1",
            "model_revision": "fixed",
            "pooling": "statistics",
            "projection_checkpoint": None,
            "output_dimension": 3,
        }

    def encode(self, audio_path: Path) -> np.ndarray:
        return self.encode_many([audio_path])[0]

    def encode_many(self, audio_paths: list[Path]) -> np.ndarray:
        self.calls += 1
        return np.asarray(
            [[float(int(path.stem)), 1.0, 0.5] for path in audio_paths], dtype=np.float32
        )


def _manifest(tmp_path: Path, count: int = 5) -> Path:
    path = tmp_path / "active_references.jsonl"
    records = [
        {
            "reference_id": f"ref-{index}",
            "audio_path": f"audio/{index}.wav",
            "concept_id": f"LOCAL:{index}",
            "term": f"term-{index}",
            "language": "en",
            "synthetic": True,
            "pronunciation_type": "synthetic_tts",
        }
        for index in range(1, count + 1)
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return path


def test_vector_shards_resume_and_finalize_without_auto_commit(tmp_path: Path) -> None:
    encoder = BatchEncoder()
    manifest = _manifest(tmp_path)
    vector_dir = tmp_path / "vectors" / "en"

    checkpoint = build_vector_shards(
        manifest, vector_dir, encoder, micro_batch_size=2, shard_size=3
    )
    calls_after_first_run = encoder.calls
    resumed = build_vector_shards(
        manifest, vector_dir, encoder, micro_batch_size=2, shard_size=3, resume=True
    )
    summary = finalize_vector_index(vector_dir, tmp_path / "indexes" / "vectors-en.npz")
    index = AudioEmbeddingIndex.load(tmp_path / "indexes" / "vectors-en.npz")

    assert len(checkpoint.shards) == 2
    assert len(resumed.shards) == 2
    assert encoder.calls == calls_after_first_run
    assert summary["row_count"] == 5
    assert summary["unexplained_inventory_difference"] == 0
    assert summary["auto_commit_enabled"] is False
    assert index.vectors.shape == (5, 3)
    assert index.provenance["output_dimension"] == 3
    assert (tmp_path / "indexes" / "vectors-en.summary.json").is_file()


def test_vector_batch_rejects_duplicate_reference_ids(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, count=2)
    records = manifest.read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(records[1])
    duplicate["reference_id"] = "ref-1"
    manifest.write_text(records[0] + "\n" + json.dumps(duplicate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        build_vector_shards(manifest, tmp_path / "vectors", BatchEncoder())
