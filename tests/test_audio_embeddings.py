from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from medterm.audio_cli import _ascii_console_json
from medterm.audio_embeddings import (
    AudioEmbeddingIndex,
    ReferencePronunciation,
    build_reference_index,
    load_reference_manifest,
    train_projection_head,
    validate_query_encoder,
)


class FakeEncoder:
    vectors = {
        "metformin-us.wav": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "metformin-gb.wav": np.array([0.95, 0.05, 0.0], dtype=np.float32),
        "metoprolol-us.wav": np.array([0.1, 0.9, 0.0], dtype=np.float32),
    }

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "embedding_model": "fake-medical-awe",
            "pooling": "statistics",
            "projection_trained": True,
            "output_dimension": 3,
        }

    def encode(self, audio_path: Path) -> np.ndarray:
        return self.vectors[audio_path.name]


def test_console_json_escapes_ipa_for_legacy_windows_terminals() -> None:
    rendered = _ascii_console_json({"phonemes": "mɛtˈfɔɹmɪn"})

    rendered.encode("cp1252")
    assert "mɛt" not in rendered
    assert "\\u025b" in rendered


def _references(tmp_path: Path) -> list[ReferencePronunciation]:
    return [
        ReferencePronunciation(
            reference_id="metformin-us",
            audio_path=tmp_path / "metformin-us.wav",
            concept_id="LOCAL:METFORMIN",
            term="metformin",
            language="en",
            country="US",
            speaker_accent="en-US",
        ),
        ReferencePronunciation(
            reference_id="metformin-gb",
            audio_path=tmp_path / "metformin-gb.wav",
            concept_id="LOCAL:METFORMIN",
            term="metformin",
            language="en",
            country="GB",
            speaker_accent="en-GB",
        ),
        ReferencePronunciation(
            reference_id="metoprolol-us",
            audio_path=tmp_path / "metoprolol-us.wav",
            concept_id="LOCAL:METOPROLOL",
            term="metoprolol",
            language="en",
            country="US",
            speaker_accent="en-US",
        ),
    ]


def test_index_returns_multiple_pronunciations_and_never_commits(tmp_path: Path) -> None:
    index = build_reference_index(_references(tmp_path), FakeEncoder())

    response = index.search_for_review(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=20,
        filters={"language": "en"},
    )

    assert response.decision == "review"
    assert response.auto_commit_enabled is False
    assert [candidate.reference_id for candidate in response.candidates[:2]] == [
        "metformin-us",
        "metformin-gb",
    ]
    assert response.embedding_provenance["projection_trained"] is True


def test_index_filters_before_scoring_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "references.npz"
    build_reference_index(_references(tmp_path), FakeEncoder()).save(path)
    restored = AudioEmbeddingIndex.load(path)

    response = restored.search_for_review(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        filters={"country": ["GB"]},
    )

    assert [candidate.reference_id for candidate in response.candidates] == ["metformin-gb"]
    assert "audio_path" not in response.candidates[0].metadata


def test_empty_metadata_filter_abstains(tmp_path: Path) -> None:
    index = build_reference_index(_references(tmp_path), FakeEncoder())

    response = index.search_for_review(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        filters={"language": "ja"},
    )

    assert response.decision == "abstain"
    assert response.auto_commit_enabled is False
    assert response.candidates == []


def test_manifest_resolves_relative_audio_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "references.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "reference_id": "metformin-us",
                "audio_path": "audio/metformin.wav",
                "concept_id": "LOCAL:METFORMIN",
                "term": "metformin",
                "language": "en",
            }
        ),
        encoding="utf-8",
    )

    references = load_reference_manifest(manifest)

    assert references[0].audio_path == tmp_path / "audio" / "metformin.wav"


def test_mismatched_encoder_is_rejected(tmp_path: Path) -> None:
    index = build_reference_index(_references(tmp_path), FakeEncoder())

    class MismatchedEncoder(FakeEncoder):
        @property
        def provenance(self) -> dict[str, object]:
            return {**super().provenance, "embedding_model": "different-model"}

    try:
        validate_query_encoder(index, MismatchedEncoder())
    except ValueError as exc:
        assert "embedding_model" in str(exc)
    else:
        raise AssertionError("Mismatched encoder should have been rejected")


def test_projection_training_writes_auditable_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    checkpoint_path = tmp_path / "projection.pt"

    summary = train_projection_head(
        features,
        ["term-a", "term-a", "term-b", "term-b"],
        checkpoint_path,
        output_size=2,
        epochs=5,
        device="cpu",
        training_provenance={"dataset": "synthetic-unit-test"},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert checkpoint["trained"] is True
    assert checkpoint["input_size"] == 4
    assert checkpoint["output_size"] == 2
    assert checkpoint["training_provenance"]["dataset"] == "synthetic-unit-test"
    assert np.isfinite(summary["final_loss"])
