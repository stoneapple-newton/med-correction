from __future__ import annotations

import json
from pathlib import Path

from medterm.terminology import DictionaryArtifact, DictionaryEntry
from medterm.tts import SynthesisResult, build_synthetic_references, write_reference_manifest


class FakeSynthesizer:
    @property
    def provenance(self) -> dict[str, object]:
        return {
            "tts_model_id": "hexgrad/Kokoro-82M",
            "tts_voice": "af_heart",
            "tts_language_code": "a",
            "tts_speed": 1.0,
            "tts_sample_rate": 24_000,
        }

    def synthesize(
        self, text: str, output_path: Path, *, phonemes: str | None = None
    ) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF-synthetic-test")
        return SynthesisResult(output_path, 24_000, phonemes or f"g2p:{text}")


def _artifact() -> DictionaryArtifact:
    from datetime import UTC, datetime

    return DictionaryArtifact(
        artifact_version="dict-test-v1",
        created_at=datetime.now(UTC),
        source_sha256="a" * 64,
        entries=[
            DictionaryEntry(
                concept_id="LOCAL:METFORMIN",
                term="metformin",
                language="en",
                pronunciations=["mɛtˈfɔɹmɪn"],
                normalized_term="metformin",
                normalized_aliases=["metformin"],
                pronunciation_forms=["mɛtˈfɔɹmɪn"],
            ),
            DictionaryEntry(
                concept_id="LOCAL:ZH",
                term="二甲双胍",
                language="zh",
                normalized_term="二甲双胍",
                normalized_aliases=["二甲双胍"],
                pronunciation_forms=[],
            ),
        ],
    )


def test_build_synthetic_references_preserves_tts_and_terminology_provenance(
    tmp_path: Path,
) -> None:
    references = build_synthetic_references(
        _artifact(), FakeSynthesizer(), tmp_path / "audio", language="en"
    )

    assert len(references) == 1
    reference = references[0]
    assert reference.audio_path.is_file()
    assert reference.pronunciation_type == "synthetic_tts"
    assert reference.synthetic is True
    assert reference.tts_phonemes == "mɛtˈfɔɹmɪn"
    assert reference.tts_model_id == "hexgrad/Kokoro-82M"
    assert reference.terminology_version == "dict-test-v1"
    assert reference.terminology_source_sha256 == "a" * 64


def test_manifest_uses_relative_audio_paths_and_round_trips_metadata(tmp_path: Path) -> None:
    references = build_synthetic_references(
        _artifact(), FakeSynthesizer(), tmp_path / "audio", language="en"
    )
    manifest = tmp_path / "references.jsonl"

    write_reference_manifest(references, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["audio_path"].startswith("audio/")
    assert payload["synthetic"] is True
    assert payload["tts_voice"] == "af_heart"


def test_existing_audio_is_reused_without_losing_curated_phonemes(tmp_path: Path) -> None:
    synthesizer = FakeSynthesizer()
    first = build_synthetic_references(_artifact(), synthesizer, tmp_path, language="en")
    first[0].audio_path.write_bytes(b"existing")

    second = build_synthetic_references(_artifact(), synthesizer, tmp_path, language="en")

    assert second[0].audio_path.read_bytes() == b"existing"
    assert second[0].tts_phonemes == "mɛtˈfɔɹmɪn"


def test_missing_language_is_an_explicit_error(tmp_path: Path) -> None:
    try:
        build_synthetic_references(_artifact(), FakeSynthesizer(), tmp_path, language="ko")
    except ValueError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("A missing terminology language should not be fabricated")


def test_mismatched_kokoro_language_code_is_rejected(tmp_path: Path) -> None:
    class ChineseVoice(FakeSynthesizer):
        @property
        def provenance(self) -> dict[str, object]:
            return {**super().provenance, "tts_language_code": "z"}

    try:
        build_synthetic_references(_artifact(), ChineseVoice(), tmp_path, language="en")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("A mismatched language/voice pipeline should be rejected")
