from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from medterm.audio_embeddings import ReferencePronunciation
from medterm.terminology import DictionaryArtifact

KOKORO_LANGUAGE_CODES: dict[str, frozenset[str]] = {
    "en": frozenset({"a", "b"}),
    "es": frozenset({"e"}),
    "fr": frozenset({"f"}),
    "hi": frozenset({"h"}),
    "it": frozenset({"i"}),
    "ja": frozenset({"j"}),
    "pt": frozenset({"p"}),
    "zh": frozenset({"z"}),
}


class TTSUnavailableError(RuntimeError):
    """Raised when the optional local TTS runtime or model is unavailable."""


class TTSGenerationError(RuntimeError):
    """Raised when TTS produces no usable audio for an isolated term."""


@dataclass(frozen=True)
class KokoroConfig:
    model_id: str = "hexgrad/Kokoro-82M"
    voice: str = "af_heart"
    language_code: str = "a"
    device: str = "auto"
    speed: float = 1.0
    sample_rate: int = 24_000


@dataclass(frozen=True)
class SynthesisResult:
    audio_path: Path
    sample_rate: int
    phonemes: str | None


class TermSynthesizer(Protocol):
    @property
    def provenance(self) -> dict[str, Any]: ...

    def synthesize(
        self, text: str, output_path: Path, *, phonemes: str | None = None
    ) -> SynthesisResult: ...


class KokoroSynthesizer:
    """Lazy local Kokoro-82M adapter for isolated terminology pronunciations."""

    def __init__(self, config: KokoroConfig | None = None) -> None:
        self.config = config or KokoroConfig()
        if not self.config.model_id.strip() or not self.config.voice.strip():
            raise ValueError("Kokoro model_id and voice must be non-empty.")
        if self.config.speed <= 0 or self.config.sample_rate <= 0:
            raise ValueError("Kokoro speed and sample rate must be positive.")
        self._pipeline: Any | None = None
        self._soundfile: Any | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "tts_model_id": self.config.model_id,
            "tts_voice": self.config.voice,
            "tts_language_code": self.config.language_code,
            "tts_device": self.config.device,
            "tts_speed": self.config.speed,
            "tts_sample_rate": self.config.sample_rate,
            "pronunciation_type": "synthetic_tts",
        }

    def synthesize(
        self, text: str, output_path: Path, *, phonemes: str | None = None
    ) -> SynthesisResult:
        if not text.strip():
            raise ValueError("TTS text must be non-empty.")
        if output_path.suffix.casefold() != ".wav":
            raise ValueError("Kokoro reference audio must use a .wav extension.")
        self._load_pipeline()
        try:
            if phonemes:
                generated = self._pipeline.generate_from_tokens(
                    phonemes, voice=self.config.voice, speed=self.config.speed
                )
            else:
                generated = self._pipeline(
                    text,
                    voice=self.config.voice,
                    speed=self.config.speed,
                    split_pattern=None,
                )
            chunks: list[np.ndarray] = []
            generated_phonemes: list[str] = []
            for result in generated:
                if result.audio is None:
                    continue
                audio = result.audio
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                array = np.asarray(audio, dtype=np.float32).reshape(-1)
                if array.size:
                    chunks.append(array)
                if getattr(result, "phonemes", None):
                    generated_phonemes.append(str(result.phonemes))
        except Exception as exc:
            raise TTSGenerationError(f"Kokoro failed to synthesize term: {text!r}") from exc
        if not chunks:
            raise TTSGenerationError(f"Kokoro produced no audio for term: {text!r}")
        waveform = np.concatenate(chunks)
        if not np.all(np.isfinite(waveform)) or float(np.max(np.abs(waveform))) == 0.0:
            raise TTSGenerationError(f"Kokoro produced invalid audio for term: {text!r}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.stem}.{uuid.uuid4().hex}.tmp.wav"
        )
        try:
            self._soundfile.write(
                temporary, waveform, self.config.sample_rate, subtype="PCM_16"
            )
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return SynthesisResult(
            audio_path=output_path,
            sample_rate=self.config.sample_rate,
            phonemes=" ".join(generated_phonemes) or phonemes,
        )

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import soundfile as sf
            from kokoro import KPipeline
        except ImportError as exc:
            raise TTSUnavailableError(
                "Install the 'tts' optional dependency to enable Kokoro-82M."
            ) from exc
        device = None if self.config.device == "auto" else self.config.device
        try:
            self._pipeline = KPipeline(
                lang_code=self.config.language_code,
                repo_id=self.config.model_id,
                device=device,
            )
        except Exception as exc:
            raise TTSUnavailableError(
                f"Unable to load local Kokoro model {self.config.model_id!r}."
            ) from exc
        self._soundfile = sf


def build_synthetic_references(
    artifact: DictionaryArtifact,
    synthesizer: TermSynthesizer,
    output_dir: Path,
    *,
    language: str = "en",
    limit: int | None = None,
    overwrite: bool = False,
) -> list[ReferencePronunciation]:
    """Synthesize one auditable reference per terminology entry for one language."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided.")
    language_code = synthesizer.provenance.get("tts_language_code")
    if language_code is not None:
        supported_codes = KOKORO_LANGUAGE_CODES.get(language)
        if supported_codes is None:
            raise ValueError(
                f"Kokoro-82M does not support terminology language {language!r}."
            )
        if str(language_code) not in supported_codes:
            choices = ", ".join(sorted(supported_codes))
            raise ValueError(
                f"Kokoro language code {language_code!r} does not match {language!r}; "
                f"expected one of: {choices}."
            )
    selected = [entry for entry in artifact.entries if entry.language == language]
    if limit is not None:
        selected = selected[:limit]
    references: list[ReferencePronunciation] = []
    provenance = synthesizer.provenance
    for entry in selected:
        identity = f"{entry.concept_id}|{entry.term}|{language}|{provenance}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        slug = _safe_slug(entry.term)
        reference_id = f"kokoro-{slug}-{digest}"
        audio_path = output_dir / f"{reference_id}.wav"
        curated_phonemes = entry.pronunciations[0] if entry.pronunciations else None
        result = None
        if overwrite or not audio_path.is_file():
            result = synthesizer.synthesize(
                entry.term, audio_path, phonemes=curated_phonemes
            )
        references.append(
            ReferencePronunciation(
                reference_id=reference_id,
                audio_path=audio_path,
                concept_id=entry.concept_id,
                term=entry.term,
                language=entry.language,
                pronunciation_type="synthetic_tts",
                source="kokoro_local_synthetic_reference",
                concept_type="medical_term",
                synthetic=True,
                tts_model_id=_optional_string(provenance.get("tts_model_id")),
                tts_voice=_optional_string(provenance.get("tts_voice")),
                tts_language_code=_optional_string(provenance.get("tts_language_code")),
                tts_phonemes=(
                    result.phonemes if result is not None else curated_phonemes
                ),
                tts_sample_rate=_optional_int(provenance.get("tts_sample_rate")),
                tts_speed=_optional_float(provenance.get("tts_speed")),
                terminology_version=artifact.artifact_version,
                terminology_source_sha256=artifact.source_sha256,
            )
        )
    if not references:
        raise ValueError(f"Terminology artifact has no entries for language {language!r}.")
    return references


def write_reference_manifest(
    references: list[ReferencePronunciation], manifest_path: Path
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for reference in references:
        payload = reference.model_dump(mode="json")
        try:
            relative = os.path.relpath(reference.audio_path, manifest_path.parent)
            payload["audio_path"] = Path(relative).as_posix()
        except ValueError:
            payload["audio_path"] = reference.audio_path.as_posix()
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:48] or "term"


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
