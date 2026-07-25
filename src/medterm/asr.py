from __future__ import annotations

from pathlib import Path
from typing import Protocol

from medterm.models import TokenMetadata


class ASRUnavailableError(RuntimeError):
    pass


class ASRResult(tuple):
    __slots__ = ()

    @property
    def text(self) -> str:
        return self[0]

    @property
    def tokens(self) -> list[TokenMetadata]:
        return self[1]


class ASRProvider(Protocol):
    def transcribe(self, audio_path: Path, locale: str | None = None) -> ASRResult: ...


class FasterWhisperASR:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio_path: Path, locale: str | None = None) -> ASRResult:
        if not self.model_name:
            raise ASRUnavailableError(
                "No ASR model is configured. Provide a transcript or set MEDTERM_WHISPER_MODEL."
            )
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ASRUnavailableError(
                "Install the 'asr' optional dependency to enable audio transcription."
            ) from exc
        if self._model is None:
            self._model = WhisperModel(self.model_name, device="auto", compute_type="auto")
        segments, _ = self._model.transcribe(
            str(audio_path), language=(locale or "").split("-")[0] or None, word_timestamps=True
        )
        words = []
        text_parts = []
        cursor = 0
        for segment in segments:
            for word in segment.words or []:
                token_text = word.word.strip()
                if not token_text:
                    continue
                if text_parts:
                    cursor += 1
                start = cursor
                text_parts.append(token_text)
                cursor += len(token_text)
                words.append(
                    TokenMetadata(
                        text=token_text,
                        char_start=start,
                        char_end=cursor,
                        confidence=word.probability,
                        start_time=word.start,
                        end_time=word.end,
                    )
                )
        return ASRResult((" ".join(text_parts), words))
