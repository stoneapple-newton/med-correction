from __future__ import annotations

import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status

from medterm import __version__
from medterm.asr import ASRUnavailableError, FasterWhisperASR
from medterm.audit import AuditStore
from medterm.config import Settings, get_settings
from medterm.language import LanguageIdentifier
from medterm.matcher import MedicalTermMatcher
from medterm.models import (
    DictionarySearchResult,
    MatchRequest,
    MatchResponse,
    ReviewRecord,
    ReviewRequest,
)
from medterm.terminology import DictionaryIndex, build_artifact, load_artifact


class AppServices:
    def __init__(self, settings: Settings) -> None:
        if not settings.artifact_path.exists():
            if not settings.source_terms_path.exists():
                raise FileNotFoundError(
                    f"Dictionary artifact and source are missing: {settings.artifact_path}"
                )
            build_artifact(settings.source_terms_path, settings.artifact_path)
        artifact = load_artifact(settings.artifact_path)
        self.index = DictionaryIndex(artifact)
        self.matcher = MedicalTermMatcher(
            self.index,
            LanguageIdentifier(settings.fasttext_model_path),
            review_threshold=settings.review_threshold,
            suspect_threshold=settings.suspect_threshold,
            force_review_margin=settings.force_review_margin,
            max_span_tokens=settings.max_span_tokens,
        )
        self.audit = AuditStore(settings.audit_db_path)
        self.asr = FasterWhisperASR(settings.whisper_model)
        self.settings = settings


@lru_cache
def get_services() -> AppServices:
    return AppServices(get_settings())


def create_app() -> FastAPI:
    app = FastAPI(
        title="Medical Term Review API",
        version=__version__,
        summary="Review-first medical term span detection and dictionary matching",
    )

    @app.get("/health")
    def health(services: AppServices = Depends(get_services)) -> dict[str, str | int | bool]:
        return {
            "status": "ok",
            "version": __version__,
            "artifact_version": services.index.artifact.artifact_version,
            "dictionary_entries": len(services.index.artifact.entries),
            "auto_commit_enabled": False,
        }

    @app.post("/v1/match", response_model=MatchResponse)
    def match_text(
        request: MatchRequest, services: AppServices = Depends(get_services)
    ) -> MatchResponse:
        response = services.matcher.match(request)
        services.audit.save_match(response)
        return response

    @app.get("/v1/dictionary/search", response_model=list[DictionarySearchResult])
    def search_dictionary(
        q: str = Query(min_length=1, max_length=200),
        limit: int = Query(default=10, ge=1, le=50),
        services: AppServices = Depends(get_services),
    ) -> list[DictionarySearchResult]:
        return [
            DictionarySearchResult(
                concept_id=entry.concept_id,
                term=entry.term,
                aliases=entry.aliases,
                language=entry.language,
                risk_tier=entry.risk_tier,
                provenance=entry.provenance,
                similarity=round(score, 4),
            )
            for entry, score in services.index.search(q, limit)
        ]

    @app.post("/v1/match/audio", response_model=MatchResponse)
    def match_audio(
        audio: UploadFile = File(...),
        transcript: str | None = Form(default=None),
        locale: str | None = Form(default=None),
        top_k: int = Form(default=5, ge=1, le=20),
        services: AppServices = Depends(get_services),
    ) -> MatchResponse:
        request_id = str(uuid.uuid4())
        suffix = Path(audio.filename or "audio.bin").suffix or ".bin"
        services.settings.audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = services.settings.audio_dir / f"{request_id}{suffix}"
        with audio_path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
        tokens = []
        if not transcript:
            try:
                asr_result = services.asr.transcribe(audio_path, locale)
                transcript, tokens = asr_result.text, asr_result.tokens
            except ASRUnavailableError as exc:
                audio_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
                ) from exc
        response = services.matcher.match(
            MatchRequest(
                text=transcript,
                locale=locale,
                tokens=tokens,
                top_k=top_k,
                request_id=request_id,
            )
        )
        services.audit.save_match(response)
        return response

    @app.get("/v1/matches/{request_id}", response_model=MatchResponse)
    def get_match(request_id: str, services: AppServices = Depends(get_services)) -> MatchResponse:
        response = services.audit.get_match(request_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Match request not found")
        return response

    @app.post(
        "/v1/matches/{request_id}/spans/{span_id}/review",
        response_model=ReviewRecord,
        status_code=201,
    )
    def review_span(
        request_id: str,
        span_id: str,
        review: ReviewRequest,
        services: AppServices = Depends(get_services),
    ) -> ReviewRecord:
        response = services.audit.get_match(request_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Match request not found")
        span = next((item for item in response.spans if item.span_id == span_id), None)
        if span is None:
            raise HTTPException(status_code=404, detail="Span not found")
        if review.concept_id and review.concept_id not in {
            item.concept_id for item in span.candidates
        }:
            raise HTTPException(
                status_code=422, detail="concept_id is not a candidate for this span"
            )
        return services.audit.save_review(request_id, span_id, review, response.artifact_version)

    @app.get("/v1/matches/{request_id}/reviews", response_model=list[ReviewRecord])
    def list_reviews(
        request_id: str, services: AppServices = Depends(get_services)
    ) -> list[ReviewRecord]:
        if services.audit.get_match(request_id) is None:
            raise HTTPException(status_code=404, detail="Match request not found")
        return services.audit.list_reviews(request_id)

    return app


app = create_app()
