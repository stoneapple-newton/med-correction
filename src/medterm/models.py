from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TokenMetadata(BaseModel):
    text: str
    char_start: int | None = None
    char_end: int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)


class NBestHypothesis(BaseModel):
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class MatchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    locale: str | None = None
    speaker_id: str | None = None
    tokens: list[TokenMetadata] = Field(default_factory=list)
    n_best: list[NBestHypothesis] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    request_id: str | None = None


class StructuredMedication(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    dosage_forms: list[str] = Field(default_factory=list)


class Pronunciation(BaseModel):
    value: str
    alphabet: Literal["ipa", "arpabet", "fallback"] = "ipa"
    source: Literal["curated", "cmudict", "epitran", "fallback"]


class ScoreBreakdown(BaseModel):
    phonetic: float
    orthographic: float
    alias: float
    language: float
    asr: float
    prior: float
    phonetic_feature: float
    ipa_ngram: float
    mutation: float = 0.0
    scoring_profile: Literal["phonetic_first", "graphemic_fallback"] = "phonetic_first"


class CandidateResult(BaseModel):
    concept_id: str
    term: str
    matched_alias: str
    language: str
    risk_tier: str
    score: float
    score_breakdown: ScoreBreakdown
    provenance: str
    source_vocabulary: str = "unspecified"
    source_release: str = "unspecified"
    licensing_status: str = "unverified"
    review_status: str = "unreviewed"
    retrieval_channels: list[str] = Field(default_factory=list)


class SpanEvidence(BaseModel):
    oov: float
    low_confidence: float
    nbest_disagreement: float
    boundary_risk: float
    mixed_script: float
    neighbor_hint: float
    risk_score: float


class SpanResult(BaseModel):
    span_id: str
    span_text: str
    char_start: int
    char_end: int
    context: str
    language: str
    language_confidence: float
    language_alternatives: list[str] = Field(default_factory=list)
    script: str
    normalized_span: str
    structured: StructuredMedication
    pronunciations: list[Pronunciation]
    evidence: SpanEvidence
    candidates: list[CandidateResult]
    decision: Literal["abstain", "review", "commit"]
    decision_reason: list[str]
    artifact_version: str
    audio_start: float | None = None
    audio_end: float | None = None


class MatchResponse(BaseModel):
    request_id: str
    normalized_text: str
    artifact_version: str
    auto_commit_enabled: bool = False
    spans: list[SpanResult]
    created_at: datetime


class ReviewRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=100)
    action: Literal["accept_candidate", "keep_original", "unresolved"]
    concept_id: str | None = None
    note: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def require_concept_for_accept(self) -> ReviewRequest:
        if self.action == "accept_candidate" and not self.concept_id:
            raise ValueError("concept_id is required when accepting a candidate")
        return self


class ReviewRecord(BaseModel):
    review_id: str
    request_id: str
    span_id: str
    reviewer_id: str
    action: str
    concept_id: str | None
    note: str | None
    artifact_version: str
    created_at: datetime


class DictionarySearchResult(BaseModel):
    concept_id: str
    term: str
    aliases: list[str]
    language: str
    risk_tier: str
    provenance: str
    similarity: float
