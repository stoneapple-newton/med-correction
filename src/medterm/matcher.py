from __future__ import annotations

import uuid
from datetime import UTC, datetime

from rapidfuzz.fuzz import ratio

from medterm.detection import DetectedSpan, SuspiciousSpanDetector
from medterm.language import LanguageIdentifier
from medterm.models import (
    CandidateResult,
    MatchRequest,
    MatchResponse,
    ScoreBreakdown,
    SpanResult,
)
from medterm.normalization import (
    detect_scripts,
    extract_medication_fields,
    normalize_term,
    normalize_text,
    strip_structured_tokens,
    tokenize_with_offsets,
)
from medterm.pronunciation import PhoneticSimilarity, PronunciationEngine, ngram_similarity
from medterm.terminology import DictionaryEntry, DictionaryIndex


class MedicalTermMatcher:
    def __init__(
        self,
        index: DictionaryIndex,
        language_identifier: LanguageIdentifier,
        review_threshold: float = 0.55,
        suspect_threshold: float = 0.30,
        force_review_margin: float = 0.12,
        max_span_tokens: int = 5,
    ) -> None:
        self.index = index
        self.language_identifier = language_identifier
        self.review_threshold = review_threshold
        self.force_review_margin = force_review_margin
        self.pronunciation = PronunciationEngine(index.curated)
        self.phonetic = PhoneticSimilarity()
        self.detector = SuspiciousSpanDetector(index, suspect_threshold, max_span_tokens)

    def match(self, request: MatchRequest) -> MatchResponse:
        request_id = request.request_id or str(uuid.uuid4())
        neighbor_scores = self._neighbor_scores(request)
        detected = self.detector.detect(request, neighbor_scores)
        spans = [self._match_span(request, request_id, span) for span in detected]
        return MatchResponse(
            request_id=request_id,
            normalized_text=normalize_text(request.text),
            artifact_version=self.index.artifact.artifact_version,
            auto_commit_enabled=False,
            spans=spans,
            created_at=datetime.now(UTC),
        )

    def _neighbor_scores(self, request: MatchRequest) -> dict[tuple[int, int], float]:
        tokens = tokenize_with_offsets(request.text)
        scores: dict[tuple[int, int], float] = {}
        for size in range(1, min(self.detector.max_tokens, len(tokens)) + 1):
            for start in range(len(tokens) - size + 1):
                end = start + size
                span = request.text[tokens[start].start : tokens[end - 1].end]
                language, _, _ = self.language_identifier.detect(span, request.locale)
                normalized = strip_structured_tokens(span) or normalize_term(span, language)
                pronunciations = [
                    item.value for item in self.pronunciation.generate(normalized, language)
                ]
                retrieved = self.index.retrieve(normalized, pronunciations, limit_per_channel=5)
                best = 0.0
                for entry_id, raw in retrieved.items():
                    entry = self.index.artifact.entries[entry_id]
                    orthographic = raw.get("orthographic", 0.0)
                    phonetic = self._best_phonetic(pronunciations, entry)[0]
                    best = max(best, 0.65 * phonetic + 0.35 * orthographic)
                scores[(start, end)] = best
        return scores

    def _match_span(self, request: MatchRequest, request_id: str, span: DetectedSpan) -> SpanResult:
        language, language_confidence, alternatives = self.language_identifier.detect(
            span.text, request.locale
        )
        scripts = detect_scripts(span.text)
        script = scripts[0] if len(scripts) == 1 else "+".join(scripts)
        structured = extract_medication_fields(span.text)
        normalized = strip_structured_tokens(span.text) or normalize_term(span.text, language)
        retrieval_languages = [language]
        if language_confidence < 0.5:
            retrieval_languages.extend(alternatives[:1])
        pronunciations = []
        for candidate_language in dict.fromkeys(retrieval_languages):
            pronunciations.extend(self.pronunciation.generate(normalized, candidate_language))
        pronunciations = list(
            {(item.value, item.source, item.alphabet): item for item in pronunciations}.values()
        )
        retrieved = self.index.retrieve(normalized, [item.value for item in pronunciations])
        candidates = [
            self._score_candidate(
                normalized,
                pronunciations=[item.value for item in pronunciations],
                languages=retrieval_languages,
                asr_score=1.0 - span.evidence.low_confidence,
                entry=self.index.artifact.entries[entry_id],
                raw=raw,
            )
            for entry_id, raw in retrieved.items()
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[: request.top_k]
        decision, reasons = self._decide(
            candidates,
            language,
            scripts,
            structured.strengths,
            language_confidence,
            bool(pronunciations),
        )
        context_start = max(0, span.start - 80)
        context_end = min(len(request.text), span.end + 80)
        audio_start, audio_end = self._audio_offsets(request, span)
        return SpanResult(
            span_id=str(uuid.uuid5(uuid.UUID(request_id), f"{span.start}:{span.end}"))
            if self._is_uuid(request_id)
            else str(uuid.uuid4()),
            span_text=span.text,
            char_start=span.start,
            char_end=span.end,
            context=request.text[context_start:context_end],
            language=language,
            language_confidence=round(language_confidence, 4),
            language_alternatives=alternatives,
            script=script,
            normalized_span=normalized,
            structured=structured,
            pronunciations=pronunciations,
            evidence=span.evidence,
            candidates=candidates,
            decision=decision,
            decision_reason=reasons,
            artifact_version=self.index.artifact.artifact_version,
            audio_start=audio_start,
            audio_end=audio_end,
        )

    def _score_candidate(
        self,
        normalized: str,
        pronunciations: list[str],
        languages: list[str],
        asr_score: float,
        entry: DictionaryEntry,
        raw: dict[str, object],
    ) -> CandidateResult:
        matched_alias = str(raw.get("matched_alias") or entry.normalized_term)
        orthographic = max(
            float(raw.get("orthographic", 0.0)),
            max(
                (ratio(normalized, alias) / 100 for alias in entry.normalized_aliases), default=0.0
            ),
        )
        feature, trigram = self._best_phonetic(pronunciations, entry)
        phonetic = 0.70 * feature + 0.30 * trigram
        alias = 1.0 if normalized in entry.normalized_aliases else 0.0
        if entry.language == languages[0]:
            language_score = 1.0
        elif entry.language in languages[1:]:
            language_score = 0.7
        else:
            language_score = 0.4 if languages[0] == "und" else 0.0
        score = (
            0.40 * phonetic
            + 0.20 * orthographic
            + 0.10 * alias
            + 0.10 * language_score
            + 0.10 * asr_score
            + 0.10 * entry.prior
        )
        breakdown = ScoreBreakdown(
            phonetic=round(phonetic, 4),
            orthographic=round(orthographic, 4),
            alias=alias,
            language=language_score,
            asr=round(asr_score, 4),
            prior=entry.prior,
            phonetic_feature=round(feature, 4),
            ipa_ngram=round(trigram, 4),
        )
        return CandidateResult(
            concept_id=entry.concept_id,
            term=entry.term,
            matched_alias=matched_alias,
            language=entry.language,
            risk_tier=entry.risk_tier,
            score=round(score, 4),
            score_breakdown=breakdown,
            provenance=entry.provenance,
        )

    def _best_phonetic(
        self, pronunciations: list[str], entry: DictionaryEntry
    ) -> tuple[float, float]:
        if not pronunciations or not entry.pronunciation_forms:
            return 0.0, 0.0
        pairs = [
            (
                self.phonetic.feature_similarity(query, candidate),
                ngram_similarity(query, candidate),
            )
            for query in pronunciations
            for candidate in entry.pronunciation_forms
        ]
        return max(pairs, key=lambda pair: 0.7 * pair[0] + 0.3 * pair[1])

    def _decide(
        self,
        candidates: list[CandidateResult],
        language: str,
        scripts: list[str],
        strengths: list[str],
        language_confidence: float,
        has_pronunciation: bool,
    ) -> tuple[str, list[str]]:
        if not has_pronunciation:
            return "abstain", ["pronunciation_failed"]
        if language == "und" or scripts == ["Zyyy"]:
            return "abstain", ["language_or_script_unresolved"]
        if not candidates or candidates[0].score < self.review_threshold:
            return "abstain", ["below_review_threshold"]
        reasons = ["candidate_above_review_threshold"]
        margin = candidates[0].score - (candidates[1].score if len(candidates) > 1 else 0.0)
        if margin < self.force_review_margin:
            reasons.append("low_margin")
        if candidates[0].risk_tier.casefold() in {"high", "lasa", "critical"}:
            reasons.append("high_risk_medication")
        if len(scripts) > 1:
            reasons.append("mixed_script")
        if language_confidence < 0.5:
            reasons.append("language_uncertain")
        if strengths:
            reasons.append("dose_adjacent")
        return "review", reasons

    @staticmethod
    def _audio_offsets(
        request: MatchRequest, span: DetectedSpan
    ) -> tuple[float | None, float | None]:
        matching = [
            token
            for token in request.tokens
            if token.char_start is not None
            and token.char_end is not None
            and token.char_start < span.end
            and token.char_end > span.start
        ]
        starts = [item.start_time for item in matching if item.start_time is not None]
        ends = [item.end_time for item in matching if item.end_time is not None]
        return (min(starts) if starts else None, max(ends) if ends else None)

    @staticmethod
    def _is_uuid(value: str) -> bool:
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            return False
