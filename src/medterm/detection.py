from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key

from medterm.models import MatchRequest, SpanEvidence
from medterm.normalization import (
    TextToken,
    detect_scripts,
    has_suspicious_mixed_scripts,
    normalize_term,
    tokenize_with_offsets,
)
from medterm.terminology import DictionaryIndex


@dataclass(frozen=True)
class DetectedSpan:
    text: str
    start: int
    end: int
    token_start: int
    token_end: int
    evidence: SpanEvidence
    retrieval_hint: float = 0.0


class SuspiciousSpanDetector:
    def __init__(
        self, index: DictionaryIndex, threshold: float = 0.30, max_tokens: int = 5
    ) -> None:
        self.index = index
        self.threshold = threshold
        self.max_tokens = max_tokens

    def detect(
        self, request: MatchRequest, neighbor_scores: dict[tuple[int, int], float]
    ) -> list[DetectedSpan]:
        tokens = tokenize_with_offsets(request.text)
        metadata_confidence = self._token_confidences(request, tokens)
        spans: list[DetectedSpan] = []
        safe_exact_windows: list[tuple[int, int]] = []
        for size in range(1, min(self.max_tokens, len(tokens)) + 1):
            for start in range(len(tokens) - size + 1):
                end = start + size
                text = request.text[tokens[start].start : tokens[end - 1].end]
                if self.index.is_exact(normalize_term(text)):
                    safe_exact_windows.append((start, end))
        for window_size in range(1, min(self.max_tokens, len(tokens)) + 1):
            for token_start in range(0, len(tokens) - window_size + 1):
                token_end = token_start + window_size
                first, last = tokens[token_start], tokens[token_end - 1]
                span_text = request.text[first.start : last.end]
                normalized = normalize_term(span_text)
                exact = self.index.is_exact(normalized)
                neighbor = neighbor_scores.get((token_start, token_end), 0.0)
                confidence_values = metadata_confidence[token_start:token_end]
                low_confidence = (
                    max(0.0, 1.0 - sum(confidence_values) / len(confidence_values))
                    if confidence_values
                    else 0.0
                )
                disagreement = self._nbest_disagreement(request, span_text)
                mixed_script = 1.0 if has_suspicious_mixed_scripts(span_text) else 0.0
                boundary = self._boundary_risk(tokens, token_start, token_end, neighbor)
                oov = 0.0 if exact else 1.0
                neighbor_hint = 1.0 if neighbor >= 0.52 else 0.0
                risk = (
                    0.25 * oov
                    + 0.20 * low_confidence
                    + 0.15 * disagreement
                    + 0.15 * boundary
                    + 0.15 * mixed_script
                    + 0.10 * neighbor_hint
                )
                evidence = SpanEvidence(
                    oov=oov,
                    low_confidence=round(low_confidence, 4),
                    nbest_disagreement=round(disagreement, 4),
                    boundary_risk=boundary,
                    mixed_script=mixed_script,
                    neighbor_hint=neighbor_hint,
                    risk_score=round(risk, 4),
                )
                # Plain OOV text is insufficient: require a terminology neighbor or ASR evidence.
                has_trigger = (
                    neighbor_hint or low_confidence >= 0.25 or disagreement >= 0.5 or mixed_script
                )
                contains_safe_exact = any(
                    token_start <= exact_start
                    and exact_end <= token_end
                    and (exact_start, exact_end) != (token_start, token_end)
                    for exact_start, exact_end in safe_exact_windows
                )
                if (
                    risk >= self.threshold
                    and has_trigger
                    and not (exact and low_confidence == 0 and not mixed_script)
                    and not (
                        contains_safe_exact
                        and low_confidence == 0
                        and disagreement < 0.5
                        and not mixed_script
                    )
                ):
                    spans.append(
                        DetectedSpan(
                            text=span_text,
                            start=first.start,
                            end=last.end,
                            token_start=token_start,
                            token_end=token_end,
                            evidence=evidence,
                        )
                    )
        spans.extend(self._anchored_low_confidence_spans(request, tokens, neighbor_scores))
        spans.extend(self._approximate_asian_spans(request))
        return self._select_nonredundant(spans, neighbor_scores)

    def _anchored_low_confidence_spans(
        self,
        request: MatchRequest,
        tokens: list[TextToken],
        neighbor_scores: dict[tuple[int, int], float],
    ) -> list[DetectedSpan]:
        anchored: list[DetectedSpan] = []
        for item in request.tokens:
            if (
                item.confidence is None
                or item.confidence > 0.75
                or item.char_start is None
                or item.char_end is None
                or item.char_start >= item.char_end
                or item.char_end > len(request.text)
            ):
                continue
            overlaps = [
                index
                for index, token in enumerate(tokens)
                if token.start < item.char_end and token.end > item.char_start
            ]
            if not overlaps:
                continue
            token_start, token_end = min(overlaps), max(overlaps) + 1
            span_text = request.text[item.char_start : item.char_end]
            mixed_script = 1.0 if has_suspicious_mixed_scripts(span_text) else 0.0
            low_confidence = 1.0 - item.confidence
            exact = self.index.is_exact(normalize_term(span_text))
            neighbor = neighbor_scores.get((token_start, token_end), 0.0)
            risk = max(
                self.threshold,
                0.25 * (not exact) + 0.20 * low_confidence + 0.15 * mixed_script,
            )
            anchored.append(
                DetectedSpan(
                    text=span_text,
                    start=item.char_start,
                    end=item.char_end,
                    token_start=token_start,
                    token_end=token_end,
                    evidence=SpanEvidence(
                        oov=0.0 if exact else 1.0,
                        low_confidence=round(low_confidence, 4),
                        nbest_disagreement=0.0,
                        boundary_risk=0.0,
                        mixed_script=mixed_script,
                        neighbor_hint=1.0 if neighbor >= 0.52 else 0.0,
                        risk_score=round(risk, 4),
                    ),
                    retrieval_hint=neighbor,
                )
            )
        return anchored

    def _approximate_asian_spans(self, request: MatchRequest) -> list[DetectedSpan]:
        languages = self._asian_languages(request)
        matches = self.index.find_approximate_asian_spans(request.text, languages)
        return [
            DetectedSpan(
                text=request.text[match.start : match.end],
                start=match.start,
                end=match.end,
                token_start=-(match.start + 1),
                token_end=-(match.end + 1),
                evidence=SpanEvidence(
                    oov=1.0,
                    low_confidence=0.0,
                    nbest_disagreement=0.0,
                    boundary_risk=1.0,
                    mixed_script=(
                        1.0
                        if has_suspicious_mixed_scripts(
                            request.text[match.start : match.end]
                        )
                        else 0.0
                    ),
                    neighbor_hint=1.0,
                    risk_score=round(0.25 + 0.15 + 0.10 * match.score, 4),
                ),
                retrieval_hint=match.score,
            )
            for match in matches
        ]

    @staticmethod
    def _asian_languages(request: MatchRequest) -> list[str]:
        if request.locale:
            return [request.locale.split("-")[0].split("_")[0].casefold()]
        scripts = detect_scripts(request.text)
        mapping = {"Hani": "zh", "Hira": "ja", "Kana": "ja", "Hang": "ko", "Deva": "hi"}
        return list(dict.fromkeys(mapping[script] for script in scripts if script in mapping))

    @staticmethod
    def _token_confidences(request: MatchRequest, tokens: list[TextToken]) -> list[float]:
        if not request.tokens:
            return []
        if all(
            item.char_start is not None and item.char_end is not None for item in request.tokens
        ):
            values = []
            for token in tokens:
                overlaps = [
                    item.confidence
                    for item in request.tokens
                    if item.confidence is not None
                    and item.char_start is not None
                    and item.char_end is not None
                    and item.char_start < token.end
                    and item.char_end > token.start
                ]
                values.append(sum(overlaps) / len(overlaps) if overlaps else 1.0)
            return values
        values = [
            item.confidence if item.confidence is not None else 1.0 for item in request.tokens
        ]
        return values[: len(tokens)] + [1.0] * max(0, len(tokens) - len(values))

    @staticmethod
    def _nbest_disagreement(request: MatchRequest, span_text: str) -> float:
        if not request.n_best:
            return 0.0
        normalized = normalize_term(span_text)
        appearances = sum(normalized in normalize_term(item.text) for item in request.n_best)
        return 1.0 - appearances / len(request.n_best)

    @staticmethod
    def _boundary_risk(
        tokens: list[TextToken], token_start: int, token_end: int, neighbor_score: float
    ) -> float:
        if neighbor_score < 0.52:
            return 0.0
        # Multi-token ASR fragments matching a single dictionary term are a common boundary anomaly.
        return 1.0 if token_end - token_start > 1 else 0.0

    @staticmethod
    def _select_nonredundant(
        spans: list[DetectedSpan], neighbor_scores: dict[tuple[int, int], float]
    ) -> list[DetectedSpan]:
        def retrieval_score(span: DetectedSpan) -> float:
            return max(
                span.retrieval_hint,
                neighbor_scores.get((span.token_start, span.token_end), 0),
            )

        def compare(left: DetectedSpan, right: DetectedSpan) -> int:
            left_score, right_score = retrieval_score(left), retrieval_score(right)
            if abs(left_score - right_score) > 0.04:
                return -1 if left_score > right_score else 1
            left_length, right_length = left.end - left.start, right.end - right.start
            if left_length != right_length:
                return -1 if left_length < right_length else 1
            if left.evidence.risk_score != right.evidence.risk_score:
                return -1 if left.evidence.risk_score > right.evidence.risk_score else 1
            return 0

        ranked = sorted(spans, key=cmp_to_key(compare))
        selected: list[DetectedSpan] = []
        for span in ranked:
            if any(span.start < other.end and span.end > other.start for other in selected):
                continue
            selected.append(span)
        return sorted(selected, key=lambda span: span.start)
