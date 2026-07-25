from __future__ import annotations

from dataclasses import dataclass

from medterm.models import MatchRequest, SpanEvidence
from medterm.normalization import TextToken, detect_scripts, normalize_term, tokenize_with_offsets
from medterm.terminology import DictionaryIndex


@dataclass(frozen=True)
class DetectedSpan:
    text: str
    start: int
    end: int
    token_start: int
    token_end: int
    evidence: SpanEvidence


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
                mixed_script = 1.0 if len(detect_scripts(span_text)) > 1 else 0.0
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
                if (
                    risk >= self.threshold
                    and has_trigger
                    and not (exact and low_confidence == 0 and not mixed_script)
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
        return self._select_nonredundant(spans, neighbor_scores)

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
        ranked = sorted(
            spans,
            key=lambda span: (
                neighbor_scores.get((span.token_start, span.token_end), 0),
                span.evidence.risk_score,
                span.token_end - span.token_start,
            ),
            reverse=True,
        )
        selected: list[DetectedSpan] = []
        for span in ranked:
            if any(span.start < other.end and span.end > other.start for other in selected):
                continue
            selected.append(span)
        return sorted(selected, key=lambda span: span.start)
