from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jiwer import cer, wer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from medterm.matcher import MedicalTermMatcher
from medterm.models import MatchRequest, NBestHypothesis, SpanResult, TokenMetadata
from medterm.normalization import normalize_term


class GoldSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    span_text: str | None = None
    gold_term: str | None = None
    concept_id: str | None = None
    language: str | None = None
    script: str | None = None
    code_switch: bool = False
    dose_adjacent: bool = False
    risk_tier: str | None = None
    error_type: str | None = None
    term_category: str | None = None

    @model_validator(mode="after")
    def end_must_follow_start(self) -> GoldSpan:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str = "unversioned"
    case_id: str = ""
    text: str = Field(min_length=1)
    locale: str | None = None
    tags: list[str] = Field(default_factory=list)
    split: str = "test"
    source_kind: str = "synthetic"
    provenance: str | None = None
    audio_file: str | None = None
    tokens: list[TokenMetadata] = Field(default_factory=list)
    n_best: list[NBestHypothesis] = Field(default_factory=list)
    reference_transcript: str | None = None
    asr_hypothesis: str | None = None
    gold_spans: list[GoldSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_gold_offsets(self) -> EvaluationRecord:
        for span in self.gold_spans:
            if span.char_end > len(self.text):
                raise ValueError(
                    f"gold span [{span.char_start}, {span.char_end}) exceeds text length"
                )
            observed = self.text[span.char_start : span.char_end]
            if span.span_text is not None and observed != span.span_text:
                raise ValueError(
                    f"gold span_text {span.span_text!r} does not match text slice {observed!r}"
                )
        return self


@dataclass
class DetectionCounts:
    gold: int = 0
    predicted: int = 0
    exact_tp: int = 0
    overlap_tp: int = 0

    def report(self) -> dict[str, float | int]:
        return {
            "gold": self.gold,
            "predicted": self.predicted,
            "exact_true_positives": self.exact_tp,
            "exact_precision": _ratio(self.exact_tp, self.predicted),
            "exact_recall": _ratio(self.exact_tp, self.gold),
            "exact_f1": _f1(self.exact_tp, self.predicted, self.gold),
            "overlap_true_positives": self.overlap_tp,
            "overlap_false_positives": self.predicted - self.overlap_tp,
            "overlap_false_negatives": self.gold - self.overlap_tp,
            "overlap_precision": _ratio(self.overlap_tp, self.predicted),
            "overlap_recall": _ratio(self.overlap_tp, self.gold),
            "overlap_f1": _f1(self.overlap_tp, self.predicted, self.gold),
        }


@dataclass
class CorrectionCounts:
    gold: int = 0
    top_1_hits: int = 0
    top_5_hits: int = 0
    reciprocal_rank_sum: float = 0.0

    def report(self) -> dict[str, float | int]:
        return {
            "gold": self.gold,
            "accuracy_at_1": _ratio(self.top_1_hits, self.gold),
            "recall_at_5": _ratio(self.top_5_hits, self.gold),
            "mrr": round(self.reciprocal_rank_sum / self.gold, 4) if self.gold else 0.0,
        }


@dataclass
class EvaluationMetrics:
    records: int = 0
    detection: DetectionCounts = field(default_factory=DetectionCounts)
    correction: CorrectionCounts = field(default_factory=CorrectionCounts)
    records_with_false_positives: int = 0
    negative_records: int = 0
    negative_records_with_predictions: int = 0
    recall_at_1_hits: int = 0
    recall_at_5_hits: int = 0
    reciprocal_rank_sum: float = 0.0
    gold_concept_spans: int = 0
    review_spans: int = 0
    abstain_spans: int = 0
    auto_commit_count: int = 0

    def report(self) -> dict[str, Any]:
        detection = self.detection.report()
        predicted = self.detection.predicted
        ranked = self.gold_concept_spans
        report: dict[str, Any] = {
            "records": self.records,
            "gold_spans": self.detection.gold,
            "predicted_spans": predicted,
            # Compatibility aliases retained for existing report consumers.
            "span_detection_recall": detection["overlap_recall"],
            "recall_at_1": _ratio(self.recall_at_1_hits, ranked),
            "recall_at_5": _ratio(self.recall_at_5_hits, ranked),
            "mrr": round(self.reciprocal_rank_sum / ranked, 4) if ranked else 0.0,
            "false_positive_rate": _ratio(
                int(detection["overlap_false_positives"]), predicted
            ),
            "review_rate": _ratio(self.review_spans, predicted),
            "abstain_rate": _ratio(self.abstain_spans, predicted),
            "auto_commit_count": self.auto_commit_count,
            "detection": detection,
            "correction": self.correction.report(),
            "negative_record_false_positive_rate": _ratio(
                self.negative_records_with_predictions, self.negative_records
            ),
            "records_with_false_positives": self.records_with_false_positives,
            "record_false_positive_rate": _ratio(
                self.records_with_false_positives, self.records
            ),
        }
        return report


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _f1(true_positives: int, predicted: int, gold: int) -> float:
    denominator = predicted + gold
    return round(2 * true_positives / denominator, 4) if denominator else 0.0


def spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and left_end > right_start


def _match_spans(
    predictions: list[SpanResult], gold_spans: list[GoldSpan]
) -> tuple[list[tuple[int, int]], int]:
    """Return one-to-one overlap matches and the number with exact boundaries."""
    matches: list[tuple[int, int]] = []
    used_predictions: set[int] = set()
    used_gold: set[int] = set()

    for gold_index, gold in enumerate(gold_spans):
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in used_predictions:
                continue
            if (prediction.char_start, prediction.char_end) == (gold.char_start, gold.char_end):
                matches.append((prediction_index, gold_index))
                used_predictions.add(prediction_index)
                used_gold.add(gold_index)
                break
    exact_count = len(matches)

    adjacency: dict[int, list[int]] = {}
    for gold_index, gold in enumerate(gold_spans):
        if gold_index in used_gold:
            continue
        candidates = [
            prediction_index
            for prediction_index, prediction in enumerate(predictions)
            if prediction_index not in used_predictions
            and spans_overlap(
                prediction.char_start,
                prediction.char_end,
                gold.char_start,
                gold.char_end,
            )
        ]
        adjacency[gold_index] = sorted(
            candidates,
            key=lambda prediction_index: _intersection_over_union(
                predictions[prediction_index], gold
            ),
            reverse=True,
        )

    assigned_prediction: dict[int, int] = {}

    def assign(gold_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency.get(gold_index, []):
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous_gold = assigned_prediction.get(prediction_index)
            if previous_gold is None or assign(previous_gold, visited):
                assigned_prediction[prediction_index] = gold_index
                return True
        return False

    for gold_index in adjacency:
        assign(gold_index, set())
    matches.extend(
        (prediction_index, gold_index)
        for prediction_index, gold_index in assigned_prediction.items()
    )
    return matches, exact_count


def _intersection_over_union(prediction: SpanResult, gold: GoldSpan) -> float:
    intersection = max(
        0,
        min(prediction.char_end, gold.char_end) - max(prediction.char_start, gold.char_start),
    )
    union = max(prediction.char_end, gold.char_end) - min(prediction.char_start, gold.char_start)
    return intersection / union if union else 0.0


def _coerce_records(records: list[dict[str, Any] | EvaluationRecord]) -> list[EvaluationRecord]:
    validated: list[EvaluationRecord] = []
    seen_case_ids: set[str] = set()
    for position, raw_record in enumerate(records, start=1):
        if isinstance(raw_record, dict) and not raw_record.get("case_id"):
            raw_record = {**raw_record, "case_id": f"record-{position:05d}"}
        try:
            record = (
                raw_record
                if isinstance(raw_record, EvaluationRecord)
                else EvaluationRecord.model_validate(raw_record)
            )
        except ValidationError as exc:
            raise ValueError(f"invalid evaluation record {position}: {exc}") from exc
        if not record.case_id:
            record = record.model_copy(update={"case_id": f"record-{position:05d}"})
        if record.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {record.case_id}")
        seen_case_ids.add(record.case_id)
        validated.append(record)
    return validated


def evaluate_records(
    matcher: MedicalTermMatcher, records: list[dict[str, Any] | EvaluationRecord]
) -> dict[str, Any]:
    validated = _coerce_records(records)
    metrics = EvaluationMetrics(records=len(validated))
    references: list[str] = []
    hypotheses: list[str] = []
    slice_counts: dict[str, DetectionCounts] = defaultdict(DetectionCounts)
    slice_correction_counts: dict[str, CorrectionCounts] = defaultdict(CorrectionCounts)
    case_errors: list[dict[str, Any]] = []

    for record in validated:
        if record.reference_transcript is not None and record.asr_hypothesis is not None:
            references.append(record.reference_transcript)
            hypotheses.append(record.asr_hypothesis)
        response = matcher.match(
            MatchRequest(
                text=record.text,
                locale=record.locale,
                tokens=record.tokens,
                n_best=record.n_best,
                top_k=5,
            )
        )
        predictions = response.spans
        gold_spans = record.gold_spans
        matches, exact_count = _match_spans(predictions, gold_spans)
        matched_prediction_indices = {prediction for prediction, _ in matches}
        matched_gold_indices = {gold for _, gold in matches}

        metrics.detection.gold += len(gold_spans)
        metrics.detection.predicted += len(predictions)
        metrics.detection.exact_tp += exact_count
        metrics.detection.overlap_tp += len(matches)
        metrics.gold_concept_spans += sum(span.concept_id is not None for span in gold_spans)
        metrics.correction.gold += sum(span.gold_term is not None for span in gold_spans)
        metrics.review_spans += sum(span.decision == "review" for span in predictions)
        metrics.abstain_spans += sum(span.decision == "abstain" for span in predictions)
        metrics.auto_commit_count += sum(span.decision == "commit" for span in predictions)
        if not gold_spans:
            metrics.negative_records += 1
            metrics.negative_records_with_predictions += bool(predictions)
        if len(matched_prediction_indices) < len(predictions):
            metrics.records_with_false_positives += 1

        for prediction_index, gold_index in matches:
            gold = gold_spans[gold_index]
            prediction = predictions[prediction_index]
            if gold.gold_term is not None:
                language = gold.language or record.locale or "en"
                language = language.split("-")[0]
                gold_term = normalize_term(gold.gold_term, language)
                candidate_terms = [
                    normalize_term(candidate.term, language)
                    for candidate in prediction.candidates
                ]
                if gold_term in candidate_terms:
                    rank = candidate_terms.index(gold_term) + 1
                    metrics.correction.reciprocal_rank_sum += 1 / rank
                    metrics.correction.top_5_hits += 1
                    metrics.correction.top_1_hits += rank == 1
                    for tag in set(record.tags):
                        correction = slice_correction_counts[tag]
                        correction.reciprocal_rank_sum += 1 / rank
                        correction.top_5_hits += 1
                        correction.top_1_hits += rank == 1
            if gold.concept_id is None:
                continue
            candidate_ids = [
                candidate.concept_id for candidate in predictions[prediction_index].candidates
            ]
            if gold.concept_id in candidate_ids:
                rank = candidate_ids.index(gold.concept_id) + 1
                metrics.reciprocal_rank_sum += 1 / rank
                metrics.recall_at_5_hits += 1
                metrics.recall_at_1_hits += rank == 1

        for tag in set(record.tags):
            counts = slice_counts[tag]
            counts.gold += len(gold_spans)
            counts.predicted += len(predictions)
            counts.exact_tp += exact_count
            counts.overlap_tp += len(matches)
            slice_correction_counts[tag].gold += sum(
                span.gold_term is not None for span in gold_spans
            )

        false_positives = [
            {
                "char_start": prediction.char_start,
                "char_end": prediction.char_end,
                "span_text": prediction.span_text,
            }
            for prediction_index, prediction in enumerate(predictions)
            if prediction_index not in matched_prediction_indices
        ]
        false_negatives = [
            {
                "char_start": gold.char_start,
                "char_end": gold.char_end,
                "span_text": gold.span_text or record.text[gold.char_start : gold.char_end],
                "concept_id": gold.concept_id,
            }
            for gold_index, gold in enumerate(gold_spans)
            if gold_index not in matched_gold_indices
        ]
        boundary_mismatches = [
            {
                "gold": {
                    "char_start": gold_spans[gold_index].char_start,
                    "char_end": gold_spans[gold_index].char_end,
                    "span_text": gold_spans[gold_index].span_text
                    or record.text[
                        gold_spans[gold_index].char_start : gold_spans[gold_index].char_end
                    ],
                },
                "prediction": {
                    "char_start": predictions[prediction_index].char_start,
                    "char_end": predictions[prediction_index].char_end,
                    "span_text": predictions[prediction_index].span_text,
                },
            }
            for prediction_index, gold_index in matches
            if (
                predictions[prediction_index].char_start,
                predictions[prediction_index].char_end,
            )
            != (gold_spans[gold_index].char_start, gold_spans[gold_index].char_end)
        ]
        if false_positives or false_negatives or boundary_mismatches:
            case_errors.append(
                {
                    "case_id": record.case_id,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "boundary_mismatches": boundary_mismatches,
                }
            )

    report = metrics.report()
    versions = sorted({record.dataset_version for record in validated})
    report["dataset_versions"] = versions
    report["slices"] = {
        tag: counts.report() for tag, counts in sorted(slice_counts.items())
    }
    report["correction_slices"] = {
        tag: counts.report() for tag, counts in sorted(slice_correction_counts.items())
    }
    report["case_errors"] = case_errors
    if references:
        report["wer"] = round(float(wer(references, hypotheses)), 4)
        report["cer"] = round(float(cer(references, hypotheses)), 4)
    return report


def load_jsonl(path: Path) -> list[EvaluationRecord]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
    return _coerce_records(records)
