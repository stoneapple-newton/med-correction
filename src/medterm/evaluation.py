from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jiwer import cer, wer

from medterm.matcher import MedicalTermMatcher
from medterm.models import MatchRequest


@dataclass
class EvaluationMetrics:
    records: int = 0
    gold_spans: int = 0
    detected_gold_spans: int = 0
    recall_at_1_hits: int = 0
    recall_at_5_hits: int = 0
    reciprocal_rank_sum: float = 0.0
    predicted_spans: int = 0
    false_positive_spans: int = 0
    review_spans: int = 0
    abstain_spans: int = 0

    def report(self) -> dict[str, float | int]:
        gold = max(self.gold_spans, 1)
        predicted = max(self.predicted_spans, 1)
        return {
            "records": self.records,
            "gold_spans": self.gold_spans,
            "predicted_spans": self.predicted_spans,
            "span_detection_recall": round(self.detected_gold_spans / gold, 4),
            "recall_at_1": round(self.recall_at_1_hits / gold, 4),
            "recall_at_5": round(self.recall_at_5_hits / gold, 4),
            "mrr": round(self.reciprocal_rank_sum / gold, 4),
            "false_positive_rate": round(self.false_positive_spans / predicted, 4),
            "review_rate": round(self.review_spans / predicted, 4),
            "abstain_rate": round(self.abstain_spans / predicted, 4),
        }


def spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and left_end > right_start


def evaluate_records(
    matcher: MedicalTermMatcher, records: list[dict[str, Any]]
) -> dict[str, float | int]:
    metrics = EvaluationMetrics(records=len(records))
    references: list[str] = []
    hypotheses: list[str] = []
    for record in records:
        if (
            record.get("reference_transcript") is not None
            and record.get("asr_hypothesis") is not None
        ):
            references.append(str(record["reference_transcript"]))
            hypotheses.append(str(record["asr_hypothesis"]))
        response = matcher.match(
            MatchRequest(
                text=record["text"],
                locale=record.get("locale"),
                tokens=record.get("tokens", []),
                n_best=record.get("n_best", []),
                top_k=5,
            )
        )
        metrics.predicted_spans += len(response.spans)
        metrics.review_spans += sum(span.decision == "review" for span in response.spans)
        metrics.abstain_spans += sum(span.decision == "abstain" for span in response.spans)
        matched_prediction_ids: set[str] = set()
        for gold in record.get("gold_spans", []):
            metrics.gold_spans += 1
            overlapping = [
                span
                for span in response.spans
                if spans_overlap(
                    span.char_start,
                    span.char_end,
                    int(gold["char_start"]),
                    int(gold["char_end"]),
                )
            ]
            if not overlapping:
                continue
            metrics.detected_gold_spans += 1
            best = max(
                overlapping, key=lambda span: span.candidates[0].score if span.candidates else 0
            )
            matched_prediction_ids.add(best.span_id)
            candidate_ids = [candidate.concept_id for candidate in best.candidates]
            concept_id = gold["concept_id"]
            if concept_id in candidate_ids:
                rank = candidate_ids.index(concept_id) + 1
                metrics.reciprocal_rank_sum += 1 / rank
                metrics.recall_at_5_hits += 1
                metrics.recall_at_1_hits += rank == 1
        metrics.false_positive_spans += sum(
            span.span_id not in matched_prediction_ids for span in response.spans
        )
    report = metrics.report()
    if references:
        report["wer"] = round(float(wer(references, hypotheses)), 4)
        report["cer"] = round(float(cer(references, hypotheses)), 4)
    report["auto_commit_count"] = 0
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
