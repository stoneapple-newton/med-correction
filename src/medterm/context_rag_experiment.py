from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rapidfuzz.fuzz import ratio

from medterm.normalization import normalize_term


class ContextTermRecord(BaseModel):
    """Synthetic, language-specific knowledge used only by the offline experiment."""

    model_config = ConfigDict(extra="forbid")

    concept_id: str
    term: str
    language: str
    context_terms: list[str] = Field(default_factory=list)
    provenance: str


class ContextRagCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    language: str
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    span_text: str
    gold_concept_id: str | None = None
    negative_control: bool = False

    @model_validator(mode="after")
    def validate_case(self) -> ContextRagCase:
        if self.char_end > len(self.text):
            raise ValueError("candidate span exceeds source text")
        if self.text[self.char_start : self.char_end] != self.span_text:
            raise ValueError("span_text does not match the original-text offsets")
        if self.negative_control == (self.gold_concept_id is not None):
            raise ValueError("positive cases need gold_concept_id; negative controls must omit it")
        return self


class ContextRagDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    records: list[ContextTermRecord]
    cases: list[ContextRagCase]


class RankedContextCandidate(BaseModel):
    concept_id: str
    term: str
    language: str
    score: float
    surface_score: float
    context_score: float
    provenance: str


@dataclass(frozen=True)
class _ArmSettings:
    mode: Literal["baseline", "context_rag"]
    surface_floor: float
    context_retrieval_floor: float


def load_context_rag_dataset(path: Path) -> ContextRagDataset:
    return ContextRagDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _remove_span(text: str, start: int, end: int) -> str:
    # Removing the observed span prevents the supposed context signal from leaking the answer.
    return f"{text[:start]} {text[end:]}"


class DummyContextRetriever:
    """A transparent lexical stand-in for language-specific database RAG.

    It deliberately has no generative component. Records are filtered by language, retrieval
    evidence is exposed, and every output remains a controlled-dictionary candidate.
    """

    def __init__(self, records: list[ContextTermRecord]) -> None:
        self.records = records

    @staticmethod
    def _context_score(context: str, record: ContextTermRecord) -> float:
        normalized_context = normalize_term(context, record.language)
        if not record.context_terms:
            return 0.0
        hits = sum(
            normalize_term(item, record.language) in normalized_context
            for item in record.context_terms
        )
        return min(1.0, hits / min(2, len(record.context_terms)))

    def rank(
        self,
        case: ContextRagCase,
        *,
        mode: Literal["baseline", "context_rag"],
        top_k: int = 5,
    ) -> list[RankedContextCandidate]:
        settings = (
            _ArmSettings("baseline", surface_floor=0.85, context_retrieval_floor=1.1)
            if mode == "baseline"
            else _ArmSettings("context_rag", surface_floor=0.85, context_retrieval_floor=1.0)
        )
        observed = normalize_term(case.text[case.char_start : case.char_end], case.language)
        context = _remove_span(case.text, case.char_start, case.char_end)
        ranked: list[RankedContextCandidate] = []
        for record in self.records:
            if record.language != case.language:
                continue
            surface = ratio(observed, normalize_term(record.term, record.language)) / 100
            context_score = self._context_score(context, record)
            retrieved = surface >= settings.surface_floor or (
                mode == "context_rag"
                and surface >= 0.25
                and context_score >= settings.context_retrieval_floor
            )
            if not retrieved:
                continue
            score = surface if mode == "baseline" else 0.65 * surface + 0.35 * context_score
            ranked.append(
                RankedContextCandidate(
                    concept_id=record.concept_id,
                    term=record.term,
                    language=record.language,
                    score=round(score, 4),
                    surface_score=round(surface, 4),
                    context_score=round(context_score, 4),
                    provenance=record.provenance,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.concept_id))
        return ranked[:top_k]


def _evaluate_arm(
    retriever: DummyContextRetriever,
    cases: list[ContextRagCase],
    mode: Literal["baseline", "context_rag"],
) -> dict[str, Any]:
    positives = [case for case in cases if not case.negative_control]
    negatives = [case for case in cases if case.negative_control]
    top_1_hits = 0
    top_5_hits = 0
    reciprocal_rank_sum = 0.0
    correct_margins: list[float] = []
    gold_selection_margins: list[float] = []
    negative_predictions = 0
    case_results: list[dict[str, Any]] = []
    for case in cases:
        candidates = retriever.rank(case, mode=mode)
        ids = [candidate.concept_id for candidate in candidates]
        rank = (
            ids.index(case.gold_concept_id) + 1
            if case.gold_concept_id is not None and case.gold_concept_id in ids
            else None
        )
        if rank == 1:
            top_1_hits += 1
            runner_up = candidates[1].score if len(candidates) > 1 else 0.0
            correct_margins.append(candidates[0].score - runner_up)
        if rank is None:
            gold_selection_margins.append(-1.0)
        else:
            gold_score = candidates[rank - 1].score
            best_other = max(
                (
                    candidate.score
                    for index, candidate in enumerate(candidates)
                    if index != rank - 1
                ),
                default=0.0,
            )
            gold_selection_margins.append(gold_score - best_other)
        if rank is not None and rank <= 5:
            top_5_hits += 1
            reciprocal_rank_sum += 1 / rank
        if case.negative_control and candidates:
            negative_predictions += 1
        case_results.append(
            {
                "case_id": case.case_id,
                "gold_rank": rank,
                "decision": "review" if candidates else "abstain",
                "candidates": [candidate.model_dump() for candidate in candidates],
            }
        )
    positive_count = len(positives)
    negative_count = len(negatives)
    return {
        "positive_cases": positive_count,
        "negative_controls": negative_count,
        "correction_sensitivity_at_5": round(top_5_hits / positive_count, 4)
        if positive_count
        else 0.0,
        "selection_accuracy_at_1": round(top_1_hits / positive_count, 4)
        if positive_count
        else 0.0,
        "mean_reciprocal_rank": round(reciprocal_rank_sum / positive_count, 4)
        if positive_count
        else 0.0,
        "mean_correct_top_1_margin": round(sum(correct_margins) / len(correct_margins), 4)
        if correct_margins
        else 0.0,
        "mean_gold_selection_margin": round(
            sum(gold_selection_margins) / len(gold_selection_margins), 4
        )
        if gold_selection_margins
        else 0.0,
        "negative_control_selectivity": round(1 - negative_predictions / negative_count, 4)
        if negative_count
        else 0.0,
        "auto_commit_count": 0,
        "cases": case_results,
    }


def run_context_rag_experiment(dataset: ContextRagDataset) -> dict[str, Any]:
    retriever = DummyContextRetriever(dataset.records)
    baseline = _evaluate_arm(retriever, dataset.cases, "baseline")
    context_rag = _evaluate_arm(retriever, dataset.cases, "context_rag")
    comparable_metrics = (
        "correction_sensitivity_at_5",
        "selection_accuracy_at_1",
        "mean_reciprocal_rank",
        "mean_gold_selection_margin",
        "negative_control_selectivity",
    )
    languages = sorted({case.language for case in dataset.cases})
    language_slices: dict[str, dict[str, Any]] = {}
    for language in languages:
        language_cases = [case for case in dataset.cases if case.language == language]
        language_slices[language] = {}
        for mode in ("baseline", "context_rag"):
            arm = _evaluate_arm(retriever, language_cases, mode)
            language_slices[language][mode] = {
                key: value for key, value in arm.items() if key != "cases"
            }
    return {
        "dataset_version": dataset.dataset_version,
        "definitions": {
            "correction_sensitivity_at_5": "fraction of positive spans with gold in top five",
            "selection_accuracy_at_1": "fraction of positive spans with gold ranked first",
            "mean_correct_top_1_margin": "mean first-minus-second score when gold ranks first",
            "mean_gold_selection_margin": (
                "mean gold-minus-best-alternative score; unretrieved gold is scored -1"
            ),
            "negative_control_selectivity": "fraction of negative controls that abstain",
        },
        "baseline": baseline,
        "context_rag": context_rag,
        "language_slices": language_slices,
        "delta": {
            metric: round(context_rag[metric] - baseline[metric], 4)
            for metric in comparable_metrics
        },
        "safety": {
            "source_text_mutated": False,
            "controlled_candidates_only": True,
            "auto_commit_enabled": False,
            "intended_use": "offline synthetic reviewer-assistance experiment only",
        },
    }


def write_context_rag_report(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    report = run_context_rag_experiment(load_context_rag_dataset(dataset_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
