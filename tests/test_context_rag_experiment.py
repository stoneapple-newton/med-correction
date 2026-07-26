from __future__ import annotations

from pathlib import Path

from medterm.context_rag_experiment import (
    DummyContextRetriever,
    load_context_rag_dataset,
    run_context_rag_experiment,
)

DATASET = Path(__file__).parents[1] / "data" / "context_rag_dummy_v1.json"


def test_context_rag_improves_sensitivity_and_selectability() -> None:
    report = run_context_rag_experiment(load_context_rag_dataset(DATASET))

    assert report["delta"]["correction_sensitivity_at_5"] > 0
    assert report["delta"]["selection_accuracy_at_1"] > 0
    assert report["delta"]["mean_gold_selection_margin"] > 0
    assert report["context_rag"]["negative_control_selectivity"] == 1.0
    assert report["safety"]["auto_commit_enabled"] is False


def test_context_retrieval_is_language_bounded_and_preserves_source() -> None:
    dataset = load_context_rag_dataset(DATASET)
    case = next(item for item in dataset.cases if item.case_id == "zh-context-rescue")
    source = case.text

    candidates = DummyContextRetriever(dataset.records).rank(case, mode="context_rag")

    assert candidates[0].concept_id == "RxCUI:6809"
    assert all(item.language == "zh" for item in candidates)
    assert case.text == source
    assert all(item.provenance.startswith("synthetic-") for item in candidates)


def test_context_signal_does_not_select_a_term_without_surface_support() -> None:
    dataset = load_context_rag_dataset(DATASET)
    case = next(item for item in dataset.cases if item.case_id == "en-negative-diabetes")

    assert DummyContextRetriever(dataset.records).rank(case, mode="context_rag") == []
