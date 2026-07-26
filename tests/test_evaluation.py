import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from medterm.evaluation import evaluate_records, load_jsonl
from medterm.matcher import MedicalTermMatcher
from medterm.normalization import detect_scripts


def test_evaluation_reports_retrieval_and_workload_metrics(
    matcher: MedicalTermMatcher,
) -> None:
    records = [
        {
            "text": "met for men",
            "asr_hypothesis": "met for men",
            "reference_transcript": "metformin",
            "locale": "en-US",
            "gold_spans": [
                {
                    "char_start": 0,
                    "char_end": 11,
                    "gold_term": "metformin",
                    "concept_id": "RxCUI:6809",
                }
            ],
        },
        {"text": "Continue metformin.", "locale": "en-US", "gold_spans": []},
    ]
    report = evaluate_records(matcher, records)
    assert report["records"] == 2
    assert report["gold_spans"] == 1
    assert report["recall_at_1"] == 1.0
    assert report["mrr"] == 1.0
    assert report["correction"]["accuracy_at_1"] == 1.0
    assert report["coverage"] == {
        "gold": 1,
        "terminology_coverage": 1.0,
        "candidate_coverage_at_5": 1.0,
        "surface_candidate_coverage_at_5": 1.0,
        "oracle_rerank_accuracy_at_1": 1.0,
    }
    assert report["concept_identity"]["status"] == "applicable"
    assert report["review_rate"] == 1.0
    assert report["wer"] > 0
    assert report["cer"] > 0
    assert report["auto_commit_count"] == 0
    assert report["evaluation_scope"] == {
        "retrieval_backend": "medterm.terminology.DictionaryIndex",
        "context_rag_connected": False,
        "result_classification": "dictionary_baseline",
        "rag_uplift_measurable": False,
        "next_required_step": (
            "connect the evaluator to the versioned RAG retrieval backend and rerun the same "
            "dataset before reporting uplift"
        ),
    }
    assert report["targets"]["sensitivity"] == {
        "target": 0.95,
        "observed": 1.0,
        "met": True,
    }
    assert report["targets"]["selectivity"]["observed"] == 1.0


def test_versioned_detection_dataset_is_valid_and_reports_slices(
    matcher: MedicalTermMatcher,
) -> None:
    dataset_path = Path(__file__).parents[1] / "data" / "evaluation_detection_v1.jsonl"
    records = load_jsonl(dataset_path)

    report = evaluate_records(matcher, records)

    assert len(records) == 14
    assert report["dataset_versions"] == ["term-trace-detection-v1"]
    assert report["detection"]["gold"] == 12
    assert report["detection"]["overlap_recall"] == 1.0
    assert report["detection"]["exact_recall"] < report["detection"]["overlap_recall"]
    assert report["slices"]["negative"]["predicted"] == 0
    assert report["slices"]["mixed_script"]["gold"] == 1
    assert report["correction_slices"]["positive"]["gold"] == 12
    assert report["case_errors"]
    assert report["auto_commit_count"] == 0


def test_load_jsonl_rejects_gold_text_that_disagrees_with_offsets(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(
        '{"case_id":"bad-offset","text":"met for men","gold_spans":'
        '[{"char_start":0,"char_end":3,"span_text":"metformin"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match text slice"):
        load_jsonl(dataset)


def test_load_jsonl_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "duplicates.jsonl"
    record = '{"case_id":"same","text":"synthetic","gold_spans":[]}\n'
    dataset.write_text(record + record, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_jsonl(dataset)


def test_overlap_matching_does_not_credit_one_prediction_twice() -> None:
    prediction = SimpleNamespace(
        char_start=0,
        char_end=7,
        span_text="aaa bbb",
        candidates=[],
        decision="abstain",
    )
    matcher = SimpleNamespace(match=lambda request: SimpleNamespace(spans=[prediction]))
    records = [
        {
            "case_id": "one-to-one",
            "text": "aaa bbb",
            "gold_spans": [
                {"char_start": 0, "char_end": 3, "span_text": "aaa"},
                {"char_start": 4, "char_end": 7, "span_text": "bbb"},
            ],
        }
    ]

    report = evaluate_records(matcher, records)

    assert report["detection"]["overlap_true_positives"] == 1
    assert report["detection"]["overlap_false_negatives"] == 1
    assert len(report["case_errors"][0]["false_negatives"]) == 1


def test_multilingual_dataset_has_balanced_language_and_condition_coverage() -> None:
    project_root = Path(__file__).parents[1]
    dataset_path = project_root / "data" / "evaluation_multilingual_v1.jsonl"
    manifest_path = project_root / "data" / "evaluation_multilingual_v1.manifest.json"
    records = load_jsonl(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    language_counts = Counter(record.locale.split("-")[0] for record in records)
    condition_counts = Counter(record.tags[-1] for record in records)

    assert len(records) == 770
    assert set(language_counts) == {"en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "hi"}
    assert set(language_counts.values()) == {70}
    assert condition_counts == {
        "character_deletion": 110,
        "character_transposition": 110,
        "word_boundary": 110,
        "character_substitution": 110,
        "low_confidence": 110,
        "script_confusion": 110,
        "correct_term_control": 110,
    }
    assert manifest["record_count"] == len(records)
    assert manifest["evaluation_contract_version"] == 2
    assert manifest["contains_real_patient_data"] is False
    assert manifest["contains_third_party_dataset_rows"] is False
    assert all(
        record.text[span.char_start : span.char_end] == span.span_text
        for record in records
        for span in record.gold_spans
    )
    assert all(
        span.benchmark_key
        and span.canonical_term
        and span.source_vocabulary
        and span.authority_concept_id
        for record in records
        for span in record.gold_spans
    )
    assert all(
        len(detect_scripts(span.span_text or "")) > 1
        for record in records
        if "script_confusion" in record.tags
        for span in record.gold_spans
    )

    subprocess.run(
        [sys.executable, "scripts/build_multilingual_evaluation.py", "--check"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_incompatible_benchmark_id_namespace_is_explicitly_not_applicable(
    matcher: MedicalTermMatcher,
) -> None:
    report = evaluate_records(
        matcher,
        [
            {
                "case_id": "synthetic-id",
                "text": "met for men",
                "locale": "en-US",
                "gold_spans": [
                    {
                        "char_start": 0,
                        "char_end": 11,
                        "canonical_term": "metformin",
                        "authority_concept_id": "SYNTH:en:metformin",
                    }
                ],
            }
        ],
    )

    assert report["concept_identity"] == {
        "status": "not_applicable",
        "reason": "no gold authority IDs share a dictionary namespace",
        "gold": 0,
        "excluded_gold_spans": 1,
        "accuracy_at_1": None,
        "recall_at_5": None,
        "mrr": None,
    }
    assert report["recall_at_1"] is None


def test_concept_identity_uses_original_candidate_rank() -> None:
    candidates = [
        SimpleNamespace(concept_id="LOCAL:other", term="other", language="en"),
        SimpleNamespace(concept_id="RxCUI:6809", term="metformin", language="en"),
    ]
    prediction = SimpleNamespace(
        char_start=0,
        char_end=11,
        span_text="met for men",
        candidates=candidates,
        decision="review",
    )
    entry = SimpleNamespace(
        concept_id="RxCUI:6809",
        language="en",
        normalized_aliases=["metformin"],
        source_vocabulary="RxNorm",
        source_release="test",
        licensing_status="test-only",
        review_status="reviewed",
    )
    matcher = SimpleNamespace(
        index=SimpleNamespace(artifact=SimpleNamespace(entries=[entry])),
        match=lambda request: SimpleNamespace(spans=[prediction]),
    )

    report = evaluate_records(
        matcher,
        [
            {
                "case_id": "rank-two",
                "text": "met for men",
                "locale": "en-US",
                "gold_spans": [
                    {
                        "char_start": 0,
                        "char_end": 11,
                        "canonical_term": "metformin",
                        "authority_concept_id": "RxCUI:6809",
                    }
                ],
            }
        ],
    )

    assert report["concept_identity"]["accuracy_at_1"] == 0.0
    assert report["concept_identity"]["recall_at_5"] == 1.0
    assert report["concept_identity"]["mrr"] == 0.5
