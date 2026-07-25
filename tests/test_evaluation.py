from medterm.evaluation import evaluate_records
from medterm.matcher import MedicalTermMatcher


def test_evaluation_reports_retrieval_and_workload_metrics(
    matcher: MedicalTermMatcher,
) -> None:
    records = [
        {
            "text": "met for men",
            "asr_hypothesis": "met for men",
            "reference_transcript": "metformin",
            "locale": "en-US",
            "gold_spans": [{"char_start": 0, "char_end": 11, "concept_id": "RxCUI:6809"}],
        },
        {"text": "Continue metformin.", "locale": "en-US", "gold_spans": []},
    ]
    report = evaluate_records(matcher, records)
    assert report["records"] == 2
    assert report["gold_spans"] == 1
    assert report["recall_at_1"] == 1.0
    assert report["mrr"] == 1.0
    assert report["review_rate"] == 1.0
    assert report["wer"] > 0
    assert report["cer"] > 0
    assert report["auto_commit_count"] == 0
