"""Upload synthetic Asian cases to LangSmith and evaluate the bounded DeepSeek graph."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from langsmith import Client

from medterm.config import get_settings
from medterm.evaluation import load_jsonl, spans_overlap
from medterm.llm_experiment import build_correction_graph, graph_target
from medterm.normalization import normalize_term
from medterm.terminology import DictionaryIndex, build_artifact

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation_multilingual_v1.jsonl"
DEFAULT_DATASET_NAME = "term-trace-asian-synthetic-v1"


def asian_records(path: Path) -> list[Any]:
    return [record for record in load_jsonl(path) if "region:asia" in record.tags]


def upload_dataset(client: Client, path: Path, dataset_name: str) -> Any:
    records = asian_records(path)
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        existing = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
        if existing != len(records):
            raise RuntimeError(
                f"LangSmith dataset {dataset_name!r} exists with {existing} examples; "
                f"expected {len(records)}. Use a new versioned name instead of mutating it."
            )
        return dataset

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Project-generated synthetic Chinese, Japanese, Korean, and Hindi medical-term "
            "corruption cases. Contains no real patient data; requires native-speaker and "
            "clinical review and must not support clinical-performance claims."
        ),
        metadata={
            "dataset_version": "term-trace-multilingual-v1",
            "source_kind": "synthetic",
            "contains_real_patient_data": False,
            "languages": ["zh", "ja", "ko", "hi"],
        },
    )
    examples = [
        {
            "inputs": {"text": record.text, "locale": record.locale},
            "outputs": {
                "gold_spans": [span.model_dump() for span in record.gold_spans],
                "is_negative": not record.gold_spans,
            },
            "metadata": {
                "case_id": record.case_id,
                "tags": record.tags,
                "split": record.split,
                "source_kind": record.source_kind,
            },
        }
        for record in records
    ]
    for start in range(0, len(examples), 100):
        client.create_examples(dataset_id=dataset.id, examples=examples[start : start + 100])
    return dataset


def _balanced_examples(client: Client, dataset_name: str, limit: int | None) -> list[Any]:
    examples = list(client.list_examples(dataset_name=dataset_name))
    if limit is None or limit >= len(examples):
        return examples
    groups: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        groups[(example.inputs.get("locale") or "und").split("-")[0]].append(example)
    selected: list[Any] = []
    while len(selected) < limit and any(groups.values()):
        for language in sorted(groups):
            if groups[language] and len(selected) < limit:
                selected.append(groups[language].pop(0))
    return selected


def controlled_metrics(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> list[dict[str, Any]]:
    predictions = outputs.get("predictions", [])
    gold_spans = reference_outputs.get("gold_spans", [])
    metrics: list[dict[str, Any]] = [
        {
            "key": "auto_commit_safety",
            "score": float(outputs.get("auto_commit_enabled") is False),
        }
    ]
    if not gold_spans:
        metrics.append({"key": "selectivity", "score": float(not predictions)})
        return metrics

    gold = gold_spans[0]
    overlapping = [
        prediction
        for prediction in predictions
        if spans_overlap(
            prediction["char_start"],
            prediction["char_end"],
            gold["char_start"],
            gold["char_end"],
        )
    ]
    metrics.append({"key": "sensitivity", "score": float(bool(overlapping))})
    correct = bool(overlapping) and normalize_term(
        overlapping[0]["candidate_term"], gold.get("language") or "en"
    ) == normalize_term(gold.get("gold_term") or "", gold.get("language") or "en")
    metrics.append({"key": "correction_at_1", "score": float(correct)})
    return metrics


def run_evaluation(
    client: Client,
    dataset_name: str,
    *,
    limit: int | None,
    max_concurrency: int,
) -> Any:
    settings = get_settings()
    artifact = build_artifact(settings.source_terms_path, settings.artifact_path)
    graph = build_correction_graph(DictionaryIndex(artifact))
    examples = _balanced_examples(client, dataset_name, limit)
    return client.evaluate(
        graph_target(graph),
        data=examples,
        evaluators=[controlled_metrics],
        experiment_prefix="term-trace-deepseek-v4-flash-langgraph",
        description=(
            "Synthetic-only bounded dictionary selection experiment; not wired to clinical API"
        ),
        metadata={
            "model": "deepseek-v4-flash",
            "framework": "langgraph",
            "auto_commit_enabled": False,
            "privacy_scope": "synthetic_only",
        },
        max_concurrency=max_concurrency,
        upload_results=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["upload", "evaluate"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--limit",
        type=int,
        default=56,
        help="balanced evaluation sample; use 0 for all examples",
    )
    parser.add_argument("--max-concurrency", type=int, default=2)
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY is required")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "term-trace-development")
    client = Client()
    dataset = upload_dataset(client, args.dataset, args.name)
    print(f"LangSmith dataset ready: {dataset.name} ({dataset.id})")
    if args.action == "evaluate":
        result = run_evaluation(
            client,
            args.name,
            limit=None if args.limit == 0 else args.limit,
            max_concurrency=args.max_concurrency,
        )
        print(result)


if __name__ == "__main__":
    main()
