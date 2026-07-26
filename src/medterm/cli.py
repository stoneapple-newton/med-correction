from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import uvicorn

from medterm.config import get_settings
from medterm.evaluation import evaluate_records, load_jsonl
from medterm.language import LanguageIdentifier
from medterm.matcher import MedicalTermMatcher
from medterm.terminology import DictionaryIndex, build_artifact, load_artifact


def build_command() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned medical dictionary artifact")
    parser.add_argument("--source", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--version", type=str)
    args = parser.parse_args()
    settings = get_settings()
    artifact = build_artifact(
        settings.source_terms_path
        if not args.source
        else settings.source_terms_path.__class__(args.source),
        settings.artifact_path
        if not args.output
        else settings.artifact_path.__class__(args.output),
        args.version,
    )
    print(f"Built {artifact.artifact_version} with {len(artifact.entries)} entries")


def api_command() -> None:
    parser = argparse.ArgumentParser(description="Run the Medical Term Review API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("medterm.api:app", host=args.host, port=args.port, reload=args.reload)


def evaluate_command() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a JSONL medical-term dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, help="write the complete JSON report to this path")
    args = parser.parse_args()
    settings = get_settings()
    if not settings.artifact_path.exists():
        build_artifact(settings.source_terms_path, settings.artifact_path)
    matcher = MedicalTermMatcher(
        DictionaryIndex(load_artifact(settings.artifact_path)),
        LanguageIdentifier(settings.fasttext_model_path),
        review_threshold=settings.review_threshold,
        suspect_threshold=settings.suspect_threshold,
        force_review_margin=settings.force_review_margin,
        max_span_tokens=settings.max_span_tokens,
    )
    report = evaluate_records(
        matcher,
        load_jsonl(args.dataset),
        target_sensitivity=settings.target_sensitivity,
        target_selectivity=settings.target_selectivity,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    report["reproducibility"] = {
        "report_schema_version": 3,
        "code_commit": commit.stdout.strip() if commit.returncode == 0 else "unavailable",
        "code_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "dictionary_version": matcher.index.artifact.artifact_version,
        "dictionary_source_sha256": matcher.index.artifact.source_sha256,
        "retrieval_index_version": matcher.index.artifact.artifact_version,
        "fasttext_available": matcher.language_identifier.model is not None,
        "panphon_available": matcher.phonetic._distance is not None,
        "dataset_path": args.dataset.as_posix(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "evaluation_parameters": {
            "review_threshold": settings.review_threshold,
            "suspect_threshold": settings.suspect_threshold,
            "force_review_margin": settings.force_review_margin,
            "max_span_tokens": settings.max_span_tokens,
            "target_sensitivity": settings.target_sensitivity,
            "target_selectivity": settings.target_selectivity,
        },
    }
    content = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8", newline="\n")
    else:
        print(content, end="")
