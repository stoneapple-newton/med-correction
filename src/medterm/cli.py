from __future__ import annotations

import argparse
import json
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
    print(json.dumps(evaluate_records(matcher, load_jsonl(args.dataset)), indent=2))
