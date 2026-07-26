from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from medterm.audio_embeddings import (
    AudioEmbeddingIndex,
    SpeechContentEncoder,
    SpeechEncoderConfig,
    build_reference_index,
    load_reference_manifest,
    train_projection_head,
    validate_query_encoder,
)
from medterm.config import get_settings


def _ascii_console_json(payload: object) -> str:
    data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    return json.dumps(data, indent=2, ensure_ascii=True)


def _encoder_from_args(args: argparse.Namespace) -> SpeechContentEncoder:
    settings = get_settings()
    return SpeechContentEncoder(
        SpeechEncoderConfig(
            model_name=args.model or settings.audio_embedding_model,
            device=args.device or settings.audio_embedding_device,
            pooling=args.pooling or settings.audio_embedding_pooling,
            projection_checkpoint=(
                args.projection
                if args.projection is not None
                else settings.audio_embedding_projection_checkpoint
            ),
            max_seconds=settings.audio_embedding_max_seconds,
        )
    )


def _add_encoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pooling", choices=["mean", "statistics"])
    parser.add_argument("--projection", type=Path)


def index_command() -> None:
    parser = argparse.ArgumentParser(
        description="Build a review-only acoustic pronunciation index from a JSONL manifest"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    _add_encoder_arguments(parser)
    args = parser.parse_args()
    encoder = _encoder_from_args(args)
    index = build_reference_index(load_reference_manifest(args.manifest), encoder)
    index.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reference_count": len(index.metadata),
                "embedding_provenance": index.provenance,
                "auto_commit_enabled": False,
            },
            indent=2,
        )
    )


def search_command() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve acoustic candidates for phoneme/ASR/human review"
    )
    parser.add_argument("index", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--language")
    parser.add_argument("--country", action="append")
    parser.add_argument("--concept-type", default="medication")
    _add_encoder_arguments(parser)
    args = parser.parse_args()
    index = AudioEmbeddingIndex.load(args.index)
    encoder = _encoder_from_args(args)
    validate_query_encoder(index, encoder)
    filters: dict[str, str | list[str]] = {"concept_type": args.concept_type}
    if args.language:
        filters["language"] = args.language
    if args.country:
        filters["country"] = args.country
    query = encoder.encode(args.audio)
    validate_query_encoder(index, encoder)
    response = index.search_for_review(query, top_k=args.top_k, filters=filters)
    # Windows terminals may default to CP1252, while auditable pronunciation metadata contains
    # IPA. ASCII-safe JSON keeps the CLI machine-readable without dropping phonetic evidence.
    print(_ascii_console_json(response))


def train_command() -> None:
    parser = argparse.ArgumentParser(
        description="Train an experimental 256-D medical-term projection checkpoint"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    _add_encoder_arguments(parser)
    args = parser.parse_args()
    if args.projection is not None:
        parser.error("--projection cannot be used while training a new projection")
    encoder = _encoder_from_args(args)
    references = load_reference_manifest(args.manifest)
    features = [encoder.extract_pooled(reference.audio_path) for reference in references]
    summary = train_projection_head(
        np.stack(features),
        [reference.concept_id for reference in references],
        args.output,
        output_size=args.output_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        device=args.device or get_settings().audio_embedding_device,
        seed=args.seed,
        training_provenance={
            **encoder.provenance,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "training_summary": summary,
                "warning": (
                    "Experimental checkpoint only; held-out clinical validation is required."
                ),
                "auto_commit_enabled": False,
            },
            indent=2,
        )
    )
