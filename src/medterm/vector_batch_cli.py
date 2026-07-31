from __future__ import annotations

import argparse
import json
from pathlib import Path

from medterm.audio_embeddings import SpeechContentEncoder, SpeechEncoderConfig
from medterm.vector_batch import build_vector_shards, finalize_vector_index


def vector_batch_command() -> None:
    parser = argparse.ArgumentParser(
        description="Encode active TTS references into resumable vector shards."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=5_000)
    parser.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pooling", choices=["mean", "statistics"], default="statistics")
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    encoder = SpeechContentEncoder(
        SpeechEncoderConfig(
            model_name=args.model,
            device=args.device,
            pooling=args.pooling,
            projection_checkpoint=args.projection,
        )
    )
    result = build_vector_shards(
        args.manifest,
        args.output,
        encoder,
        micro_batch_size=args.micro_batch_size,
        shard_size=args.shard_size,
        resume=args.resume,
    )
    print(result.model_dump_json(indent=2))


def finalize_index_command() -> None:
    parser = argparse.ArgumentParser(
        description="Validate vector shards and finalize one language index."
    )
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_vector_index(args.vectors, args.output), indent=2))
