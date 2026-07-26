from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from medterm.audio_embeddings import (
    SpeechContentEncoder,
    SpeechEncoderConfig,
    build_reference_index,
)
from medterm.config import get_settings
from medterm.terminology import build_artifact, load_artifact
from medterm.tts import (
    KokoroConfig,
    KokoroSynthesizer,
    build_synthetic_references,
    write_reference_manifest,
)


def build_command() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize controlled medical terms with local Kokoro-82M and build "
            "a review-only sound-vector index"
        )
    )
    parser.add_argument("--artifact", type=Path, default=settings.artifact_path)
    parser.add_argument("--source", type=Path, default=settings.source_terms_path)
    parser.add_argument("--audio-dir", type=Path, default=settings.tts_output_dir)
    parser.add_argument("--manifest", type=Path, default=settings.tts_manifest_path)
    parser.add_argument("--index", type=Path, default=settings.tts_index_path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tts-model", default=settings.tts_model_id)
    parser.add_argument("--voice", default=settings.tts_voice)
    parser.add_argument("--language-code", default=settings.tts_language_code)
    parser.add_argument(
        "--tts-device",
        choices=["auto", "cpu", "cuda"],
        default=settings.tts_device,
    )
    parser.add_argument("--speed", type=float, default=settings.tts_speed)
    parser.add_argument("--embedding-model", default=settings.audio_embedding_model)
    parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default=settings.audio_embedding_device,
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "statistics"],
        default=settings.audio_embedding_pooling,
    )
    parser.add_argument(
        "--projection",
        type=Path,
        default=settings.audio_embedding_projection_checkpoint,
    )
    args = parser.parse_args()

    if not args.artifact.is_file():
        build_artifact(args.source, args.artifact)
    artifact = load_artifact(args.artifact)
    synthesizer = KokoroSynthesizer(
        KokoroConfig(
            model_id=args.tts_model,
            voice=args.voice,
            language_code=args.language_code,
            device=args.tts_device,
            speed=args.speed,
        )
    )
    references = build_synthetic_references(
        artifact,
        synthesizer,
        args.audio_dir,
        language=args.language,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    write_reference_manifest(references, args.manifest)

    encoder = SpeechContentEncoder(
        SpeechEncoderConfig(
            model_name=args.embedding_model,
            device=args.embedding_device,
            pooling=args.pooling,
            projection_checkpoint=args.projection,
            max_seconds=settings.audio_embedding_max_seconds,
        )
    )
    index = build_reference_index(references, encoder)
    index.provenance["reference_audio"] = {
        **synthesizer.provenance,
        "synthetic": True,
        "terminology_version": artifact.artifact_version,
        "terminology_source_sha256": artifact.source_sha256,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    index.save(args.index)
    print(
        json.dumps(
            {
                "audio_directory": str(args.audio_dir),
                "manifest": str(args.manifest),
                "index": str(args.index),
                "reference_count": len(references),
                "tts_provenance": synthesizer.provenance,
                "embedding_provenance": index.provenance,
                "decision": "review",
                "auto_commit_enabled": False,
                "warning": (
                    "Synthetic TTS references are development aids and require held-out "
                    "evaluation against validated human recordings."
                ),
            },
            indent=2,
        )
    )
