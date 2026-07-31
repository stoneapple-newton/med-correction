from __future__ import annotations

import argparse
import json
from pathlib import Path

from medterm.audio_embeddings import SpeechContentEncoder, SpeechEncoderConfig
from medterm.bulk import atomic_write_text
from medterm.medrag_batch import (
    batch_paths,
    build_tts_batch,
    export_sqlite_batch,
    reconcile_batch,
)
from medterm.tts import KokoroConfig, KokoroSynthesizer
from medterm.vector_batch import build_vector_shards, finalize_vector_index


def batch_command() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic, resumable med-rag terminology batch."
    )
    parser.add_argument("--stage", choices=["export", "tts", "vectors", "all"], required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--expected-sqlite-sha256", required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-language", choices=["en", "zh-Hans"], required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--limit", type=int, default=5_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--tts-model", default="hexgrad/Kokoro-82M")
    parser.add_argument("--tts-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--voice")
    parser.add_argument("--language-code")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--embedding-model", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--pooling", choices=["mean", "statistics"], default="statistics")
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=5_000)
    args = parser.parse_args()

    language = "zh" if args.source_language == "zh-Hans" else args.source_language
    defaults = {
        "en": ("a", "af_heart"),
        "zh": ("z", "zf_xiaobei"),
    }
    language_code, voice = defaults[language]
    paths = batch_paths(args.build_root, language, args.batch_id)
    result: dict[str, object] = {}

    if args.stage in {"export", "all"}:
        selection, artifact = export_sqlite_batch(
            args.sqlite,
            args.build_root,
            release_id=args.release_id,
            batch_id=args.batch_id,
            source_language=args.source_language,
            offset=args.offset,
            limit=args.limit,
            expected_sqlite_sha256=args.expected_sqlite_sha256,
        )
        result["selection"] = selection.model_dump(mode="json")
        result["dictionary_entries"] = len(artifact.entries)

    if args.stage in {"tts", "all"}:
        checkpoint = build_tts_batch(
            paths["dictionary"],
            paths["audio"],
            paths["references"],
            paths["failures"],
            paths["tts_checkpoint"],
            KokoroSynthesizer(
                KokoroConfig(
                    model_id=args.tts_model,
                    voice=args.voice or voice,
                    language_code=args.language_code or language_code,
                    device=args.tts_device,
                    speed=args.speed,
                )
            ),
            language=language,
            checkpoint_every=args.checkpoint_every,
        )
        result["tts"] = checkpoint.model_dump(mode="json")

    if args.stage in {"vectors", "all"}:
        encoder = SpeechContentEncoder(
            SpeechEncoderConfig(
                model_name=args.embedding_model,
                device=args.embedding_device,
                pooling=args.pooling,
                projection_checkpoint=args.projection,
            )
        )
        vector_checkpoint = build_vector_shards(
            paths["references"],
            paths["vectors"],
            encoder,
            micro_batch_size=args.micro_batch_size,
            shard_size=args.shard_size,
            resume=(paths["vectors"] / "checkpoint.json").is_file(),
        )
        index_summary = finalize_vector_index(paths["vectors"], paths["index"])
        reconciliation = reconcile_batch(
            paths["references"],
            paths["failures"],
            paths["tts_checkpoint"],
            paths["index"].with_suffix(".summary.json"),
        )
        atomic_write_text(
            paths["reconciliation"],
            json.dumps(reconciliation, indent=2, sort_keys=True) + "\n",
        )
        result["vectors"] = vector_checkpoint.model_dump(mode="json")
        result["index"] = index_summary
        result["reconciliation"] = reconciliation

    result["decision"] = "review"
    result["auto_commit_enabled"] = False
    print(json.dumps(result, indent=2, ensure_ascii=False))
