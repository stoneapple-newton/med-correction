from __future__ import annotations

import argparse
import json
from pathlib import Path

from medterm.bulk import finalize_dictionary, import_rxnorm_release, register_source


def register_source_command() -> None:
    parser = argparse.ArgumentParser(description="Register an approved terminology source release.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=Path("data/builds"))
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--source-release", required=True)
    parser.add_argument(
        "--license-status",
        required=True,
        choices=["approved_for_local_processing", "approved_for_artifact_distribution"],
    )
    parser.add_argument("--license-approval-id", required=True)
    parser.add_argument("--license-owner", required=True)
    parser.add_argument("--license-evidence", required=True)
    parser.add_argument("--license-approval-date", required=True)
    args = parser.parse_args()
    manifest = register_source(**vars(args))
    print(manifest.model_dump_json(indent=2))


def import_terminology_command() -> None:
    parser = argparse.ArgumentParser(
        description="Stream an approved RxNorm release into deterministic JSONL shards."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--parse-checkpoint-rows", type=int, default=100_000)
    parser.add_argument("--max-concepts", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = import_rxnorm_release(
        args.manifest,
        args.output,
        concept_batch_size=args.batch_size,
        parse_checkpoint_rows=args.parse_checkpoint_rows,
        max_concepts=args.max_concepts,
        resume=args.resume,
    )
    print(summary.model_dump_json(indent=2))


def finalize_dictionary_command() -> None:
    parser = argparse.ArgumentParser(
        description="Validate import shards and finalize a dictionary artifact."
    )
    parser.add_argument("--import-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version")
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_dictionary(args.import_dir, args.output, version=args.version), indent=2
        )
    )
