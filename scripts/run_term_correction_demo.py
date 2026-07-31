from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from medterm.config import get_settings
from medterm.language import LanguageIdentifier
from medterm.matcher import MedicalTermMatcher
from medterm.models import MatchRequest
from medterm.terminology import DictionaryIndex, build_artifact, load_artifact


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_demo(manifest_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = get_settings()
    artifact_path = _resolve(root, str(settings.artifact_path))
    source_terms_path = _resolve(root, str(settings.source_terms_path))
    if not artifact_path.exists():
        build_artifact(source_terms_path, artifact_path)

    matcher = MedicalTermMatcher(
        DictionaryIndex(load_artifact(artifact_path)),
        LanguageIdentifier(settings.fasttext_model_path),
        review_threshold=settings.review_threshold,
        suspect_threshold=settings.suspect_threshold,
        force_review_margin=settings.force_review_margin,
        max_span_tokens=settings.max_span_tokens,
    )

    results: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        transcript_path = _resolve(root, case["mistranscribed_transcript_file"])
        audio_path = _resolve(root, case["audio_clip"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing demo audio clip: {audio_path}")
        audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        audio_hash_matches = audio_sha256 == case["audio_clip_sha256"]
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        response = matcher.match(
            MatchRequest(text=transcript, locale=case["locale"], top_k=5)
        )
        matching_spans = [
            span
            for span in response.spans
            if span.candidates
            and span.candidates[0].concept_id == case["expected_concept_id"]
            and span.candidates[0].term == case["expected_term"]
        ]
        passed = (
            not response.auto_commit_enabled
            and audio_hash_matches
            and bool(matching_spans)
            and all(span.decision == "review" for span in matching_spans)
        )
        results.append(
            {
                "id": case["id"],
                "input_transcript": transcript,
                "expected_term": case["expected_term"],
                "audio_sha256": audio_sha256,
                "audio_hash_matches": audio_hash_matches,
                "detected_spans": [
                    {
                        "span_text": span.span_text,
                        "char_start": span.char_start,
                        "char_end": span.char_end,
                        "decision": span.decision,
                        "top_candidate": (
                            {
                                "concept_id": span.candidates[0].concept_id,
                                "term": span.candidates[0].term,
                                "score": span.candidates[0].score,
                                "score_breakdown": span.candidates[0].score_breakdown.model_dump(),
                            }
                            if span.candidates
                            else None
                        ),
                    }
                    for span in response.spans
                ],
                "auto_commit_enabled": response.auto_commit_enabled,
                "passed": passed,
            }
        )

    report = {
        "manifest": str(manifest_path),
        "artifact_version": matcher.index.artifact.artifact_version,
        "all_passed": all(item["passed"] for item in results),
        "results": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the English and Chinese audio transcript correction demo."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("demo/term-correction/manifest.json"),
    )
    args = parser.parse_args()
    report = run_demo(args.manifest.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
