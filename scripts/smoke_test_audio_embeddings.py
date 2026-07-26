from __future__ import annotations

import argparse
import json
from pathlib import Path

from medterm.audio_embeddings import (
    ReferencePronunciation,
    SpeechContentEncoder,
    SpeechEncoderConfig,
    build_reference_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small, non-clinical speech-embedding retrieval smoke test"
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("query", type=Path)
    parser.add_argument("different", type=Path)
    parser.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pooling", default="statistics", choices=["mean", "statistics"])
    args = parser.parse_args()

    encoder = SpeechContentEncoder(
        SpeechEncoderConfig(
            model_name=args.model,
            device=args.device,
            pooling=args.pooling,
        )
    )
    reference_vector = encoder.encode(args.reference)
    query_vector = encoder.encode(args.query)
    different_vector = encoder.encode(args.different)
    index = build_reference_index(
        [
            ReferencePronunciation(
                reference_id="synthetic-metformin",
                audio_path=args.reference,
                concept_id="LOCAL:METFORMIN",
                term="metformin",
                language="en",
                country="US",
                pronunciation_type="synthetic",
                speaker_accent="synthetic-en-US",
                source="local_smoke_test",
            ),
            ReferencePronunciation(
                reference_id="synthetic-metoprolol",
                audio_path=args.different,
                concept_id="LOCAL:METOPROLOL",
                term="metoprolol",
                language="en",
                country="US",
                pronunciation_type="synthetic",
                speaker_accent="synthetic-en-US",
                source="local_smoke_test",
            ),
        ],
        encoder,
    )
    response = index.search_for_review(query_vector, filters={"language": "en"})
    print(
        json.dumps(
            {
                "embedding_provenance": encoder.provenance,
                "reference_query_cosine": float(reference_vector @ query_vector),
                "different_query_cosine": float(different_vector @ query_vector),
                "search_response": response.model_dump(mode="json"),
                "interpretation": (
                    "execution smoke test only; synthetic examples do not measure medical-term "
                    "accuracy or clinical performance"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
