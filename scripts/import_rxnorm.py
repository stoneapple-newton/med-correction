"""Convert an extracted RxNorm RXNCONSO.RRF file into the project's source CSV schema."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FIELDS = [
    "RXCUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "RXAUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
    "EMPTY",
]
PREFERRED_TTYS = {"IN", "PIN", "MIN", "SCD", "SBD", "BN", "PSN", "SY"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rxnconso", type=Path, help="Path to extracted RXNCONSO.RRF")
    parser.add_argument("output", type=Path, help="Destination CSV")
    parser.add_argument("--max-concepts", type=int)
    args = parser.parse_args()

    concepts: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.rxnconso.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for values in csv.reader(handle, delimiter="|"):
            if len(values) < len(FIELDS):
                continue
            row = dict(zip(FIELDS, values, strict=False))
            if (
                row["LAT"] != "ENG"
                or row["SUPPRESS"] != "N"
                or row["SAB"] != "RXNORM"
                or row["TTY"] not in PREFERRED_TTYS
            ):
                continue
            concepts[row["RXCUI"]].append(row)
            if args.max_concepts and len(concepts) >= args.max_concepts:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "concept_id",
            "term",
            "aliases",
            "language",
            "risk_tier",
            "prior",
            "provenance",
            "pronunciations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rxcui, rows in concepts.items():
            preferred = next((row for row in rows if row["ISPREF"] == "Y"), rows[0])
            aliases = list(
                dict.fromkeys(row["STR"] for row in rows if row["STR"] != preferred["STR"])
            )
            writer.writerow(
                {
                    "concept_id": f"RxCUI:{rxcui}",
                    "term": preferred["STR"],
                    "aliases": "|".join(aliases),
                    "language": "en",
                    "risk_tier": "standard",
                    "prior": "0.5",
                    "provenance": "RxNorm",
                    "pronunciations": "",
                }
            )
    print(f"Wrote {len(concepts)} concepts to {args.output}")


if __name__ == "__main__":
    main()
